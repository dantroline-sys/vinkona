# Vinkona Notifications — contract for the Flutter client (and the calendar tool)

Vinkona can push **notifications** — mainly reminders for upcoming appointments, plus
ad-hoc reminders you ask for ("remind me to call mum at 5"). The cascade server queues
them; the **client polls** for ones whose time has come and shows them (a bell, a
badge, an OS notification — the client's choice).

Notifications are **per-profile** (they live in the active profile's memory DB).

## Client side (Flutter)

Poll the cascade server (same host/port as the WebSocket, e.g. `https://<box>:8998`):

### `GET /api/notifications`
Returns notifications that are now due **and marks them delivered** (you've got them):
```json
{ "notifications": [
  { "id": 12, "text": "Dentist — today at 15:00", "kind": "appointment",
    "deliver_at": 1750000000.0, "created_at": 1749990000.0, "source": "calendar" }
] }
```
- Poll on whatever cadence suits (e.g. every 30–60 s while the app is open).
- `kind` is `appointment` (calendar-derived) or `reminder` (the user asked for it).
- Show them however you like; this is just the text to surface.

### `GET /api/notifications?peek=1`
Same list, but **does not** mark them delivered — use this to drive a bell badge /
unread count without consuming them, then do a normal `GET` when the user opens the
bell.

> Delivery is "mark on fetch": once a normal `GET` returns a notification it won't come
> back. If you want client-side read/unread state beyond that, track it in the app.

There's no push channel — polling keeps it simple and works when no voice session is
open. (If you'd prefer a WebSocket push later, we can add one.)

## Calendar tool side (Mac tool host)

The scheduler reminds you about events by calling your existing read tool
(`notifications.calendar_tool`, default `calendar_range`) and reading its result. For
reminders to work, that tool's result must be JSON with event start times:

```json
{ "ok": true, "result": "{\"events\": [
  {\"id\": \"ABC123\", \"title\": \"Dentist\", \"start\": \"2026-06-23T15:00\", \"end\": \"2026-06-23T16:00\"}
]}" }
```
- `result` is a string (as for every tool); the JSON lives inside it.
- A bare JSON array (no `events` wrapper) is also accepted.
- `start` is ISO-8601 in local time (a trailing `Z` is fine). `id` lets us de-duplicate
  reminders; if absent we key on title+start.

The scheduler creates a reminder at each lead time in `notifications.lead_times_min`
(default 1 day and 1 hour before). It re-scans every `poll_interval_s` but won't
duplicate a reminder it already queued.

## Turning it on
In `config.json`: set `notifications.enabled: true` (and tune `lead_times_min`). For
calendar-derived reminders the tool host must be reachable (Tier-2 tools enabled). The
`remind_me` tool works as soon as notifications are enabled, no calendar needed.

One-source-of-truth note: keep the merge of your other calendars into the single
"Vinkona" calendar on the Mac side (see MAC_TOOLS.md). The scheduler reads whatever
`calendar_range` returns, so if that already spans everything, reminders cover it all.
