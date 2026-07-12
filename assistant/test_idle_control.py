"""Unit tests for idle_control — schedule/override math, no deps."""
import idle_control as ic


def check(label, cond):
    print(("  ok  " if cond else "  FAIL ") + label)
    if not cond:
        check.failed += 1
check.failed = 0


def main():
    # parse_hm
    check("parse 09:30", ic.parse_hm("09:30") == 570)
    check("parse 00:00", ic.parse_hm("00:00") == 0)
    check("reject 24:00", ic.parse_hm("24:00") is None)
    check("reject junk", ic.parse_hm("nope") is None)

    # simple window
    check("inside 10-14 at 12:00", ic.in_window(720, "10:00", "14:00"))
    check("edge start inclusive", ic.in_window(600, "10:00", "14:00"))
    check("edge end exclusive", not ic.in_window(840, "10:00", "14:00"))
    check("before window", not ic.in_window(540, "10:00", "14:00"))

    # midnight-wrapping window 22:00-06:00
    check("wrap: 23:00 inside", ic.in_window(1380, "22:00", "06:00"))
    check("wrap: 03:00 inside", ic.in_window(180, "22:00", "06:00"))
    check("wrap: 12:00 outside", not ic.in_window(720, "22:00", "06:00"))
    check("empty window (s==e)", not ic.in_window(600, "10:00", "10:00"))

    qh = [{"start": "10:00", "end": "14:00"}]

    # override precedence
    check("paused forces suppressed", ic.is_suppressed("paused", 0, qh))
    check("active forces not-suppressed even in window",
          not ic.is_suppressed("active", 720, qh))
    check("auto in window → suppressed", ic.is_suppressed("auto", 720, qh))
    check("auto out of window → not suppressed", not ic.is_suppressed("", 900, qh))
    check("unknown override treated as auto", ic.is_suppressed("wat", 720, qh))

    # multi-window
    qh2 = [{"start": "10:00", "end": "12:00"}, {"start": "22:00", "end": "06:00"}]
    check("multi: 11:00 suppressed", ic.is_suppressed("auto", 660, qh2))
    check("multi: 23:30 suppressed", ic.is_suppressed("auto", 1410, qh2))
    check("multi: 15:00 free", not ic.is_suppressed("auto", 900, qh2))

    # describe
    d = ic.describe("auto", 720, qh)
    check("describe suppressed in window", d["suppressed"] and d["reason"] == "quiet hours")
    check("describe next boundary = 14:00 (840)", d["next_change_min"] == 840)
    d2 = ic.describe("paused", 900, qh)
    check("describe paused reason", d2["override"] == "paused" and d2["suppressed"])
    d3 = ic.describe("auto", 900, qh)
    check("describe active-now next = 10:00 (600)", d3["next_change_min"] == 600)
    d4 = ic.describe("auto", 720, [])
    check("no windows → no next_change", d4["next_change_min"] is None and not d4["suppressed"])

    print()
    if check.failed:
        print(f"{check.failed} FAILED")
        raise SystemExit(1)
    print("All idle_control tests passed")


if __name__ == "__main__":
    main()
