"""Local weather genre — keyless Open-Meteo (geocode once, then forecast).

One tool, `weather(location?)`, matching the Mac host's.  The default place
comes from tools.local.weather.location; a spoken location overrides it.
Geocoding results are cached in-process so the usual case is one HTTP call.
"""
import json
import urllib.parse

# WMO weather interpretation codes → words (the full table, coarsened).
_WMO = {0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
        45: "fog", 48: "freezing fog", 51: "light drizzle", 53: "drizzle",
        55: "heavy drizzle", 56: "freezing drizzle", 57: "freezing drizzle",
        61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain",
        67: "freezing rain", 71: "light snow", 73: "snow", 75: "heavy snow",
        77: "snow grains", 80: "light showers", 81: "showers", 82: "heavy showers",
        85: "snow showers", 86: "snow showers", 95: "thunderstorm",
        96: "thunderstorm with hail", 99: "thunderstorm with hail"}

_geocache: dict = {}


def _json(fetch, url, purpose):
    status, body = fetch(url, timeout=15.0, purpose=purpose)
    if status != 200:
        raise RuntimeError(f"HTTP {status}")
    return json.loads(body.decode("utf-8", errors="replace"))


def _geocode(fetch, place: str) -> dict:
    key = place.strip().lower()
    if key in _geocache:
        return _geocache[key]
    q = urllib.parse.quote(place.strip())
    data = _json(fetch, "https://geocoding-api.open-meteo.com/v1/search"
                        f"?name={q}&count=1&language=en&format=json", "local_weather")
    hits = data.get("results") or []
    if not hits:
        raise RuntimeError(f"no such place: {place}")
    hit = {"lat": hits[0]["latitude"], "lon": hits[0]["longitude"],
           "name": hits[0].get("name", place),
           "country": hits[0].get("country", "")}
    _geocache[key] = hit
    return hit


def _wmo(code) -> str:
    try:
        return _WMO.get(int(code), "changeable")
    except Exception:
        return "changeable"


def forecast_prose(fetch, place: str) -> str:
    loc = _geocode(fetch, place)
    data = _json(fetch, "https://api.open-meteo.com/v1/forecast"
                        f"?latitude={loc['lat']}&longitude={loc['lon']}"
                        "&current=temperature_2m,apparent_temperature,precipitation,"
                        "weather_code,wind_speed_10m,relative_humidity_2m"
                        "&daily=temperature_2m_max,temperature_2m_min,"
                        "precipitation_probability_max,weather_code"
                        "&forecast_days=2&timezone=auto", "local_weather")
    cur, day = data.get("current") or {}, data.get("daily") or {}
    where = loc["name"] + (f", {loc['country']}" if loc.get("country") else "")
    lines = [f"Weather for {where}:"]
    if cur:
        feels = cur.get("apparent_temperature")
        feels_txt = f" (feels like {feels:.0f}°)" if isinstance(feels, (int, float)) else ""
        lines.append(f"- Now: {_wmo(cur.get('weather_code'))}, "
                     f"{cur.get('temperature_2m', '?')}°C{feels_txt}, "
                     f"wind {cur.get('wind_speed_10m', '?')} km/h, "
                     f"humidity {cur.get('relative_humidity_2m', '?')}%")
    highs, lows = day.get("temperature_2m_max") or [], day.get("temperature_2m_min") or []
    rains, codes = day.get("precipitation_probability_max") or [], day.get("weather_code") or []
    for i, label in enumerate(("Today", "Tomorrow")):
        if i < len(highs):
            rain = f", {rains[i]}% chance of rain" if i < len(rains) and rains[i] is not None else ""
            code = _wmo(codes[i]) if i < len(codes) else ""
            lines.append(f"- {label}: {code}, {lows[i]:.0f}–{highs[i]:.0f}°C{rain}")
    return "\n".join(lines)


def tools(gcfg: dict, env: dict) -> list:
    default_place = str(gcfg.get("location") or "").strip()

    def weather(args):
        place = str(args.get("location") or "").strip() or default_place
        if not place:
            return {"ok": False, "error": "no location configured — set a default "
                    "town under Settings → Local tools → Weather, or name one"}
        return forecast_prose(env["fetch"], place)

    return [
        ({"name": "weather",
          "description": "Current weather and today/tomorrow forecast"
                         + (f" (defaults to {default_place})" if default_place else "")
                         + ". Give a location to ask about somewhere else.",
          "parameters": {"type": "object", "properties": {
              "location": {"type": "string", "description": "town or city name"}}}},
         weather),
    ]


def probe(gcfg: dict, env: dict) -> dict:
    place = str(gcfg.get("location") or "").strip()
    if not place:
        return {"ok": False, "detail": "set a default town or city first"}
    return {"ok": True, "detail": forecast_prose(env["fetch"], place)}
