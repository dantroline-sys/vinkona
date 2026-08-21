"""Keep egress.toml's managed block in step with tools.local's configuration.

Deny-by-default stays true: the local toolset gets exactly the destinations
the user configured — the fixed keyless API hosts for an enabled research /
weather genre, the hosts of the feed URLs they listed, the host of the CalDAV
address they entered — and nothing when a genre is off.  The rules live inside
a clearly marked managed block that is regenerated as a whole; everything the
user wrote by hand outside the markers is preserved byte-for-byte.

IMAP is not HTTP and does not pass through the broker; when the mail genre is
on, the block says so in a comment naming the configured servers, so reading
egress.toml still tells the whole outbound story.
"""
import urllib.parse

BEGIN = "# ── BEGIN local_tools managed rules (regenerated from tools.local — do not edit inside) ──"
END = "# ── END local_tools managed rules ──"

RESEARCH_HOSTS = [
    "www.ebi.ac.uk",             # Europe PMC
    "api.openalex.org",          # OpenAlex
    "api.fda.gov",               # openFDA
    "en.wikipedia.org",          # reference_lookup
    "www.wikidata.org",
    "en.wiktionary.org",         # define_word
    "api.stackexchange.com",     # qa_search
    "hn.algolia.com",            # hn_search
    "api.gdeltproject.org",      # events_search
    "openlibrary.org",           # books_search
    "archive.org",               # archive_search + wayback_lookup
    "web.archive.org",
]
WEATHER_HOSTS = ["api.open-meteo.com", "geocoding-api.open-meteo.com"]


def _host_port(url: str) -> tuple | None:
    try:
        u = urllib.parse.urlsplit(str(url).strip())
        if u.scheme not in ("http", "https") or not u.hostname:
            return None
        return u.hostname, u.port or (80 if u.scheme == "http" else 443)
    except Exception:
        return None


def _rule(name, hosts, port, methods, purpose):
    hosts_s = ", ".join(f'"{h}"' for h in sorted(set(hosts)))
    methods_s = ", ".join(f'"{m}"' for m in methods)
    return (f"[[rule]]\nname    = \"{name}\"\nhosts   = [{hosts_s}]\n"
            f"port    = {port}\nmethods = [{methods_s}]\n"
            f"purpose = \"{purpose}\"\n")


def render(cfg: dict) -> str:
    """The managed block's full text for this configuration."""
    lcfg = (cfg.get("tools") or {}).get("local") or {}
    on = lambda g: bool(lcfg.get("enabled")) and bool((lcfg.get(g) or {}).get("enabled"))
    parts = [BEGIN]
    if on("research"):
        parts.append(_rule("local-research", RESEARCH_HOSTS, 443, ["GET"],
                           "local toolset: the keyless research tools (Europe PMC, "
                           "OpenAlex, openFDA, Wikipedia/Wikidata, Wiktionary, Stack "
                           "Exchange, HN, GDELT, Open Library, Internet Archive)"))
    if on("weather"):
        parts.append(_rule("local-weather", WEATHER_HOSTS, 443, ["GET"],
                           "local toolset: Open-Meteo forecasts (keyless)"))
    if on("news"):
        by_port: dict = {}
        for f in (lcfg.get("news") or {}).get("feeds") or []:
            hp = _host_port(f.get("url"))
            if hp:
                by_port.setdefault(hp[1], set()).add(hp[0])
        for port in sorted(by_port):
            parts.append(_rule(f"local-news-{port}", sorted(by_port[port]), port,
                               ["GET"],
                               "local toolset: the news feeds the user follows"))
    if on("calendar"):
        hp = _host_port((lcfg.get("calendar") or {}).get("caldav_url"))
        if hp:
            parts.append(_rule("local-caldav", [hp[0]], hp[1],
                               ["GET", "PUT", "DELETE", "PROPFIND", "REPORT"],
                               "local toolset: the user's CalDAV calendar account "
                               "(writes only ever land on the Vinkona calendar)"))
    if on("mail"):
        servers = ", ".join(sorted({
            f"{a.get('host')}:{a.get('port', 993)}"
            for a in (lcfg.get("mail") or {}).get("accounts") or []
            if str(a.get("host") or "").strip()})) or "none configured"
        parts.append("# mail (IMAP) is not HTTP and does not pass through the broker:\n"
                     f"# direct TLS, read-only, to the configured server(s): {servers}\n")
    if len(parts) == 1:
        parts.append("# (no local-tools genres enabled — nothing granted)\n")
    parts.append(END)
    return "\n".join(parts)


def ensure(cfg: dict, path=None) -> bool:
    """Rewrite the managed block iff it changed; True when the file was touched."""
    from amiga_net.policy import POLICY_PATH
    p = path or POLICY_PATH
    try:
        text = p.read_text()
    except FileNotFoundError:
        text = ""
    block = render(cfg)
    if BEGIN in text and END in text:
        head, rest = text.split(BEGIN, 1)
        _, tail = rest.split(END, 1)
        new = head.rstrip("\n") + "\n\n" + block + tail
    else:
        new = (text.rstrip("\n") + "\n\n" if text.strip() else "") + block + "\n"
    if new == text:
        return False
    p.write_text(new)
    return True
