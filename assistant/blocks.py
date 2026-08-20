"""
VIN-TOOL-01 block layer: the closed semantic type set, the block contract, the
registry, mechanical JSON-schema derivation, the fixtures runner, and the hash
manifest.  See self_tooling_spec.md — §3/§4 body, §0.1/0.7 adaptations.

Design rules enforced here, not merely documented:
  * Ports carry SEMANTIC types from the closed set — a bare dict/list/str port
    is rejected at registration.
  * Params schemas are DERIVED from the declarative spec at call time; no
    schema is ever hand-written or stored anywhere.
  * A block declaring mutate/fs-write MUST declare its rollback surface.
  * Every block ships ≥ 2 fixtures (one an edge case, by convention).
  * Side effects (net, LM) reach a block only through the injected Ctx, so
    fixture/shadow runs replay them and NEVER touch the world.  A replay miss
    fails loudly — degrade or fail, never invent (§3 failure-mode policy).

Pure stdlib, by the dependency ratchet (§0.1: pydantic rejected in writing).
"""
from __future__ import annotations

import hashlib
import inspect
import re
from dataclasses import dataclass

# ── The closed semantic type set (§4) ─────────────────────────────────────────
BASE_TYPES = frozenset({
    "FeedRef", "Document", "Passage", "Record", "Table", "Event",
    "FileRef", "ImageRef", "AudioRef", "Query", "Digest",
})
CONTAINERS = ("List", "Ranked")          # List[X] unordered-ish; Ranked[X] ordered

CAPABILITIES = frozenset({
    "net", "fs-read", "fs-write", "mutate", "process", "sensor", "biometric",
})
_ROLLBACK_REQUIRED = frozenset({"mutate", "fs-write"})

PARAM_TYPES = frozenset({"string", "number", "integer", "boolean", "array"})

_SEMVER = re.compile(r"^\d+\.\d+\.\d+$")
_TYPE = re.compile(r"^(?:(List|Ranked)\[([A-Za-z]+)\]|([A-Za-z]+))$")


class BlockError(Exception):
    """A block-layer failure with a reason a person can read."""


def parse_type(s: str) -> tuple[str | None, str]:
    """'Ranked[Document]' -> ('Ranked', 'Document'); 'Query' -> (None, 'Query').
    Raises on anything outside the closed set — including bare dict/list/str."""
    m = _TYPE.match(s or "")
    if not m:
        raise BlockError(f"malformed type {s!r}")
    container, base = (m.group(1), m.group(2)) if m.group(1) else (None, m.group(3))
    if base not in BASE_TYPES:
        raise BlockError(f"type {s!r} is outside the closed semantic set")
    return container, base


def types_compatible(out_type: str, in_type: str) -> bool:
    """Edges connect EXACT types only; coercion is an adapter block's job (§4).
    The one directional allowance: Ranked[X] may feed a List[X] port (order is
    extra information, safely forgotten) — never the reverse."""
    if out_type == in_type:
        return True
    oc, ob = parse_type(out_type)
    ic, ib = parse_type(in_type)
    return oc == "Ranked" and ic == "List" and ob == ib


# ── Runtime shape checks per semantic type ────────────────────────────────────
# Light structural validation: enough to catch a block emitting the wrong shape
# in fixtures and dry-runs, without pretending to be a full schema language.

def _is_str(v) -> bool:
    return isinstance(v, str) and bool(v.strip())


_BASE_CHECKS = {
    "FeedRef":  lambda v: isinstance(v, dict) and _is_str(v.get("url")),
    "Document": lambda v: isinstance(v, dict) and isinstance(v.get("text"), str),
    "Passage":  lambda v: isinstance(v, dict) and _is_str(v.get("text")),
    "Record":   lambda v: isinstance(v, dict),
    "Table":    lambda v: isinstance(v, dict) and isinstance(v.get("rows"), list)
                and all(isinstance(r, dict) for r in v.get("rows", [])),
    "Event":    lambda v: isinstance(v, dict) and _is_str(v.get("kind")),
    "FileRef":  lambda v: isinstance(v, dict) and _is_str(v.get("path")),
    "ImageRef": lambda v: isinstance(v, dict) and _is_str(v.get("path")),
    "AudioRef": lambda v: isinstance(v, dict) and _is_str(v.get("path")),
    "Query":    lambda v: isinstance(v, dict) and _is_str(v.get("text")),
    "Digest":   lambda v: isinstance(v, dict) and isinstance(v.get("text"), str),
}


def check_value(type_str: str, value) -> str | None:
    """None when `value` fits the semantic type's shape, else the reason."""
    container, base = parse_type(type_str)
    if container:
        if not isinstance(value, list):
            return f"expected a list for {type_str}, got {type(value).__name__}"
        for i, item in enumerate(value):
            if not _BASE_CHECKS[base](item):
                return f"item {i} is not a valid {base}"
        return None
    if not _BASE_CHECKS[base](value):
        return f"not a valid {base}"
    return None


# ── The block contract (§3) ───────────────────────────────────────────────────
@dataclass(frozen=True)
class Block:
    name: str
    version: str                      # semver; graphs pin it
    summary: str                      # written for retrieval — the model matches on this
    ports_in: dict                    # port name -> semantic type
    ports_out: dict
    params: dict                      # name -> declarative spec (schema derived, §0.1)
    capabilities: frozenset
    fixtures: tuple                   # ≥2 of {name, inputs, params, ctx, expect}
    failure_modes: tuple              # enumerated strings; fail/degrade, never invent
    fn: object                        # fn(inputs, params, ctx) -> outputs by port
    rollback: str = ""                # write-surface description; required iff mutate/fs-write

    @property
    def key(self) -> str:
        return f"{self.name}@{self.version}"


def _validate_param_spec(name: str, spec: dict):
    if not isinstance(spec, dict) or spec.get("type") not in PARAM_TYPES:
        raise BlockError(f"param {name!r}: type must be one of {sorted(PARAM_TYPES)}")
    if not _is_str(spec.get("description", "")):
        raise BlockError(f"param {name!r}: a description is required (the model reads it)")
    if "enum" in spec and (not isinstance(spec["enum"], list) or not spec["enum"]):
        raise BlockError(f"param {name!r}: enum must be a non-empty list")


def _validate_block(b: Block):
    if not re.match(r"^[a-z][a-z0-9_]*$", b.name):
        raise BlockError(f"bad block name {b.name!r}")
    if not _SEMVER.match(b.version):
        raise BlockError(f"{b.name}: version {b.version!r} is not semver")
    if not _is_str(b.summary):
        raise BlockError(f"{b.name}: summary is required")
    for side, ports in (("in", b.ports_in), ("out", b.ports_out)):
        for port, t in ports.items():
            try:
                parse_type(t)
            except BlockError as e:
                raise BlockError(f"{b.name}: port {side}:{port}: {e}") from None
    if not b.ports_out:
        raise BlockError(f"{b.name}: a block must produce something")
    for pname, spec in b.params.items():
        _validate_param_spec(f"{b.name}.{pname}", spec)
    unknown = set(b.capabilities) - CAPABILITIES
    if unknown:
        raise BlockError(f"{b.name}: unknown capabilities {sorted(unknown)}")
    if set(b.capabilities) & _ROLLBACK_REQUIRED and not _is_str(b.rollback):
        raise BlockError(f"{b.name}: mutate/fs-write requires a declared rollback surface")
    if len(b.fixtures) < 2:
        raise BlockError(f"{b.name}: at least 2 fixtures required (one an edge case)")
    if not b.failure_modes:
        raise BlockError(f"{b.name}: enumerate the failure modes")
    if not callable(b.fn):
        raise BlockError(f"{b.name}: fn is not callable")


# ── Registry ──────────────────────────────────────────────────────────────────
_REGISTRY: dict = {}                  # name -> {version -> Block}


def register(b: Block) -> Block:
    _validate_block(b)
    versions = _REGISTRY.setdefault(b.name, {})
    if b.version in versions:
        raise BlockError(f"{b.key} is already registered")
    versions[b.version] = b
    return b


def block(**kw):
    """Decorator: declare-and-register.  The decorated function IS the block's fn."""
    def wrap(fn):
        register(Block(fn=fn,
                       capabilities=frozenset(kw.pop("capabilities", ())),
                       fixtures=tuple(kw.pop("fixtures", ())),
                       failure_modes=tuple(kw.pop("failure_modes", ())),
                       **kw))
        return fn
    return wrap


def get(name: str, version: str | None = None) -> Block:
    versions = _REGISTRY.get(name)
    if not versions:
        raise BlockError(f"no block named {name!r}")
    if version is None:
        latest = max(versions, key=lambda v: tuple(int(x) for x in v.split(".")))
        return versions[latest]
    if version not in versions:
        raise BlockError(f"{name}@{version} not in registry (have {sorted(versions)})")
    return versions[version]


def names() -> list:
    return sorted(_REGISTRY)


def catalogue() -> list:
    """What the model sees (summaries, ports, params) — never source (§3)."""
    out = []
    for name in names():
        b = get(name)
        out.append({"name": b.name, "version": b.version, "summary": b.summary,
                    "ports_in": dict(b.ports_in), "ports_out": dict(b.ports_out),
                    "params": params_json_schema(b),
                    "capabilities": sorted(b.capabilities)})
    return out


def clear_registry():
    """Test hook."""
    _REGISTRY.clear()


# ── Mechanical schema derivation (§0.1 — the pydantic replacement) ────────────
def params_json_schema(b: Block) -> dict:
    """JSON schema for a block's params, derived from the declarative spec at
    call time.  Nothing is stored; there is nothing to drift."""
    props, required = {}, []
    for name, spec in sorted(b.params.items()):
        p = {"type": spec["type"], "description": spec["description"]}
        if "enum" in spec:
            p["enum"] = list(spec["enum"])
        if spec["type"] == "array":
            p["items"] = dict(spec.get("items", {"type": "string"}))
        if "default" in spec:
            p["default"] = spec["default"]
        else:
            required.append(name)
        props[name] = p
    schema = {"type": "object", "properties": props, "additionalProperties": False}
    if required:
        schema["required"] = required
    return schema


def fill_param_defaults(b: Block, params: dict) -> dict:
    out = dict(params or {})
    for name, spec in b.params.items():
        if name not in out and "default" in spec:
            out[name] = spec["default"]
    return out


_PY_TYPES = {"string": str, "boolean": bool, "array": list}


def check_params(b: Block, params: dict) -> list:
    """Errors for unknown names, missing required, wrong types, enum misses."""
    errs = []
    for name in params or {}:
        if name not in b.params:
            errs.append(f"unknown param {name!r} (schema allows nothing extra)")
    for name, spec in b.params.items():
        if name not in (params or {}):
            if "default" not in spec:
                errs.append(f"missing required param {name!r}")
            continue
        v = params[name]
        t = spec["type"]
        if t == "integer":
            ok = isinstance(v, int) and not isinstance(v, bool)
        elif t == "number":
            ok = isinstance(v, (int, float)) and not isinstance(v, bool)
        else:
            ok = isinstance(v, _PY_TYPES[t])
        if not ok:
            errs.append(f"param {name!r}: expected {t}")
            continue
        if "enum" in spec and v not in spec["enum"]:
            errs.append(f"param {name!r}: {v!r} not in {spec['enum']}")
    return errs


# ── Ctx: injected side effects (§2.2) ─────────────────────────────────────────
class Ctx:
    """What a block may touch beyond its inputs.  The engine injects a live Ctx
    scoped to the graph's capability set; fixtures inject a ReplayCtx.  Blocks
    never import sockets, files, or LM clients themselves."""

    def net(self, url: str) -> str:
        raise BlockError("this Ctx has no network")

    def llm(self, prompt: str) -> str:
        raise BlockError("this Ctx has no language model")

    def now(self) -> float:
        raise BlockError("this Ctx has no clock")

    def write(self, relpath: str, text: str) -> str:
        raise BlockError("this Ctx has no writable store")

    def news(self, query: str, hours: int, limit: int) -> list:
        raise BlockError("this Ctx has no news archive")


class ReplayCtx(Ctx):
    """Canned side effects for fixture/shadow runs.  A miss FAILS — a fixture
    that would silently invent data is the exact failure class this contract
    exists to remove.  Writes never touch disk: they are recorded in
    self.writes so a fixture can assert WHAT would have been written."""

    def __init__(self, canned: dict | None = None):
        c = canned or {}
        self._canned = c
        self._net = dict(c.get("net", {}))
        self._llm = c.get("llm")          # str (one reply) or {substr: reply}
        self._now = c.get("now", 0.0)
        self.writes: list = []            # (relpath, text) — shadow, no disk

    def net(self, url: str) -> str:
        if url not in self._net:
            raise BlockError(f"replay miss: no canned response for {url!r}")
        return self._net[url]

    def llm(self, prompt: str) -> str:
        if isinstance(self._llm, str):
            return self._llm
        if isinstance(self._llm, dict):
            for sub, reply in self._llm.items():
                if sub in prompt:
                    return reply
        raise BlockError("replay miss: no canned LM reply matches this prompt")

    def now(self) -> float:
        return self._now

    def write(self, relpath: str, text: str) -> str:
        self.writes.append((relpath, text))
        return f"(shadow)/{relpath}"

    def news(self, query: str, hours: int, limit: int) -> list:
        if "news" not in self._canned:
            raise BlockError("replay miss: no canned news archive")
        return list(self._canned["news"])[: max(1, int(limit))]


# ── Fixtures runner (§3, §6.2) ────────────────────────────────────────────────
def run_fixture(b: Block, fx: dict) -> tuple:
    """(ok, detail).  Checks params, runs fn under ReplayCtx, type-checks every
    declared out port, then compares against the fixture's expectations."""
    name = fx.get("name", "?")
    perrs = check_params(b, fx.get("params", {}))
    if perrs:
        return False, f"{b.key}[{name}]: bad fixture params: {perrs}"
    params = fill_param_defaults(b, fx.get("params", {}))
    try:
        got = b.fn(dict(fx.get("inputs", {})), params, ReplayCtx(fx.get("ctx")))
    except BlockError as e:
        if fx.get("expect_error"):
            return (str(fx["expect_error"]) in str(e)), \
                f"{b.key}[{name}]: error {e!r} vs expected {fx['expect_error']!r}"
        return False, f"{b.key}[{name}]: {e}"
    if fx.get("expect_error"):
        return False, f"{b.key}[{name}]: expected an error, got a result"
    if set(got or {}) != set(b.ports_out):
        return False, f"{b.key}[{name}]: ports {sorted(got or {})} != declared {sorted(b.ports_out)}"
    for port, t in b.ports_out.items():
        bad = check_value(t, got[port])
        if bad:
            return False, f"{b.key}[{name}]: out {port}: {bad}"
    for port, want in (fx.get("expect") or {}).items():
        if got.get(port) != want:
            return False, f"{b.key}[{name}]: {port} = {got.get(port)!r}, wanted {want!r}"
    return True, f"{b.key}[{name}]"


def run_all_fixtures() -> tuple:
    """(failures, count) across the whole registry — the §11 fixture CI."""
    failures, count = [], 0
    for name in names():
        for version, b in sorted(_REGISTRY[name].items()):
            for fx in b.fixtures:
                count += 1
                ok, detail = run_fixture(b, fx)
                if not ok:
                    failures.append(detail)
    return failures, count


# ── Hash manifest (§0.7) ──────────────────────────────────────────────────────
def manifest() -> dict:
    """block key -> sha256 of its fn's source.  The runtime refuses a block
    whose source no longer matches the recorded manifest."""
    out = {}
    for name in names():
        for version, b in _REGISTRY[name].items():
            src = inspect.getsource(b.fn)
            out[b.key] = hashlib.sha256(src.encode()).hexdigest()
    return out


def verify_manifest(stored: dict) -> list:
    """Mismatch/missing block keys against a previously recorded manifest."""
    current = manifest()
    return sorted(k for k, h in (stored or {}).items() if current.get(k) != h)
