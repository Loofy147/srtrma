from __future__ import annotations
import asyncio, json, time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Dict, List, Set
from enum import Enum, auto


# ── State machine ─────────────────────────────────────────────────────────────

class S(Enum):
    DIRTY = auto()
    LIVE  = auto()
    HOT   = auto()
    DEAD  = auto()


@dataclass
class Snap:
    value: Any
    ts:    float          # monotonic time of compute
    ms:    float          # wall-clock compute duration (ms)
    state: S = S.LIVE
    err:   Optional[Exception] = None

    def expired(self, ttl: float) -> bool:
        return (time.monotonic() - self.ts) > ttl


# ── Base Field ────────────────────────────────────────────────────────────────

class Field:
    def __init__(self, name: str, fn: Callable, deps: list[str] = None,
                 ttl: float = float("inf"), tag: str = "derived"):
        self.name    = name
        self.fn      = fn
        self.deps    = deps or []
        self.ttl     = ttl
        self.tag     = tag
        self._snap:    Optional[Snap] = None
        self._st       = S.DIRTY
        self._subs:    set[str] = set()
        self._calls    = 0
        self._hits     = 0
        self.version   = 0                           # v2: value version counter
        self._fut:     Optional[asyncio.Future] = None  # v2: coalescing gate

    @property
    def dirty(self) -> bool:
        if self._st != S.LIVE: return True
        if self._snap and self._snap.expired(self.ttl): return True
        if not self._snap: return True
        return False

    async def resolve(self, ctx: dict[str, Any]) -> Snap:
        self._st = S.HOT
        self._calls += 1
        t0 = time.perf_counter()
        try:
            val = (await self.fn(**ctx)
                   if asyncio.iscoroutinefunction(self.fn)
                   else self.fn(**ctx))
            snap = Snap(val, time.monotonic(), (time.perf_counter()-t0)*1e3)
            self._snap, self._st = snap, S.LIVE
            self.version += 1
            return snap
        except Exception as e:
            snap = Snap(None, time.monotonic(), (time.perf_counter()-t0)*1e3, S.DEAD, e)
            self._snap, self._st = snap, S.DEAD
            return snap


# ── Source — mutable leaf ─────────────────────────────────────────────────────

class Source(Field):
    def __init__(self, name: str, value: Any = None):
        super().__init__(name, fn=lambda: None, deps=[], tag="source")
        if value is not None:
            self._snap = Snap(value, time.monotonic(), 0)
            self._st   = S.LIVE

    def set(self, value: Any):
        """Always leaves Source in LIVE state."""
        self._snap = Snap(value, time.monotonic(), 0)
        self._st   = S.LIVE
        self.version += 1

    async def resolve(self, ctx: dict) -> Snap:
        self._calls += 1
        if self._snap: return self._snap
        raise ValueError(f"Source '{self.name}' uninitialized")


# ── Field Graph ───────────────────────────────────────────────────────────────

class FieldGraph:
    def __init__(self, name: str = "G"):
        self.name = name
        self._F: dict[str, Field] = {}

    # -- registration ---------------------------------------------------------

    def _reg(self, f: Field) -> Field:
        self._F[f.name] = f
        for dep in f.deps:
            if dep in self._F:
                self._F[dep]._subs.add(f.name)
        return f

    def source(self, name: str, value: Any = None) -> Source:
        return self._reg(Source(name, value))

    def derive(self, name: str, fn: Callable, deps: list[str],
               ttl: float = float("inf")) -> Field:
        return self._reg(Field(name, fn, deps, ttl))

    def agent(self, name: str, **kwargs) -> "AgentField":
        return self._reg(AgentField(name, self, **kwargs))

    # -- T1-FIX: invalidation skips Source origin ----------------------------

    def invalidate(self, name: str):
        """
        Propagate DIRTY to all transitive dependents.
        If origin is a Source (just set to LIVE), skip dirtying it.
        """
        f           = self._F.get(name)
        if not f: return
        skip_origin = isinstance(f, Source)
        start_set   = list(f._subs) if skip_origin else [name]
        q, seen     = list(start_set), set()
        while q:
            n = q.pop(0)
            if n in seen: continue
            seen.add(n)
            if n in self._F:
                self._F[n]._st = S.DIRTY
                q.extend(self._F[n]._subs)

    # -- T2-FIX + T5-FIX: get() with circular guard + coalescing -------------

    async def get(self, name: str,
                  _path: frozenset = frozenset()) -> Any:
        """
        T2-FIX: _path tracks the current resolution chain.
                If name already in _path → CircularDependencyError.
        T5-FIX: If a compute is already in-flight for this field,
                await the same Future instead of spawning a duplicate compute.
        """
        if name not in self._F:
            raise KeyError(f"Field '{name}' not registered")

        # T2: circular detection
        if name in _path:
            raise RecursionError(
                f"Circular dependency: {' → '.join(sorted(_path))} → {name}")

        f = self._F[name]

        # cache hit
        if not f.dirty:
            f._hits += 1
            return f._snap.value

        # T5: coalesce concurrent readers onto the same in-flight Future
        if f._fut is not None:
            val = await f._fut
            f._hits += 1
            return val

        # start new compute
        loop = asyncio.get_event_loop()
        fut  = loop.create_future()
        f._fut = fut
        try:
            new_path = _path | {name}
            # resolve deps concurrently
            dep_vals = await asyncio.gather(
                *[self.get(d, new_path) for d in f.deps],
                return_exceptions=True
            )
            for v in dep_vals:
                if isinstance(v, Exception):
                    fut.set_exception(v)
                    raise v
            ctx  = dict(zip(f.deps, dep_vals))
            snap = await f.resolve(ctx)
            if snap.state == S.DEAD:
                fut.set_exception(snap.err)
                raise snap.err
            fut.set_result(snap.value)
            return snap.value
        except Exception as e:
            if not fut.done(): fut.set_exception(e)
            raise
        finally:
            f._fut = None

    async def set(self, name: str, value: Any):
        if name not in self._F:
            self.source(name, value)
            return
        f = self._F[name]
        if not isinstance(f, Source):
            raise TypeError(f"'{name}' is not a Source field")
        f.set(value)
        self.invalidate(name)

    async def batch(self, names: list[str]) -> dict[str, Any]:
        results = await asyncio.gather(
            *[self.get(n) for n in names], return_exceptions=True)
        return {n: r for n, r in zip(names, results)}

    # -- v2 extensions --------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Current values of all LIVE fields."""
        return {n: f._snap.value
                for n, f in self._F.items()
                if f._st == S.LIVE and f._snap is not None}

    def topo(self) -> list[str]:
        """Kahn's topological sort. Raises ValueError on cycle."""
        in_deg = {}
        for n, f in self._F.items():
            in_deg[n] = sum(1 for d in f.deps if d in self._F)
        queue  = [n for n, d in in_deg.items() if d == 0]
        order  = []
        while queue:
            n = queue.pop(0)
            order.append(n)
            for m in sorted(self._F[n]._subs):
                if m in in_deg:
                    in_deg[m] -= 1
                    if in_deg[m] == 0:
                        queue.append(m)
        if len(order) != len(self._F):
            cycle = set(self._F) - set(order)
            raise ValueError(f"Cycle detected — involved: {cycle}")
        return order

    def stats(self) -> list[dict]:
        return [{
            "field":   n,
            "type":    f.tag,
            "state":   f._st.name,
            "deps":    f.deps,
            "subs":    sorted(f._subs),
            "calls":   f._calls,
            "hits":    f._hits,
            "version": f.version,
            "ms":      round(f._snap.ms, 4) if f._snap else None,
        } for n, f in self._F.items()]


# ── Agent Field — LLM as calculated field ────────────────────────────────────

class AgentField(Field):
    """
    Computed via Claude API (or mock). When tool_use returned,
    auto-spawns ToolFields into graph and recurses until end_turn.
    v2: asyncio.Lock prevents concurrent re-entry on same field.
    """
    def __init__(self, name: str, graph: FieldGraph,
                 tools:     list  = None,
                 system:    str   = "You are a helpful assistant.",
                 model:     str   = "claude-sonnet-4-20250514",
                 max_turns: int   = 8,
                 deps:      list  = None,
                 mock:      bool  = True):
        super().__init__(name, fn=self._run,
                         deps=deps or ["_messages"], tag="agent")
        self.graph     = graph
        self.tools     = tools or []
        self.system    = system
        self.model     = model
        self.max_turns = max_turns
        self.mock      = mock
        self._lock     = asyncio.Lock()   # v2
        self._trace: list = []
        self._spawned_tools: Set[str] = set()

    async def _llm(self, messages: list, api_tools: list) -> dict:
        if self.mock:
            last = messages[-1]["content"]
            if (isinstance(last, list)
                    and any(b.get("type") == "tool_result" for b in last)):
                return {"stop_reason": "end_turn",
                        "content": [{"type": "text",
                                     "text": "[MOCK] Synthesis complete."}]}
            if api_tools and not self._trace:
                # Use a query from the last message
                msg_text = str(last)
                q = msg_text[-60:]
                return {"stop_reason": "tool_use", "content": [
                    {"type": "text", "text": "Fetching data…"},
                    {"type": "tool_use", "id": "tu_m0001",
                     "name": api_tools[0]["name"],
                     "input": {"query": q}}]}
            return {"stop_reason": "end_turn",
                    "content": [{"type": "text",
                                 "text": f"[MOCK] Done: {str(last)[:80]}"}]}

        import anthropic
        client = anthropic.Anthropic()
        loop   = asyncio.get_event_loop()
        resp   = await loop.run_in_executor(None, lambda: client.messages.create(
            model=self.model, max_tokens=4096, system=self.system,
            messages=messages,
            tools=api_tools if api_tools else anthropic.NOT_GIVEN,
        ))
        return {"stop_reason": resp.stop_reason, "content": [
            {"type": "text",     "text": b.text}     if hasattr(b, "text") else
            {"type": "tool_use", "id": b.id,
             "name": b.name,     "input": b.input}
            for b in resp.content]}

    async def _run(self, _messages: list) -> dict:
        async with self._lock:
            tool_map  = {t["name"]: t["fn"] for t in self.tools if "fn" in t}
            api_tools = [{k: v for k, v in t.items() if k != "fn"}
                         for t in self.tools]
            messages  = list(_messages)
            self._trace = []
            self._spawned_tools = set()

            try:
                for turn in range(self.max_turns):
                    resp = await self._llm(messages, api_tools)
                    self._trace.append({"turn": turn, "stop": resp["stop_reason"]})

                    if resp["stop_reason"] == "end_turn":
                        text = next(
                            (b["text"] for b in resp["content"]
                             if b.get("type") == "text"), "")
                        return {"text": text, "turns": turn+1,
                                "trace": self._trace}

                    if resp["stop_reason"] == "tool_use":
                        messages.append({"role": "assistant",
                                         "content": resp["content"]})
                        results = []
                        for blk in resp["content"]:
                            if blk.get("type") != "tool_use": continue
                            fid = f"_tool_{blk['name']}_{blk['id'][-4:]}"
                            fn  = tool_map.get(blk["name"])
                            if fn:
                                inp = blk["input"]
                                self.graph._reg(
                                    Field(fid, lambda **_: fn(**inp), [], tag="tool"))
                                self._spawned_tools.add(fid)
                                val = await self.graph.get(fid)
                                results.append({
                                    "type":        "tool_result",
                                    "tool_use_id": blk["id"],
                                    "content":     (json.dumps(val)
                                                    if not isinstance(val, str)
                                                    else val),
                                })
                        messages.append({"role": "user", "content": results})

                return {"text": "max_turns_exceeded",
                        "turns": self.max_turns, "trace": self._trace}
            finally:
                self._cleanup_tools()

    def _cleanup_tools(self):
        """Garbage collect transient tool fields from the graph."""
        for fid in self._spawned_tools:
            if fid in self.graph._F:
                # Remove from subscriptions of its deps (though tool fields here have none)
                f = self.graph._F[fid]
                for dep in f.deps:
                    if dep in self.graph._F:
                        self.graph._F[dep]._subs.discard(fid)
                # Remove from graph
                del self.graph._F[fid]
        self._spawned_tools.clear()
