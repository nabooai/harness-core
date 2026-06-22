"""transport.py — generic litellm/SDK loop-hygiene + model resolution (from agent.py).

The per-event-loop litellm transport repair + `resolve_model` (wrap a `provider/...`
litellm name in a LitellmModel). Agent-agnostic — both targets call resolve_model here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from asyncio import AbstractEventLoop

    from agents import Model
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

# a model name with a `provider/` prefix is litellm-routed (the harness sweeps
# `gemini/...`, `openrouter/...`); a bare name is a native OpenAI model. The SDK's
# default provider only speaks OpenAI, so a litellm name MUST be wrapped (a bare
# `gemini/...` string silently fails to reach the provider -- the smoke caught this).
_LITELLM_PROVIDERS = (
    "gemini/",
    "openrouter/",
    "anthropic/",
    "groq/",
    "vertex_ai/",
    "bedrock/",
    "azure/",
    "together_ai/",
    "fireworks_ai/",
    "cohere/",
    "mistral/",
    "deepseek/",
)


class _PerLoopAsyncClient:
    """A per-EVENT-LOOP stand-in installed AS ``litellm.module_level_aclient``.

    litellm's streaming path reads ``litellm.module_level_aclient`` at STREAM-FETCH
    time (``streaming_handler.make_call(client=litellm.module_level_aclient)``), and
    the real object is one loop-bound ``AsyncHTTPHandler`` cached in module globals
    forever. The sweep runs every rep (and every judge call) under its OWN
    ``asyncio.run`` loop in its own thread, so concurrent reps inherited a handler
    bound to a sibling's (or a dead) loop -- the
    ``<asyncio.locks.Event ...> is bound to a different event loop`` /
    ``Event loop is closed`` MidStreamFallbackError class that killed 4/6 reps of
    the r0anchor2 e3 cell. POPPING the cached attr (the first fix attempt) only
    re-armed the race: litellm's lazy ``__getattr__`` re-creates it on whichever
    concurrent loop touches it first.

    This object is the opposite move: a REAL module attribute (so the lazy import
    never fires again) that forwards every attribute access to an
    ``AsyncHTTPHandler`` owned by the CURRENT running loop. Handlers live in a
    ``WeakKeyDictionary`` keyed by the loop OBJECT -- not ``id(loop)``, whose reuse
    after a loop is freed is exactly the collision litellm's own client cache
    suffers -- so an entry dies with its loop and can never be inherited.
    ``__class__`` mimics ``AsyncHTTPHandler`` so litellm's
    ``isinstance(client, AsyncHTTPHandler)`` guards accept the stand-in instead of
    falling back to the colliding global cache."""

    def __init__(self) -> None:
        import threading
        import weakref

        self._lock = threading.Lock()
        self._per_loop: weakref.WeakKeyDictionary[AbstractEventLoop, AsyncHTTPHandler] = (
            weakref.WeakKeyDictionary()
        )
        self._no_loop_handler: AsyncHTTPHandler | None = None

    def _new_handler(self) -> AsyncHTTPHandler:
        import litellm
        from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

        return AsyncHTTPHandler(
            timeout=getattr(litellm, "request_timeout", None),
            client_alias="fdav13 per-loop module aclient",
        )

    def _handler(self) -> AsyncHTTPHandler:
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        with self._lock:
            if loop is None:
                if self._no_loop_handler is None:
                    self._no_loop_handler = self._new_handler()
                return self._no_loop_handler
            h = self._per_loop.get(loop)
            if h is None:
                h = self._new_handler()
                self._per_loop[loop] = h
            return h

    @property  # type: ignore[misc]
    def __class__(self) -> type:  # noqa: D105 - isinstance mimicry, see class docstring
        from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

        return AsyncHTTPHandler

    def __getattr__(self, name: str) -> object:
        # a forwarding PROXY: it returns an arbitrary attribute of the wrapped handler, whose
        # type is genuinely unknown per-name -- `object` is the honest type here (not a lazy
        # escape: there is no narrower type for "any attribute of the delegate").
        return getattr(self._handler(), name)


def fresh_litellm_transport() -> None:
    """Make litellm's transport safe for THIS event loop (the per-rep/per-judge seam).

    Two moves, both required (each alone was proven insufficient live, 2026-06-12,
    /tmp/r0_anchor_sweep.log + /tmp/r0_anchor2.log):

    1. Disable the aiohttp transport: its handler holds ONE mutable ClientSession
       whose loop-mismatch "repair" schedules ``ClientSession.close()`` on the wrong
       loop (the ``loop ... is not the running loop`` tracebacks).
    2. Install ``_PerLoopAsyncClient`` AS ``litellm.module_level_aclient`` (idempotent)
       so every event loop drives its OWN handler -- see the class docstring for why
       popping the attr instead re-armed the cross-loop race.

    Plus hygiene: clear litellm's id(loop)-keyed client cache so provider-level
    clients are also rebuilt on the current loop (eviction is documented non-closing;
    a parallel rep mid-stream keeps its reference)."""
    import litellm

    litellm.disable_aiohttp_transport = True  # httpx transport: no loop-bound session
    cache = getattr(litellm, "in_memory_llm_clients_cache", None)
    if cache is not None:
        getattr(cache, "cache_dict", {}).clear()
        getattr(cache, "ttl_dict", {}).clear()
    current = litellm.__dict__.get("module_level_aclient")
    if type(current) is not _PerLoopAsyncClient:  # type(), not isinstance() -- our
        # own __class__ mimicry makes isinstance lie about the stand-in
        litellm.module_level_aclient = _PerLoopAsyncClient()  # ty: ignore[invalid-assignment]


async def aclose_current_loop_transport() -> None:
    """Close the per-loop litellm ``AsyncHTTPHandler`` for the CURRENT running loop and drop
    it from the per-loop map -- call this INSIDE the loop, right before ``asyncio.run`` tears
    it down. Without it the handler's httpx transport is finalized by GC on a DEAD loop, which
    raises ``RuntimeError: Event loop is closed`` from ``transport_stream.aclose`` once per
    run (harmless -- the run already finished -- but it floods stderr; the sweep logged 37).
    Best-effort + idempotent: not our stand-in, no running loop, or no handler for this loop
    -> a no-op; a close that itself raises is swallowed (teardown must never fail a run)."""
    import asyncio

    import litellm

    client = litellm.__dict__.get("module_level_aclient")
    if type(client) is not _PerLoopAsyncClient:  # type(): our __class__ mimicry lies to isinstance
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    handler = cast("_PerLoopAsyncClient", client)._per_loop.pop(loop, None)
    if handler is None:
        return
    try:
        await handler.close()
    except Exception:  # noqa: BLE001 -- transport teardown is best-effort, never load-bearing
        return


def resolve_model(model: object | None) -> str | Model | None:
    """Pass through a Model object or None; wrap a litellm-style string in a
    LitellmModel so a `provider/...` name actually reaches its provider. This seam owns
    the SDK-boundary type assertion: callers pass the agent-agnostic `object | None`, and
    a non-str value is contractually a `Model` (or None) — narrowed here so every call
    site gets a precisely-typed result with no cast of its own."""
    if not isinstance(model, str):
        return cast("Model | None", model)
    if model.startswith(_LITELLM_PROVIDERS):
        from agents.extensions.models.litellm_model import LitellmModel

        fresh_litellm_transport()  # per-rep loop hygiene BEFORE the first request
        return LitellmModel(model=model)
    return model  # bare name -> native OpenAI provider


_REASONING_EFFORTS = ("minimal", "low", "medium", "high")
