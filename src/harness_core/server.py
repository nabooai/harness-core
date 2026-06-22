"""server.py — the harness-core HTTP server: a graf-free read-API + dashboard over run dirs.

A standalone FastAPI app that surfaces what the harness writes (`harness_core.results`):
the per-run Result + ECONOMICS axes (PASS / TURNS / TOKENS / COST / TIME) the harness
exists to move, plus the full per-run trace (timeline + spans + artifacts). It is the
extracted, decoupled twin of the graf monorepo's `/harness-core` + `/tracesv13` webapp
views — here it depends on nothing but `harness_core` itself.

Run it:
    harness-core server                          # serve on 127.0.0.1:8077
    harness-core server --host 0.0.0.0 --port 9000
    HARNESS_RUNS_ROOTS="fdav14=fdav14/runs" harness-core server

Requires the `server` extra (`pip install harness-core[server]`): fastapi + uvicorn.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from harness_core import results

if TYPE_CHECKING:
    from fastapi import FastAPI

_STATIC = Path(__file__).resolve().parent / "static"


def _index_html() -> str:
    """The self-contained dashboard page (no build step, no CDN). Shipped as package data."""
    return (_STATIC / "index.html").read_text(encoding="utf-8")


def create_app() -> FastAPI:
    """Build the FastAPI app. Imported lazily so the rest of the library never hard-depends
    on fastapi — install the `server` extra to use this."""
    from fastapi import FastAPI, HTTPException, Query, Response
    from fastapi.responses import HTMLResponse

    app = FastAPI(title="harness-core", version="0.1.0")

    @app.get("/healthz")
    def healthz() -> dict[str, object]:
        return {"ok": True, "harnesses": results.harness_names()}

    @app.get("/api/harnesses")
    def api_harnesses() -> list[str]:
        return results.harness_names()

    @app.get("/api/runs")
    def api_runs(
        limit: int | None = Query(default=100, ge=1, le=10000),
        harness: str | None = Query(default=None),
    ) -> list[dict]:
        return results.list_cells(limit=limit, harness=harness)

    @app.get("/api/runs/{cell_id}")
    def api_run(cell_id: str) -> dict:
        cell = results.load_cell(cell_id)
        if cell is None:
            raise HTTPException(status_code=404, detail="run not found")
        return cell

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _index_html()

    @app.get("/favicon.ico")
    def favicon() -> Response:
        # the no-build dashboard ships no icon asset; answer the browser's automatic
        # request with an empty 204 so it doesn't log a 404 on every page load.
        return Response(status_code=204)

    return app


def serve(host: str = "127.0.0.1", port: int = 8077) -> None:
    """Run the server with uvicorn (blocking). Requires the `server` extra."""
    import uvicorn

    uvicorn.run(create_app(), host=host, port=port)
