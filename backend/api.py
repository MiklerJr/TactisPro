"""API HTTP para el dashboard de reproducción."""

from contextlib import asynccontextmanager
from threading import Lock

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from .inference import ReplayInference


@asynccontextmanager
async def lifespan(app):
    app.state.inference = ReplayInference()
    app.state.inference_lock = Lock()
    yield


def health(request):
    return JSONResponse({"status": "ok"})


def metrics(request):
    return JSONResponse(request.app.state.inference.metrics())


def game(request):
    inference = request.app.state.inference
    match_id = request.path_params["match_id"]
    game_num = request.path_params["game_num"]
    timings = {} if request.query_params.get("timing") == "1" else None
    with request.app.state.inference_lock:
        try:
            result = inference.game(match_id, game_num, timings)
        except KeyError as exc:
            if exc.args != ((match_id, game_num),):
                raise
            return JSONResponse({"detail": "Juego sin datos de inferencia."}, status_code=404)
    headers = {}
    if timings is not None:
        headers["Server-Timing"] = ", ".join(f"{name};dur={duration:.3f}" for name, duration in timings.items())
    return JSONResponse(result, headers=headers)


app = Starlette(
    routes=[
        Route("/health", health),
        Route("/metrics", metrics),
        Route("/matches/{match_id}/games/{game_num:int}", game),
    ],
    lifespan=lifespan,
)
