"""Measure the local R8 replay in Chromium via its DevTools endpoint."""

import argparse
import json
import platform
import subprocess
from pathlib import Path
from time import perf_counter
from urllib.parse import quote
from urllib.request import urlopen

import numpy as np
import pandas as pd
from websockets.sync.client import connect

from config import CLEAN_DATASET, MOMENTUM_DATASET, PROCESSED_DIR
from src.momentum.crossings import momentum_crossings


class Browser:
    """Small synchronous DevTools client for the isolated benchmark tab."""

    def __init__(self, connection):
        self.connection = connection
        self.request_id = 0

    def command(self, method, params):
        self.request_id += 1
        self.connection.send(json.dumps({"id": self.request_id, "method": method, "params": params}))
        while True:
            reply = json.loads(self.connection.recv(timeout=30))
            if reply.get("id") == self.request_id:
                if "error" in reply:
                    raise RuntimeError(reply["error"])
                return reply["result"]

    def evaluate(self, code):
        result = self.command("Runtime.evaluate", {
            "expression": code, "awaitPromise": True, "returnByValue": True,
        })
        if "exceptionDetails" in result:
            raise RuntimeError(result["exceptionDetails"])
        return result["result"].get("value")


BROWSER_FUNCTIONS = """
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
async function waitUntil(read) {
    for (let i = 0; i < 2000; i++) {
        const value = read();
        if (value) return value;
        await sleep(10);
    }
    throw Error("El dashboard no alcanzó el estado esperado");
}
function chart(root = document) {
    const svg = root.querySelector("svg.court-chart");
    if (svg) return svg;
    for (const e of root.querySelectorAll("*")) {
        const found = e.shadowRoot && chart(e.shadowRoot);
        if (found) return found;
    }
    return null;
}
const next = () => [...document.querySelectorAll("button")].find(e => e.textContent.trim() === "Siguiente");
const shownPoint = () => Number(document.querySelector('input[aria-label="Punto visible"]')?.value);
const frame = () => new Promise(resolve => requestAnimationFrame(resolve));
"""


def browser_samples(browser, url, limit):
    browser.command("Page.navigate", {"url": url})
    initialization = """
    (async () => {
        %s
        const svg = await waitUntil(() => chart()?.dataset.match && next() && shownPoint() && chart());
        return {match: svg.dataset.match, point: shownPoint()};
    })()
    """ % BROWSER_FUNCTIONS
    state = browser.evaluate(initialization)
    if state["point"] != 1:
        raise RuntimeError(f"El benchmark debe iniciar en el primer punto: {state}")
    match = state["match"]
    points = pd.read_parquet(CLEAN_DATASET, columns=["Gm#"], filters=[("match_id", "=", match)])
    ends = points.groupby("Gm#", sort=True).size().cumsum()
    if len(ends) < limit + 2:
        raise ValueError("El partido no tiene suficientes cierres.")
    results = []
    for game_num, position in ends.iloc[:limit + 1].items():
        script = """
        (async () => {
            %s
            for (let point = shownPoint() + 1; point <= %d; point++) {
                const closing = point === %d;
                const started = performance.now();
                next().click();
                await waitUntil(() => shownPoint() === point);
                if (closing) {
                    const svg = await waitUntil(() => {
                        const current = chart();
                        return Number(current?.dataset.completed) === %d
                            && [...current.querySelectorAll("title")].some(
                                item => item.textContent.startsWith("Pronóstico para el juego ")) && current;
                    });
                    await frame();
                    await frame();
                    const first = performance.now() - started;
                    await Promise.all(svg.getAnimations({subtree: true}).map(anim => anim.finished));
                    await frame();
                    return {first, settled: performance.now() - started};
                }
            }
            throw Error("No se alcanzó un cierre");
        })()
        """ % (BROWSER_FUNCTIONS, position, position, game_num)
        event = browser.evaluate(script)
        results.append({"game": int(game_num), **event})
        print(f"{match} juego {game_num}: {event['first']:.1f} ms", flush=True)
    return match, results[1:]


def replay_stage_samples(match, games, backend_url):
    net = pd.read_parquet(
        MOMENTUM_DATASET, columns=["game_num", "N"], filters=[("match_id", "=", match)]
    ).sort_values("game_num").set_index("game_num")["N"]
    timings = []
    for game in games:
        started = perf_counter()
        with urlopen(f"{backend_url}/matches/{quote(match, safe='')}/games/{game}?timing=1", timeout=20) as reply:
            result = json.load(reply)
            values = dict(item.strip().split(";dur=") for item in reply.headers["Server-Timing"].split(","))
        if result["prediction"]["game_num"] != game:
            raise RuntimeError("El servicio respondió con un juego distinto.")
        http = (perf_counter() - started) * 1000
        started = perf_counter()
        momentum_crossings(net.loc[:game])
        alert = (perf_counter() - started) * 1000
        timings.append({"game": game, "http": http, "alerta": alert, **{k: float(v) for k, v in values.items()}})
    return timings


def summarize(rows):
    return {
        key: dict(zip(("p50", "p95"), np.percentile([row[key] for row in rows], [50, 95]).tolist()))
        for key in rows[0] if key != "game"
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chrome", default="http://127.0.0.1:18768")
    parser.add_argument("--dashboard", default="http://127.0.0.1:18770")
    parser.add_argument("--backend", default="http://127.0.0.1:18769")
    parser.add_argument("--closures", type=int, default=30)
    args = parser.parse_args()
    if args.closures < 1:
        parser.error("--closures debe ser al menos 1")

    with urlopen(f"{args.chrome}/json/list", timeout=5) as response:
        tab = next(tab for tab in json.load(response) if tab["type"] == "page")
    with connect(tab["webSocketDebuggerUrl"]) as connection:
        browser = Browser(connection)
        version = browser.command("Browser.getVersion", {})["product"]
        match, visual = browser_samples(browser, args.dashboard, args.closures)

    stages = replay_stage_samples(match, [sample["game"] for sample in visual], args.backend)
    hardware = subprocess.check_output(["system_profiler", "SPHardwareDataType"], text=True)
    result = {
        "match_id": match, "samples": len(visual),
        "hardware": [line.strip() for line in hardware.splitlines() if line.strip().startswith(
            ("Model Name:", "Model Identifier:", "Chip:", "Memory:", "Total Number of Cores:")
        )],
        "os": platform.platform(), "browser": version,
        "summary_ms": summarize(visual) | summarize(stages),
        "visual": visual, "stages": stages,
    }
    path = Path(PROCESSED_DIR) / "r8_replay_latency.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
