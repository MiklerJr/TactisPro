const NS = "http://www.w3.org/2000/svg";

export default function render({ parentElement, data }) {
    const svg = parentElement.querySelector("svg");
    const fingerprint = JSON.stringify(data);
    if (svg.dataset.rendered === fingerprint) return;
    svg.dataset.rendered = fingerprint;
    const previous = Number(svg.dataset.completed || 0);
    const animate = svg.dataset.match === data.match
        && data.completed === previous + 1
        && !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    svg.dataset.match = data.match;
    svg.dataset.completed = data.completed;
    svg.replaceChildren();
    svg.setAttribute("aria-label", `Momentum y pronóstico. ${data.completed} de ${data.total} juegos cerrados.`);

    function node(tag, attributes, text, parent = svg) {
        const item = document.createElementNS(NS, tag);
        for (const [key, value] of Object.entries(attributes)) item.setAttribute(key, value);
        if (text != null) item.textContent = text;
        parent.append(item);
        return item;
    }

    const colors = data.palette;
    svg.style.setProperty("--chart-muted", colors.muted);
    svg.style.setProperty("--chart-border", colors.border);
    const left = 69, right = 731, top = 48, bottom = 326;
    const endGame = data.full ? data.total : Math.min(data.total, Math.max(18, data.completed + 4));
    const startGame = data.full ? 1 : Math.max(1, endGame - 17);
    const first = startGame - 0.5, last = Math.max(2, endGame) + 0.5;
    const x = game => left + (game - first) / (last - first) * (right - left);
    const y = value => top + (1 - value) / 2 * (bottom - top);

    node("rect", { width: 760, height: 390, rx: 16, fill: colors.background });
    node("rect", { x: left, y: top, width: right-left, height: y(0)-top, fill: colors.a, "fill-opacity": 0.035 });
    node("rect", { x: left, y: y(0), width: right-left, height: bottom-y(0), fill: colors.b, "fill-opacity": 0.045 });
    node("text", { x: left, y: 31, class: "zone", style: `fill:${colors.a}` }, `+ ${data.players[0]}`);
    node("text", { x: right, y: 31, class: "zone", "text-anchor": "end", style: `fill:${colors.b}` }, `− ${data.players[1]}`);

    for (const value of [-1, -0.5, 0, 0.5, 1]) {
        node("line", {
            x1:left, x2:right, y1:y(value), y2:y(value), class:"grid",
            style:value === 0 ? "stroke:#8e8376;stroke-dasharray:5 4" : "",
        });
        node("text", { x:left-10, y:y(value)+6, class:"tick", "text-anchor":"end" },
            value > 0 ? `+${value}` : String(value));
    }
    const stride = Math.max(1, Math.ceil((endGame - startGame + 1) / 8));
    for (let game = startGame; game <= endGame; game += stride) {
        node("text", { x:x(game), y:352, class:"tick", "text-anchor":"middle" }, game);
    }
    node("text", { x:(left+right)/2, y:379, class:"label", "text-anchor":"middle" }, "Juego");

    if (data.completed === 0) {
        node("text", { x:(left+right)/2, y:y(0.18), class:"message", "text-anchor":"middle" },
            "La curva empieza al cerrar el primer juego.");
    }

    function drawLine(points, series, reveal = false) {
        if (points.length < 2) return;
        const path = node("path", {
            class:"trace", stroke:series.color, "stroke-width":series.width,
            d:points.map(([game, value], i) => `${i ? "L" : "M"} ${x(game)} ${y(value)}`).join(" "),
            ...(series.dashed ? { "stroke-dasharray":"7 5" } : {}),
        });
        if (reveal && !series.dashed) {
            const length = path.getTotalLength();
            path.animate([{ strokeDasharray:`${length}`, strokeDashoffset:length },
                          { strokeDasharray:`${length}`, strokeDashoffset:0 }],
                         { duration:300, easing:"ease-out" });
        } else if (reveal) {
            path.animate([{ opacity:0 }, { opacity:1 }], { duration:260 });
        }
    }

    for (const series of data.series) {
        const points = series.points.filter(([game, value]) => game >= startGame && game <= endGame && Number.isFinite(value));
        if (animate && !series.forecast && points.at(-1)?.[0] === data.completed) {
            drawLine(points.slice(0,-1), series);
            drawLine(points.slice(-2), series, true);
        } else {
            drawLine(points, series, animate && series.forecast);
        }
        for (const [game, value] of points) {
            if (series.forecast && game === data.completed) continue;
            const active = game === data.completed || Boolean(series.forecast);
            const dot = node("circle", { cx:x(game), cy:y(value), r:active ? 5.5 : 3,
                fill:series.forecast ? colors.background : series.color, stroke:series.color,
                "stroke-width":series.forecast ? 2.5 : 1 });
            const label = `${series.label} · Juego ${game} · ${value.toFixed(3)}`;
            node("title", {}, label, dot);
            if (active && animate) dot.animate([{ opacity:0 },{ opacity:1 }], { duration:320 });
        }
    }
}
