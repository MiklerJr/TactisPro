/* === DATA === */
const momentumA = [0, 0.10, 0.22, 0.35, 0.18, 0.42, 0.55, 0.47, 0.62, 0.72,
  0.65, 0.58, 0.70, 0.80, 0.72, 0.68, 0.75, 0.82, 0.78, 0.85,
  0.72, 0.65, 0.71, 0.88, 0.91, 0.84, 0.88, 0.92, 0.85, 0.78,
  0.82, 0.88, 0.91, 0.85, 0.80, 0.72];

const momentumB = [0, 0.50, 0.55, 0.48, 0.62, 0.58, 0.45, 0.52, 0.42, 0.35,
  0.42, 0.50, 0.38, 0.28, 0.35, 0.42, 0.38, 0.25, 0.30, 0.22,
  0.35, 0.44, 0.38, 0.20, 0.15, 0.22, 0.18, 0.12, 0.20, 0.28,
  0.20, 0.15, 0.12, 0.25, 0.38, 0.41];

const turningPoints = [
  { point: 17, label: 'Crisis de Bauer', color: '#F59E0B' },
  { point: 23, label: 'Break Mendoza', color: '#10B981' },
  { point: 30, label: 'Fatiga Bauer', color: '#F59E0B' },
];

const labels = momentumA.map((_, i) => `P${i + 1}`);

/* === CHART === */
let chart;
let currentView = 'all';

function buildChart(view) {
  const ctx = document.getElementById('momentumChart').getContext('2d');

  let dataA = [...momentumA];
  let dataB = [...momentumB];
  let lbs = [...labels];

  if (view === 'recent') {
    dataA = dataA.slice(-15);
    dataB = dataB.slice(-15);
    lbs = lbs.slice(-15);
  }

  const splitIdx = Math.floor(dataA.length * 3 / 4);

  const tpIndices = turningPoints.map(tp => view === 'recent' ? tp.point - (momentumA.length - 15) : tp.point);

  function predictionStyle(idx) {
    return idx >= splitIdx;
  }

  function confidenceInterval(arr, spread) {
    return arr.map((v, i) => i >= splitIdx ? v + spread : null);
  }

  if (chart) chart.destroy();

  chart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: lbs,
      datasets: [
        /* Mendoza — actual + prediction line */
        {
          label: 'Mendoza (A)',
          data: dataA,
          borderColor: '#2563EB',
          backgroundColor: 'rgba(37,99,235,0.08)',
          tension: 0.42,
          fill: true,
          segment: {
            borderDash: ctx => predictionStyle(ctx.p0DataIndex) ? [6, 4] : [],
            borderWidth: ctx => predictionStyle(ctx.p0DataIndex) ? 1.4 : 2,
          },
          pointRadius: ctx => predictionStyle(ctx.dataIndex) ? 2 : 2.5,
          pointHoverRadius: ctx => predictionStyle(ctx.dataIndex) ? 4 : 5,
        },
        /* Mendoza — confidence upper band */
        {
          label: null,
          data: confidenceInterval(dataA, 0.07),
          borderColor: 'rgba(37,99,235,0.5)',
          backgroundColor: 'rgba(37,99,235,0.12)',
          tension: 0.35,
          fill: '+1',
          pointRadius: 0,
          pointHoverRadius: 0,
          borderWidth: 0.8,
          borderDash: [3, 3],
        },
        /* Mendoza — confidence lower band */
        {
          label: null,
          data: confidenceInterval(dataA, -0.07),
          borderColor: 'rgba(37,99,235,0.5)',
          backgroundColor: 'transparent',
          tension: 0.35,
          fill: false,
          pointRadius: 0,
          pointHoverRadius: 0,
          borderWidth: 0.8,
          borderDash: [3, 3],
        },
        /* Bauer — actual + prediction line */
        {
          label: 'Bauer (B)',
          data: dataB,
          borderColor: '#D97706',
          backgroundColor: 'rgba(217,119,6,0.06)',
          tension: 0.42,
          fill: true,
          segment: {
            borderDash: ctx => predictionStyle(ctx.p0DataIndex) ? [6, 4] : [],
            borderWidth: ctx => predictionStyle(ctx.p0DataIndex) ? 1.4 : 2,
          },
          pointRadius: ctx => predictionStyle(ctx.dataIndex) ? 2 : 2.5,
          pointHoverRadius: ctx => predictionStyle(ctx.dataIndex) ? 4 : 5,
        },
        /* Bauer — confidence upper band */
        {
          label: null,
          data: confidenceInterval(dataB, 0.07),
          borderColor: 'rgba(217,119,6,0.5)',
          backgroundColor: 'rgba(217,119,6,0.12)',
          tension: 0.35,
          fill: '+1',
          pointRadius: 0,
          pointHoverRadius: 0,
          borderWidth: 0.8,
          borderDash: [3, 3],
        },
        /* Bauer — confidence lower band */
        {
          label: null,
          data: confidenceInterval(dataB, -0.07),
          borderColor: 'rgba(217,119,6,0.5)',
          backgroundColor: 'transparent',
          tension: 0.35,
          fill: false,
          pointRadius: 0,
          pointHoverRadius: 0,
          borderWidth: 0.8,
          borderDash: [3, 3],
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#FFFFFF',
          borderColor: '#E2E8F0',
          borderWidth: 1,
          titleColor: '#4A5568',
          bodyColor: '#1A202C',
          padding: 10,
          filter: item => item.dataset.label !== null,
          callbacks: {
            title: items => `${items[0].label}`,
            label: item => ` ${item.dataset.label}: ${item.parsed.y.toFixed(2)}`
          }
        }
      },
      scales: {
        x: {
          grid: { color: '#E2E8F0', lineWidth: 0.5 },
          ticks: {
            color: '#A0AEC0',
            font: { size: 10, family: 'JetBrains Mono' },
            maxTicksLimit: view === 'recent' ? 15 : 12,
            maxRotation: 0,
          },
          border: { color: '#E2E8F0' }
        },
        y: {
          grid: { color: '#E2E8F0', lineWidth: 0.5 },
          ticks: {
            color: '#A0AEC0',
            font: { size: 10, family: 'JetBrains Mono' },
            callback: v => v.toFixed(1)
          },
          min: -0.15,
          max: 1.1,
          title: {
            display: true,
            text: 'Momentum Index',
            color: '#A0AEC0',
            font: { size: 10, family: 'JetBrains Mono' }
          },
          border: { color: '#E2E8F0' }
        }
      },
      animation: {
        onComplete: function () { drawTurningPointLines(this, tpIndices, splitIdx - 0.5); }
      }
    }
  });
}

function drawTurningPointLines(chartInstance, tpIndices, splitLineIdx) {
  const { ctx, chartArea, scales } = chartInstance;
  if (!chartArea) return;
  ctx.save();
  tpIndices.forEach((idx) => {
    if (idx < 0) return;
    const x = scales.x.getPixelForValue(idx);
    ctx.beginPath();
    ctx.strokeStyle = '#F59E0B';
    ctx.lineWidth = 1;
    ctx.globalAlpha = 0.6;
    ctx.setLineDash([4, 3]);
    ctx.moveTo(x, chartArea.top);
    ctx.lineTo(x, chartArea.bottom);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
  });
  if (splitLineIdx !== undefined) {
    const x = scales.x.getPixelForValue(splitLineIdx);
    ctx.beginPath();
    ctx.strokeStyle = '#94A3B8';
    ctx.lineWidth = 1;
    ctx.globalAlpha = 0.7;
    ctx.setLineDash([3, 3]);
    ctx.moveTo(x, chartArea.top);
    ctx.lineTo(x, chartArea.bottom);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
  }
  ctx.restore();
}

function setView(view, btn) {
  currentView = view;
  document.querySelectorAll('.card-actions .chip').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  buildChart(view);
}

/* === SURFACE SELECTOR === */
document.querySelectorAll('.surf-btn').forEach(btn => {
  btn.addEventListener('click', function () {
    document.querySelectorAll('.surf-btn').forEach(b => b.classList.remove('active'));
    this.classList.add('active');
  });
});

/* === LIVE CLOCK === */
let seconds = 83 * 60;
setInterval(() => {
  seconds++;
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  const el = document.getElementById('matchTime');
  if (el) el.textContent = `${m}h ${String(s).padStart(2, '0')}m`;
}, 1000);

/* === SCORE ANIMATION (demo) === */
const gameScores = ['0', '15', '30', '40', 'ADV', '40'];
let scoreIdx = 3;
setInterval(() => {
  scoreIdx = (scoreIdx + 1) % 5;
  const el = document.getElementById('scoreA');
  if (el) el.textContent = gameScores[scoreIdx];
}, 8000);

/* === INIT === */
document.addEventListener('DOMContentLoaded', () => {
  buildChart('all');
});

window.setView = setView;
