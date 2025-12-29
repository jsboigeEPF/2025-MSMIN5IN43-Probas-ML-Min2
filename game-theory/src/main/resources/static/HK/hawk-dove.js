const boardCanvas = document.getElementById('board');
const ctx = boardCanvas.getContext('2d');
const chartCanvas = document.getElementById('chart');

const chart = new Chart(chartCanvas, {
  type: 'line',
  data: {
    labels: [],
    datasets: [
      { label: 'Hawks', borderColor: 'red', data: [] },
      { label: 'Doves', borderColor: 'blue', data: [] }
    ]
  },
  options: {
    responsive: false,
    plugins: { legend: { position: 'top' } },
    scales: {
      x: { title: { display: true, text: 'Day' } },
      y: { title: { display: true, text: 'Population' } }
    }
  }
});

let creatures = [], foods = [], day = 0, animating = false, intervalId = null, nextCreatureId = 0;
const radius = 280, center = { x: 300, y: 300 };

const hawksSlider = document.getElementById('hawksSlider');
const dovesSlider = document.getElementById('dovesSlider');
const hawksVal = document.getElementById('hawksVal');
const dovesVal = document.getElementById('dovesVal');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const resetBtn = document.getElementById('resetBtn');

hawksSlider.oninput = () => { hawksVal.textContent = hawksSlider.value; resetCreaturesAndChart(); };
dovesSlider.oninput = () => { dovesVal.textContent = dovesSlider.value; resetCreaturesAndChart(); };

// ---------------------------
// INITIALISATION DES CREATURES
// ---------------------------
function initCreatures(hawks, doves) {
  creatures = [];
  const total = parseInt(hawks) + parseInt(doves);
  for (let i = 0; i < total; i++) {
    const angle = (2 * Math.PI * i) / total;
    const type = i < hawks ? "HAWK" : "DOVE";
    creatures.push({
      id: nextCreatureId++,
      type,
      angle,
      x: center.x + radius * Math.cos(angle),
      y: center.y + radius * Math.sin(angle),
      startX: center.x + radius * Math.cos(angle),
      startY: center.y + radius * Math.sin(angle),
      target: null,
      state: "atPerimeter",
      food: 0,
      alpha: 1
    });
  }
  draw();
}

function resetCreaturesAndChart() {
  day = 0;
  generateFoodPairsStructured(); // génère nourriture et ajuste sliders
  initCreatures(hawksSlider.value, dovesSlider.value);
  chart.data.labels = [day];
  chart.data.datasets[0].data = [creatures.filter(c => c.type === "HAWK").length];
  chart.data.datasets[1].data = [creatures.filter(c => c.type === "DOVE").length];
  chart.update();
}

// ---------------------------
// DESSIN
// ---------------------------
function draw() {
  ctx.clearRect(0, 0, boardCanvas.width, boardCanvas.height);
  ctx.beginPath();
  ctx.arc(center.x, center.y, radius, 0, 2 * Math.PI);
  ctx.stroke();

  foods.forEach(pair => {
    pair.forEach(f => {
      if (!f.eaten) {
        ctx.beginPath();
        ctx.arc(f.x, f.y, 7, 0, 2 * Math.PI);
        ctx.fillStyle = 'green';
        ctx.fill();
      }
    });
  });

  creatures.forEach(c => {
    ctx.beginPath();
    ctx.arc(c.x, c.y, 8, 0, 2 * Math.PI);
    ctx.fillStyle = c.type === "HAWK" ? `rgba(255,0,0,${c.alpha})` : `rgba(0,0,255,${c.alpha})`;
    ctx.fill();
  });
}

// ---------------------------
// MOUVEMENT
// ---------------------------
function moveTowards(obj, target, speed) {
  const dx = target.x - obj.x, dy = target.y - obj.y;
  const dist = Math.sqrt(dx * dx + dy * dy);
  if (dist < speed) { obj.x = target.x; obj.y = target.y; return true; }
  obj.x += dx / dist * speed;
  obj.y += dy / dist * speed;
  return false;
}

// ---------------------------
// NOURRITURE & MAX CREATURES
// ---------------------------
function generateFoodPairsStructured() {
  foods = [];
  const circles = [
    { radius: 100, pairs: 4 },
    { radius: 150, pairs: 6 },
    { radius: 200, pairs: 8 }
  ];
  const offset = 10;
  let totalPairs = 0;

  circles.forEach(circle => {
    const { radius, pairs } = circle;
    totalPairs += pairs;
    for (let i = 0; i < pairs; i++) {
      const angle = (2 * Math.PI / pairs) * i;
      const xCenter = center.x + radius * Math.cos(angle);
      const yCenter = center.y + radius * Math.sin(angle);
      const x1 = xCenter + offset * Math.cos(angle + Math.PI / 2);
      const y1 = yCenter + offset * Math.sin(angle + Math.PI / 2);
      const x2 = xCenter + offset * Math.cos(angle - Math.PI / 2);
      const y2 = yCenter + offset * Math.sin(angle - Math.PI / 2);
      foods.push([{ x: x1, y: y1, eaten: false, angle, r: radius }, { x: x2, y: y2, eaten: false, angle, r: radius }]);
    }
  });

  // --- Ajuster dynamiquement les sliders ---
  const maxCreatures = totalPairs * 2;
  hawksSlider.max = maxCreatures;
  dovesSlider.max = maxCreatures;
  if (hawksSlider.value > maxCreatures) hawksSlider.value = maxCreatures;
  if (dovesSlider.value > maxCreatures) dovesSlider.value = maxCreatures;
  hawksVal.textContent = hawksSlider.value;
  dovesVal.textContent = dovesSlider.value;
}

function assignTargetsToFoodStructured() {
  // On commence par mélanger les créatures
  const shuffledCreatures = creatures.slice().sort(() => Math.random() - 0.5);

  // Réinitialiser la capacité de chaque paire (max 2)
  foods.forEach(pair => pair.assignedCount = 0);

  shuffledCreatures.forEach(c => {
    // Trouver une paire qui n'est pas complète
    const availablePairs = foods.filter(pair => pair.assignedCount < 2);
    if (availablePairs.length === 0) {
      console.warn("Pas assez de paires pour toutes les créatures !");
      return;
    }

    // Choisir une paire aléatoire parmi celles disponibles
    const pair = availablePairs[Math.floor(Math.random() * availablePairs.length)];
    pair.assignedCount++;

    // Déterminer l’offset de la créature dans la paire
    const offset = pair.assignedCount === 1 ? -15 : 15;
    const xCenter = (pair[0].x + pair[1].x) / 2;
    const yCenter = (pair[0].y + pair[1].y) / 2;
    const angle = Math.atan2(center.y - yCenter, center.x - xCenter);
    const radiusAdjust = offset > 0 ? -5 : 5;

    c.target = {
      x: xCenter + (offset + radiusAdjust) * Math.cos(angle),
      y: yCenter + (offset + radiusAdjust) * Math.sin(angle),
      pair,
      offset
    };
    c.state = "toFood";
    c.food = 0;
  });
}

function eatFood(creature) {
  const pair = creature.target.pair;
  if (pair.every(f => f.eaten)) return;
  const others = creatures.filter(c => c !== creature && c.target && c.target.pair === pair);
  const other = others[0] || null;

  let foodTaken = 0;
  if (!other) foodTaken = 2;
  else if (creature.type === "DOVE" && other.type === "DOVE") { foodTaken = 1; if (other) other.food = 1; }
  else if (creature.type === "HAWK" && other.type === "HAWK") { foodTaken = 1; creature.food = 0; if (other) other.food = 0; }
  else if (creature.type === "HAWK" && other.type === "DOVE") { foodTaken = 1.5; if (other) other.food = 0.5; }
  else if (creature.type === "DOVE" && other.type === "HAWK") { foodTaken = 0.5; if (other) other.food = 1.5; }

  creature.food = foodTaken;
  pair.forEach(f => f.eaten = true);
}

function applyMortReproduction() {
  let survivors = [], children = [];
  creatures.forEach(c => {
    let survives = false, reproduce = false;

    if (c.food >= 2) { survives = true; reproduce = true; }
    else if (c.food >= 1.5) { survives = true; reproduce = Math.random() < 0.5; }
    else if (c.food >= 1) survives = true;
    else if (c.food >= 0.5) reproduce = Math.random() < 0.5;

    if (survives) survivors.push(c);
    if (reproduce) {
      children.push({
        ...c,
        id: nextCreatureId++,
        alpha: 1,
        x: c.x,
        y: c.y,
        startX: c.x,
        startY: c.y
      });
    }
  });
  return [...survivors, ...children];
}

// ---------------------------
// ANIMATION
// ---------------------------
async function animateStep(state) {
  return new Promise(resolve => {
    const speed = 4;
    function loop() {
      let done = true;
      creatures.forEach(c => {
        if (c.state === state && c.target) {
          if (!moveTowards(c, c.target, speed)) done = false;
        }
      });
      draw();
      if (!done) requestAnimationFrame(loop);
      else resolve();
    }
    loop();
  });
}

// ---------------------------
// FIN DE JOUR
// ---------------------------
async function animateEndOfDay(newCreatures) {
  const total = newCreatures.length;
  const hawks = newCreatures.filter(c => c.type === "HAWK");
  const doves = newCreatures.filter(c => c.type === "DOVE");
  let positions = [];
  const stepAngle = 2 * Math.PI / total;
  let angleCursor = 0;

  function assignEquidistant(group) {
    group.forEach(c => {
      c.targetAngle = angleCursor;
      c.angle = Math.atan2(c.y - center.y, c.x - center.x);
      c.alpha = 1;
      positions.push(c);
      angleCursor += stepAngle;
    });
  }

  assignEquidistant(hawks);
  assignEquidistant(doves);
  creatures = newCreatures.slice();

  return new Promise(resolve => {
    function loop() {
      let done = true;
      creatures.forEach(c => {
        let diff = c.targetAngle - c.angle;
        if (diff > Math.PI) diff -= 2 * Math.PI;
        else if (diff < -Math.PI) diff += 2 * Math.PI;
        if (Math.abs(diff) > 0.001) { c.angle += diff * 0.2; done = false; }
        c.x = center.x + radius * Math.cos(c.angle);
        c.y = center.y + radius * Math.sin(c.angle);
      });
      draw();
      if (!done) requestAnimationFrame(loop);
      else {
        creatures.forEach(c => { c.startX = c.x; c.startY = c.y; c.angle = c.targetAngle; });
        resolve();
      }
    }
    loop();
  });
}

// ---------------------------
// CHART & STEP DAY
// ---------------------------
function updateChart() {
  chart.data.labels.push(day);
  chart.data.datasets[0].data.push(creatures.filter(c => c.type === "HAWK").length);
  chart.data.datasets[1].data.push(creatures.filter(c => c.type === "DOVE").length);
  chart.update();
}

async function stepDay() {
  if (animating) return;
  animating = true;
  day++;

  generateFoodPairsStructured();
  assignTargetsToFoodStructured();
  await animateStep("toFood");
  creatures.forEach(c => { if (c.state === "toFood") eatFood(c); });
  creatures.forEach(c => { c.target = { x: c.startX, y: c.startY }; c.state = "toPerimeter"; });
  await animateStep("toPerimeter");

  const newCreatures = applyMortReproduction();
  await animateEndOfDay(newCreatures);

  updateChart();
  animating = false;
}

// ---------------------------
// BOUTONS
// ---------------------------
startBtn.onclick = () => { if (!intervalId) intervalId = setInterval(() => stepDay(), 3000); };
stopBtn.onclick = () => { if (intervalId) { clearInterval(intervalId); intervalId = null; } };
resetBtn.onclick = () => { if (intervalId) { clearInterval(intervalId); intervalId = null; } resetCreaturesAndChart(); };

// ---------------------------
// INIT
// ---------------------------
window.onload = () => resetCreaturesAndChart();