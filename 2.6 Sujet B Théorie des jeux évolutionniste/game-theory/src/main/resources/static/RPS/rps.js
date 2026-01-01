/*********************************************************
 * TYPES
 *********************************************************/
const Types = Object.freeze({
    ROCK: "R",
    PAPER: "P",
    SCISSOR: "S"
});

/*********************************************************
 * DOM / CANVAS
 *********************************************************/
const board = document.getElementById("board");
const ctx = board.getContext("2d");
const chartCanvas = document.getElementById("chart");

const rockInput = document.getElementById("rockInput");
const paperInput = document.getElementById("paperInput");
const scissorInput = document.getElementById("scissorInput");

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const resetBtn = document.getElementById("resetBtn");

/*********************************************************
 * CHART
 *********************************************************/
const chart = new Chart(chartCanvas, {
    type: "line",
    data: {
        labels: [],
        datasets: [
            { label: "Rock", borderColor: "red", data: [] },
            { label: "Paper", borderColor: "blue", data: [] },
            { label: "Scissors", borderColor: "green", data: [] }
        ]
    },
    options: {
        responsive: false,
        plugins: { legend: { position: "top" } },
        scales: {
            x: { title: { display: true, text: "Day" } },
            y: { title: { display: true, text: "Count" } }
        }
    }
});

/*********************************************************
 * ÉTAT GLOBAL
 *********************************************************/
const center = { x: 300, y: 300 };
const perimeterRadius = 250;
const circles = [100, 160, 220]; // points de rencontre
let points = [];
let players = [];
let day = 0;
let loop = null;
let animating = false;

// stocker la config initiale pour les zones
let initialConfig = { rock: 12, paper: 12, scissor: 12 };

/*********************************************************
 * UTILITAIRES
 *********************************************************/
function wait(ms) { return new Promise(res => setTimeout(res, ms)); }
function count(type) { return players.filter(p => p.type === type).length; }

/*********************************************************
 * INITIALISATION
 *********************************************************/
function createPlayer(type, angle) {
    const x = center.x + perimeterRadius * Math.cos(angle);
    const y = center.y + perimeterRadius * Math.sin(angle);
    return { type, x, y, startX: x, startY: y, target: null };
}

function generatePoints() {
    points = [];
    const config = [
        { r: 100, pairs: 4 },
        { r: 160, pairs: 6 },
        { r: 220, pairs: 8 }
    ];
    const offset = 12;

    config.forEach(({ r, pairs }) => {
        for (let i = 0; i < pairs; i++) {
            const angle = (2 * Math.PI * i) / pairs;
            const cx = center.x + r * Math.cos(angle);
            const cy = center.y + r * Math.sin(angle);
            points.push({ x: cx, y: cy, assigned: [], offset });
        }
    });
}

function initPlayers(rock, paper, scissor) {
    initialConfig = { rock, paper, scissor }; // mémoriser config de départ
    players = [];
    generatePoints();
    const arr = [
        ...Array(rock).fill(Types.ROCK),
        ...Array(paper).fill(Types.PAPER),
        ...Array(scissor).fill(Types.SCISSOR)
    ];
    arr.forEach((type, i) => {
        const angle = (2 * Math.PI * i) / 36;
        players.push(createPlayer(type, angle));
    });
    resetChart();
    draw();
}

/*********************************************************
 * DESSIN
 *********************************************************/
function draw() {
    ctx.clearRect(0, 0, board.width, board.height);

    const totalPlayers = 36;
    const anglePerPlayer = (2 * Math.PI) / totalPlayers;

    // zones fixes
    const zones = [
        { type: Types.ROCK, color: "red", start: 0, count: initialConfig.rock },      // <-- rouge
        { type: Types.PAPER, color: "blue", start: initialConfig.rock, count: initialConfig.paper },
        { type: Types.SCISSOR, color: "green", start: initialConfig.rock + initialConfig.paper, count: initialConfig.scissor } // <-- vert
    ];

    zones.forEach(z => {
        const minAngle = z.start * anglePerPlayer - anglePerPlayer / 2;
        const maxAngle = (z.start + z.count - 1) * anglePerPlayer + anglePerPlayer / 2;
        ctx.beginPath();
        ctx.arc(center.x, center.y, perimeterRadius + 20, minAngle, maxAngle);
        ctx.strokeStyle = z.color;
        ctx.lineWidth = 8;
        ctx.stroke();
    });

    players.forEach(p => {
        ctx.beginPath();
        ctx.arc(p.x, p.y, 8, 0, 2 * Math.PI);
        ctx.fillStyle = p.type === Types.ROCK ? "red" : p.type === Types.PAPER ? "blue" : "green";
        ctx.fill();
        ctx.strokeStyle = "black";
        ctx.lineWidth = 1;
        ctx.stroke();

        ctx.fillStyle = "black";
        ctx.font = "9px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(p.type, p.x, p.y);
    });

    // points de rencontre
    points.forEach(pt => {
        ctx.fillStyle = "black";
        ctx.fillRect(pt.x - 4, pt.y - 4, 8, 8);
    });
}

/*********************************************************
 * DEBUT DE TOUR
 *********************************************************/
function animateMove(arr, speed=5){
    return new Promise(resolve=>{
        function step(){
            let done = true;
            arr.forEach(p=>{
                if(!p.target) return;
                const dx = p.target.x - p.x;
                const dy = p.target.y - p.y;
                const dist = Math.hypot(dx,dy);
                if(dist>speed){
                    p.x += dx/dist*speed;
                    p.y += dy/dist*speed;
                    done = false;
                } else {
                    p.x = p.target.x;
                    p.y = p.target.y;
                }
            });
            draw();
            if(done) resolve();
            else requestAnimationFrame(step);
        }
        step();
    });
}

/*********************************************************
 * RPS LOGIC
 *********************************************************/
function applyRPS(a,b){
    if(a.type===b.type) return;
    if((a.type===Types.ROCK && b.type===Types.SCISSOR) ||
        (a.type===Types.PAPER && b.type===Types.ROCK) ||
        (a.type===Types.SCISSOR && b.type===Types.PAPER)){
        b.type=a.type;
    } else {
        a.type=b.type;
    }
}

/*********************************************************
 * FIN DE TOUR
 *********************************************************/
async function stepDay() {
    if (animating) return;
    animating = true;
    day++;

    points.forEach(pt => pt.assigned = []);

    players.forEach(p => {
        const freePoints = points.filter(pt => !pt.assigned || pt.assigned.length < 2);
        if (freePoints.length === 0) return;
        const pt = freePoints[Math.floor(Math.random() * freePoints.length)];
        pt.assigned = pt.assigned || [];
        pt.assigned.push(p);

        const offset = pt.assigned.length === 1 ? -12 : 12;
        p.target = { x: pt.x + offset, y: pt.y, ref: pt };
    });

    const results = [];
    points.forEach(pt => {
        if (pt.assigned && pt.assigned.length === 2) {
            results.push([...pt.assigned]);
        }
    });

    await animateMove(players);
    await wait(500);

    players.forEach(p => p.target = { x: p.startX, y: p.startY });
    await animateMove(players);

    results.forEach(pair => applyRPS(pair[0], pair[1]));

    updateChart();
    animating = false;
}

/*********************************************************
 * CHART
 *********************************************************/
function resetChart(){
    chart.data.labels=[0];
    chart.data.datasets[0].data=[count(Types.ROCK)];
    chart.data.datasets[1].data=[count(Types.PAPER)];
    chart.data.datasets[2].data=[count(Types.SCISSOR)];
    chart.update();
}

function updateChart(){
    chart.data.labels.push(day);
    chart.data.datasets[0].data.push(count(Types.ROCK));
    chart.data.datasets[1].data.push(count(Types.PAPER));
    chart.data.datasets[2].data.push(count(Types.SCISSOR));
    chart.update();
}

/*********************************************************
 * CONTROLES
 *********************************************************/
function adjustSum(){
    let rock = +rockInput.value;
    let paper = +paperInput.value;
    let scissor = +scissorInput.value;
    let total = rock + paper + scissor;
    if(total===36) return;

    if(total>36){
        let diff = total-36;
        if(scissor>=diff) scissor-=diff;
        else if(paper>=diff) paper-=diff;
        else rock-=diff;
    } else if(total<36){
        let diff = 36-total;
        scissor += diff;
    }
    rockInput.value = rock; paperInput.value = paper; scissorInput.value = scissor;
}

rockInput.oninput = paperInput.oninput = scissorInput.oninput = () => {
    adjustSum();
    initPlayers(+rockInput.value,+paperInput.value,+scissorInput.value);
};

startBtn.onclick = () => { if(!loop) loop = setInterval(stepDay,2000); };
stopBtn.onclick = () => { clearInterval(loop); loop=null; };
resetBtn.onclick = () => { stopBtn.onclick(); initPlayers(+rockInput.value,+paperInput.value,+scissorInput.value); day=0; };

window.onload = ()=>{ initPlayers(12,12,12); };