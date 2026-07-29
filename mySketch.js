let video;
let smoothPixels = [];
let backgroundPixels = [];
let historyMap = [];
let isBgInitialized = false;

// Sensitivity Parameters (Mapped from UI)
let paramDetail = 50;
let paramAbstraction = 50;
let paramSpike = 50;
let paramThickness = 3.5;
let paramSilhouette = 50;
let isCameraActive = true;

const V_WIDTH = 160;
const V_HEIGHT = 120;

function setup() {
    let container = document.getElementById('p5-canvas-container');
    let canvas = createCanvas(windowWidth, windowHeight);
    canvas.parent('p5-canvas-container');

    video = createCapture(VIDEO);
    video.size(V_WIDTH, V_HEIGHT);
    video.hide();

    smoothPixels = new Float32Array(V_WIDTH * V_HEIGHT);
    backgroundPixels = new Float32Array(V_WIDTH * V_HEIGHT);
    historyMap = new Float32Array(V_WIDTH * V_HEIGHT);

    strokeJoin(MITER);
    strokeCap(SQUARE);

    // Link UI Sliders
    document.getElementById('detail-slider').addEventListener('input', (e) => paramDetail = float(e.target.value));
    document.getElementById('abstraction-slider').addEventListener('input', (e) => paramAbstraction = float(e.target.value));
    document.getElementById('spike-slider').addEventListener('input', (e) => paramSpike = float(e.target.value));
    document.getElementById('thickness-slider').addEventListener('input', (e) => paramThickness = float(e.target.value));
    document.getElementById('silhouette-slider').addEventListener('input', (e) => paramSilhouette = float(e.target.value));

    // Camera Toggle
    document.getElementById('camera-btn').addEventListener('click', (e) => {
        isCameraActive = !isCameraActive;
        if (isCameraActive) {
            video.loop();
            e.target.innerText = "CAMERA STOP";
            e.target.style.backgroundColor = "#000";
        } else {
            video.pause();
            e.target.innerText = "CAMERA START";
            e.target.style.backgroundColor = "#ff0000";
        }
    });

    // Background Initialization
    setTimeout(() => {
        if (video.loadedmetadata) {
            video.loadPixels();
            for (let i = 0; i < V_WIDTH * V_HEIGHT; i++) {
                let idx = i << 2;
                backgroundPixels[i] = (video.pixels[idx] + video.pixels[idx + 1] + video.pixels[idx + 2]) / 3;
            }
            isBgInitialized = true;
        }
    }, 2000);
}

function draw() {
    background(255);
    if (!video.loadedmetadata) return;
    video.loadPixels();

    let pixCount = V_WIDTH * V_HEIGHT;
    let currentImportance = new Float32Array(pixCount);

    // Parameter Mapping
    let edgeThresholdHigh = map(paramDetail, 0, 100, 20, 3); // More sensitive for detail
    let edgeThresholdLow = map(paramDetail, 0, 100, 40, 8);
    let motionThreshold = map(paramSilhouette, 0, 100, 70, 10);

    for (let i = 0; i < pixCount; i++) {
        let idx = i << 2;
        let x = i % V_WIDTH;
        let y = floor(i / V_WIDTH);

        let rawB = (video.pixels[idx] + video.pixels[idx + 1] + video.pixels[idx + 2]) / 3;
        smoothPixels[i] = lerp(smoothPixels[i], rawB, 0.4);

        let isMotion = 0;
        if (isBgInitialized) {
            let diff = abs(smoothPixels[i] - backgroundPixels[i]);
            isMotion = diff > motionThreshold ? 1 : 0;
        }

        let importance = 0;
        if (x > 0 && y > 0 && x < V_WIDTH - 1 && y < V_HEIGHT - 1) {
            let b = smoothPixels[i];
            let bRight = smoothPixels[i + 1];
            let bDown = smoothPixels[i + V_WIDTH];
            let diff = abs(b - bRight) + abs(b - bDown);

            if (diff > edgeThresholdHigh) importance = 3;
            else if (diff > edgeThresholdLow) importance = 2;
            else if (isMotion) importance = 1;
        }

        // Temporal Persistence (Flicker Fix)
        // If movement is detected, use the high-importance signal.
        // If not, decay the history slowly to keep lines stable.
        let targetI = isMotion ? importance : 0;
        historyMap[i] = lerp(historyMap[i], targetI, 0.25);
        currentImportance[i] = historyMap[i];
    }

    // Draw setup
    let sc = max(width / V_WIDTH, height / V_HEIGHT);
    push();
    translate(width / 2 + (V_WIDTH * sc) / 2, height / 2 - (V_HEIGHT * sc) / 2);
    scale(-sc, sc);

    noFill();
    stroke(0);
    strokeWeight(paramThickness / sc);

    generateStrictOneStroke(currentImportance);

    pop();
}

/**
 * Strict single-path generation with dense texture logic.
 */
function generateStrictOneStroke(importanceMap) {
    let visited = new Uint8Array(importanceMap.length);

    let baseDist = map(paramSpike, 0, 100, 1, 8);
    // High point limit for dense scribbly look
    let totalPointsLimit = map(paramAbstraction, 0, 100, 400, 4000);

    // Starting point search (Prefer features > 2.0 weighted)
    let startIdx = -1;
    for (let threshold = 2.5; threshold >= 0.5; threshold -= 0.5) {
        for (let i = 0; i < importanceMap.length; i++) {
            if (importanceMap[i] >= threshold) {
                startIdx = i;
                break;
            }
        }
        if (startIdx !== -1) break;
    }

    if (startIdx === -1) return;

    let curX = startIdx % V_WIDTH;
    let curY = floor(startIdx / V_WIDTH);

    beginShape();
    for (let i = 0; i < totalPointsLimit; i++) {
        vertex(curX, curY);
        let idx = floor(curY) * V_WIDTH + floor(curX);
        if (idx >= 0 && idx < visited.length) visited[idx] = 1;

        // 1. Local Search (Stay in current area)
        let nextPoint = findNextPointLocal(importanceMap, visited, curX, curY, baseDist);

        // 2. Global Jump (If local fails, jump to nearest important area)
        if (!nextPoint) {
            nextPoint = findNearestGlobalJump(importanceMap, visited, curX, curY);
        }

        if (!nextPoint) break;

        curX = nextPoint.x;
        curY = nextPoint.y;
    }
    endShape();
}

/**
 * Standard greedy neighbor search.
 */
function findNextPointLocal(importanceMap, visited, x, y, d) {
    let angleStep = QUARTER_PI;
    let spikeOffset = paramSpike * 0.25;

    // Search for anything with moderate importance (>= 1.5)
    for (let threshold = 2.5; threshold >= 1.0; threshold -= 0.5) {
        for (let r = d; r < d * 8; r += 2) {
            for (let ang = 0; ang < TWO_PI; ang += angleStep) {
                let nx = x + cos(ang) * (r + random(-spikeOffset, spikeOffset));
                let ny = y + sin(ang) * (r + random(-spikeOffset, spikeOffset));
                let nIdx = floor(ny) * V_WIDTH + floor(nx);

                if (nx >= 0 && nx < V_WIDTH && ny >= 0 && ny < V_HEIGHT) {
                    if (importanceMap[nIdx] >= threshold && !visited[nIdx]) {
                        // Smaller visited radius for feature areas to allow dense scribbling
                        let visitR = map(importanceMap[nIdx], 1, 3, d * 0.6, d * 0.15);
                        markVisitedRadial(visited, nx, ny, visitR);
                        return { x: nx, y: ny };
                    }
                }
            }
        }
    }
    return null;
}

/**
 * Searches the entire map for the closest important point to jump to.
 */
function findNearestGlobalJump(importanceMap, visited, x, y) {
    let bestIdx = -1;
    let minDistSq = Infinity;

    for (let threshold = 2.5; threshold >= 1.0; threshold -= 1.0) {
        for (let idx = 0; idx < importanceMap.length; idx++) {
            if (importanceMap[idx] >= threshold && !visited[idx]) {
                let px = idx % V_WIDTH;
                let py = floor(idx / V_WIDTH);
                let dSq = (px - x) * (px - x) + (py - y) * (py - y);
                if (dSq < minDistSq) {
                    minDistSq = dSq;
                    bestIdx = idx;
                }
            }
        }
        if (bestIdx !== -1) {
            let jx = bestIdx % V_WIDTH;
            let jy = floor(bestIdx / V_WIDTH);
            markVisitedRadial(visited, jx, jy, 1.5);
            return { x: jx, y: jy };
        }
    }
    return null;
}

function markVisitedRadial(visited, x, y, r) {
    let rSq = r * r;
    let minY = max(0, floor(y - r));
    let maxY = min(V_HEIGHT - 1, floor(y + r));
    let minX = max(0, floor(x - r));
    let maxX = min(V_WIDTH - 1, floor(x + r));

    for (let j = minY; j <= maxY; j++) {
        for (let i = minX; i <= maxX; i++) {
            let distSq = (i - x) * (i - x) + (j - y) * (j - y);
            if (distSq < rSq) {
                visited[j * V_WIDTH + i] = 1;
            }
        }
    }
}

function windowResized() {
    resizeCanvas(windowWidth, windowHeight);
}