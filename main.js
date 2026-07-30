// One-Stroke Reality - Core Logic

// State
let state = {
    streaming: false,
    opencvReady: false,
    width: 0,
    height: 0,
    // Parameters
    threshold: 30,  // Canny Lower Threshold (Upper is 3x) - lower to pick up more details like wrinkles
    epsilon: 3.0,   // Approximation accuracy (Abstractness) - higher for sharper angles
    spike: 50,      // Miter Limit - higher for longer spikes
    thickness: 5.0, // Line width - thinner for the scribble look
    silhouette: 80, // Weight of silhouette vs internal detail - lower to emphasize internal lines
    jumpLinesVisible: true, // Whether to draw the connection lines
    segmentationReady: false,
    processingInProgress: false,
    uiVisible: true,
    portraitMode: false,
    recording: false
};

// Elements
const video = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d', { alpha: false }); // Optimize for no transparency
const toggleBtn = document.getElementById('toggleBtn');
const loader = document.getElementById('loader');

// OpenCV Variables
let src = null;
let gray = null;
let edges = null;
let constMat = null; // Reusable mat for approximation
let maskMat = null;  // For segmentation mask
let selfieSegmentation = null;
let lastResults = null;

// ----------------------------------------------------------------------------
// Initialization
// ----------------------------------------------------------------------------

function checkDependencies() {
    try {
        if (window.cv && typeof window.cv.Mat === 'function') {
            if (!state.opencvReady) {
                console.log("OpenCV Ready");
                state.opencvReady = true;
                loader.classList.add('hidden');
            }
        } else {
            // Check again after 200ms
            setTimeout(checkDependencies, 200);
        }
    } catch (e) {
        console.error("OpenCV Check Error: ", e);
        setTimeout(checkDependencies, 200);
    }
}
checkDependencies();

// エラー通知用：10秒経ってもロードが終わらない場合は画面にお知らせを出す
setTimeout(() => {
    if (!state.opencvReady) {
        const loaderText = document.querySelector('.loader-text');
        if (loaderText) {
            loaderText.innerHTML = "読み込みが完了しません。<br><br>1. インターネット接続を確認してください<br>2. F12キーを押して「Console」タブからエラーを確認してください<br>（file:// 環境ではなくローカルサーバーでの起動が必要な場合があります）";
            loaderText.style.textAlign = "center";
            loaderText.style.lineHeight = "1.6";
        }
    }
}, 10000);

toggleBtn.addEventListener('click', () => {
    if (state.streaming) {
        stopCamera();
    } else {
        startCamera();
    }
});

// Controls
document.getElementById('thresholdRange').addEventListener('input', (e) => {
    state.threshold = parseInt(e.target.value);
});
document.getElementById('epsilonRange').addEventListener('input', (e) => {
    state.epsilon = parseFloat(e.target.value);
});
document.getElementById('spikeRange').addEventListener('input', (e) => {
    state.spike = parseInt(e.target.value);
});
document.getElementById('thicknessRange').addEventListener('input', (e) => {
    state.thickness = parseFloat(e.target.value);
});
document.getElementById('silhouetteRange').addEventListener('input', (e) => {
    state.silhouette = parseInt(e.target.value);
});

const menuToggle = document.getElementById('menuToggle');
const modeMenu = document.getElementById('modeMenu');

menuToggle.addEventListener('click', (e) => {
    e.stopPropagation();
    modeMenu.classList.toggle('hidden');
});

document.addEventListener('click', (e) => {
    if (!menuToggle.contains(e.target) && !modeMenu.contains(e.target)) {
        modeMenu.classList.add('hidden');
    }
});

// Mode buttons and state
const previewModeBtn = document.getElementById('previewModeBtn');
const portraitModeBtn = document.getElementById('portraitModeBtn');
const recordingModeBtn = document.getElementById('recordingModeBtn');
const recordingIndicator = document.getElementById('recordingIndicator');
const recText = recordingIndicator.querySelector('.rec-text');

let mediaRecorder;
let fileStream;
let frameIntervalId;

// Preview Mode
previewModeBtn.addEventListener('click', () => {
    modeMenu.classList.add('hidden');
    if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen().catch(err => {
            console.error(`Error attempting to enable fullscreen: ${err.message}`);
        });
    }
    document.body.classList.add('preview-active');
});

document.addEventListener('fullscreenchange', () => {
    if (!document.fullscreenElement) {
        document.body.classList.remove('preview-active');
    }
});

// Portrait Mode
portraitModeBtn.addEventListener('click', () => {
    state.portraitMode = !state.portraitMode;
    if (state.portraitMode) {
        document.body.classList.add('portrait-active');
        portraitModeBtn.classList.add('active');
    } else {
        document.body.classList.remove('portrait-active');
        portraitModeBtn.classList.remove('active');
    }
    resizeCanvas();
});

// Recording Mode
recordingModeBtn.addEventListener('click', async () => {
    modeMenu.classList.add('hidden');
    if (!state.recording) {
        await startRecording();
    } else {
        await stopRecording();
    }
});

async function startRecording() {
    try {
        const fileHandle = await window.showSaveFilePicker({
            suggestedName: `timelapse_record_${new Date().getTime()}.webm`,
            types: [{
                description: 'WebM Video',
                accept: { 'video/webm': ['.webm'] },
            }],
        });

        fileStream = await fileHandle.createWritable();

        // 画面がフリーズする問題（captureStream(0)が元のキャンバスの描画を制限してしまう仕様）を回避するため、
        // 録画用の裏キャンバス（オフスクリーンキャンバス）を用意します。
        const recordingCanvas = document.createElement('canvas');
        recordingCanvas.width = canvas.width;
        recordingCanvas.height = canvas.height;
        const recCtx = recordingCanvas.getContext('2d', { alpha: false });

        const stream = recordingCanvas.captureStream(0);
        const videoTrack = stream.getVideoTracks()[0];

        const options = { mimeType: 'video/webm; codecs=vp9', videoBitsPerSecond: 100000 };
        try {
            mediaRecorder = new MediaRecorder(stream, options);
        } catch (e) {
            console.warn("VP9 not supported, falling back to default webm");
            try {
                mediaRecorder = new MediaRecorder(stream, { mimeType: 'video/webm', videoBitsPerSecond: 100000 });
            } catch(e2) {
                mediaRecorder = new MediaRecorder(stream, { videoBitsPerSecond: 100000 });
            }
        }

        mediaRecorder.ondataavailable = async (e) => {
            if (e.data && e.data.size > 0 && fileStream) {
                try {
                    await fileStream.write(e.data);
                } catch (err) {
                    console.error("Write error:", err);
                }
            }
        };

        const captureIntervalMS = 1000; // 1秒に1コマ (1fps)

        frameIntervalId = setInterval(() => {
            if (mediaRecorder.state === 'recording') {
                // 元のキャンバスの絵を裏キャンバスにコピーしてからキャプチャする
                recCtx.drawImage(canvas, 0, 0, recordingCanvas.width, recordingCanvas.height);
                videoTrack.requestFrame();
            }
        }, captureIntervalMS);

        mediaRecorder.start(60000); // 60秒ごとにディスクへフラッシュ
        
        state.recording = true;
        recordingModeBtn.textContent = "録画停止";
        recordingModeBtn.classList.add('active');
        
        // UI Indicator for 2 seconds
        recText.textContent = "REC START";
        recordingIndicator.classList.remove('hidden');
        setTimeout(() => {
            if (state.recording) { // Stop中に呼ばれないようにする安全策
                recordingIndicator.classList.add('hidden');
            }
        }, 2000);

    } catch (error) {
        console.error("録画の開始に失敗したか、キャンセルされました:", error);
    }
}

async function stopRecording() {
    return new Promise((resolve) => {
        if (frameIntervalId) {
            clearInterval(frameIntervalId);
            frameIntervalId = null;
        }

        if (mediaRecorder && state.recording) {
            mediaRecorder.onstop = async () => {
                if (fileStream) {
                    try {
                        await fileStream.close();
                        fileStream = null;
                    } catch (err) {
                        console.error("ファイルのクローズに失敗しました:", err);
                    }
                }
                resolve();
            };
            mediaRecorder.stop();
            state.recording = false;
            recordingModeBtn.textContent = "録画モード";
            recordingModeBtn.classList.remove('active');
            recordingIndicator.classList.add('hidden');
        } else {
            resolve();
        }
    });
}

// ----------------------------------------------------------------------------
// Camera
// ----------------------------------------------------------------------------

async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: 'user'
            },
            audio: false
        });
        video.srcObject = stream;
        await video.play();

        // Ensure video dimensions are ready before initializing arrays
        await new Promise((resolve) => {
            if (video.videoWidth > 0) {
                resolve();
            } else {
                video.onloadeddata = () => {
                    resolve();
                };
            }
        });

        state.width = video.videoWidth;
        state.height = video.videoHeight;

        // Resize canvas to match window, but process at video resolution
        resizeCanvas();
        window.addEventListener('resize', resizeCanvas);

        state.streaming = true;
        toggleBtn.textContent = "CAMERA STOP";

        if (src) {
            src.delete(); gray.delete(); edges.delete(); constMat.delete();
            if (maskMat) maskMat.delete();
            src = null;
        }

        initCV();
        initMediaPipe();
        requestAnimationFrame(processFrame);

    } catch (e) {
        console.error(e);
        alert("Camera Error: " + e.message);
    }
}

function stopCamera() {
    if (!state.streaming) return;
    state.streaming = false;
    toggleBtn.textContent = "CAMERA START";

    const stream = video.srcObject;
    if (stream) {
        stream.getTracks().forEach(t => t.stop());
    }
    video.srcObject = null;

    // Cleanup CV
    if (src) {
        src.delete(); gray.delete(); edges.delete(); constMat.delete();
        if (maskMat) maskMat.delete();
        src = null;
    }

    ctx.fillStyle = '#FFF';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function resizeCanvas() {
    if (state.portraitMode) {
        let w = window.innerWidth;
        let h = w * 16 / 9;
        if (h > window.innerHeight) {
            h = window.innerHeight;
            w = h * 9 / 16;
        }
        canvas.width = w;
        canvas.height = h;
    } else {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }
}

function initCV() {
    src = new cv.Mat(state.height, state.width, cv.CV_8UC4);
    gray = new cv.Mat(state.height, state.width, cv.CV_8UC1);
    edges = new cv.Mat(state.height, state.width, cv.CV_8UC1);
    maskMat = new cv.Mat(state.height, state.width, cv.CV_8UC1);
    constMat = new cv.Mat();
}

function initMediaPipe() {
    selfieSegmentation = new SelfieSegmentation({
        locateFile: (file) => {
            return `https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation/${file}`;
        }
    });

    selfieSegmentation.setOptions({
        modelSelection: 1, // 0 for general, 1 for landscape (faster)
    });

    selfieSegmentation.onResults((results) => {
        if (!state.streaming) {
            state.processingInProgress = false;
            return;
        }

        try {
            // Processing logic relocated here for real-time sync
            // Reduced internal processing width significantly to 240px to lighten the CPU load
            const procWidth = 240;
            const procHeight = Math.floor(procWidth * (state.height / state.width));


            if (tempCanvas.width !== procWidth || tempCanvas.height !== procHeight) {
                tempCanvas.width = procWidth;
                tempCanvas.height = procHeight;
                if (src) src.delete();
                if (gray) gray.delete();
                if (edges) edges.delete();
                if (maskMat) maskMat.delete();
                src = new cv.Mat(procHeight, procWidth, cv.CV_8UC4);
                gray = new cv.Mat(procHeight, procWidth, cv.CV_8UC1);
                edges = new cv.Mat(procHeight, procWidth, cv.CV_8UC1);
                maskMat = new cv.Mat(procHeight, procWidth, cv.CV_8UC1);
            }

            tempCtx.clearRect(0, 0, procWidth, procHeight);

            // A. Silhouette Mask
            tempCtx.drawImage(results.segmentationMask, 0, 0, procWidth, procHeight);
            let maskData = tempCtx.getImageData(0, 0, procWidth, procHeight);
            src.data.set(maskData.data);
            cv.cvtColor(src, maskMat, cv.COLOR_RGBA2GRAY);
            cv.threshold(maskMat, maskMat, 1, 255, cv.THRESH_BINARY);

            // B. Clipped Person Image
            tempCtx.globalCompositeOperation = 'source-in';
            tempCtx.drawImage(video, 0, 0, procWidth, procHeight);
            tempCtx.globalCompositeOperation = 'source-over';

            let imageData = tempCtx.getImageData(0, 0, procWidth, procHeight);
            src.data.set(imageData.data);
            cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
            cv.GaussianBlur(gray, gray, new cv.Size(5, 5), 0);
            cv.Canny(gray, edges, state.threshold, state.threshold * 3);

            // C. Find Contours
            let allPaths = [];

            // C1. Internal Contours (Canny)
            let internalContours = new cv.MatVector();
            let internalHierarchy = new cv.Mat();
            cv.findContours(edges, internalContours, internalHierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);

            for (let i = 0; i < internalContours.size(); ++i) {
                let cnt = internalContours.get(i);
                let minArea = 10 + (state.silhouette * 0.5);
                if (cv.contourArea(cnt) < minArea && cnt.rows < 10) continue;
                let approx = new cv.Mat();
                cv.approxPolyDP(cnt, approx, state.epsilon, true);
                if (approx.rows > 1) {
                    let points = [];
                    for (let j = 0; j < approx.rows; ++j) {
                        points.push({
                            x: (approx.data32S[j * 2] / procWidth) * state.width,
                            y: (approx.data32S[j * 2 + 1] / procHeight) * state.height
                        });
                    }
                    allPaths.push(points);
                }
                approx.delete();
            }
            internalContours.delete();
            internalHierarchy.delete();

            // C2. Silhouette Contour (Mask)
            let maskContours = new cv.MatVector();
            let maskHierarchy = new cv.Mat();
            cv.findContours(maskMat, maskContours, maskHierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

            for (let i = 0; i < maskContours.size(); ++i) {
                let cnt = maskContours.get(i);
                if (cv.contourArea(cnt) < 100) continue;
                let approx = new cv.Mat();
                cv.approxPolyDP(cnt, approx, state.epsilon * 0.5, true);
                if (approx.rows > 1) {
                    let points = [];
                    for (let j = 0; j < approx.rows; ++j) {
                        points.push({
                            x: (approx.data32S[j * 2] / procWidth) * state.width,
                            y: (approx.data32S[j * 2 + 1] / procHeight) * state.height
                        });
                    }
                    allPaths.push(points);
                }
                approx.delete();
            }
            maskContours.delete();
            maskHierarchy.delete();

            // D. One-Stroke Linker & Render
            // Combine paths with overlapping/scribble logic
            let finalPath = solveOneStrokeScribble(allPaths);
            render(finalPath);

        } catch (e) {
            console.error("Processing Error:", e);
        } finally {
            state.processingInProgress = false;
        }
    });
}

// ----------------------------------------------------------------------------
// Core Processing (One-Stroke Algorithm)
// ----------------------------------------------------------------------------

// Optimization: Reuse temp canvas
let tempCanvas = document.createElement('canvas');
let tempCtx = tempCanvas.getContext('2d');

async function processFrame() {
    if (!state.streaming || state.processingInProgress) {
        requestAnimationFrame(processFrame);
        return;
    }

    try {
        state.processingInProgress = true;
        // Trigger MediaPipe - result handled in onResults callback
        await selfieSegmentation.send({ image: video });
    } catch (e) {
        console.error("MediaPipe Error:", e);
        state.processingInProgress = false;
    }

    requestAnimationFrame(processFrame);
}

function solveOneStrokeScribble(paths) {
    if (paths.length === 0) return [];

    let result = [];
    let visited = new Array(paths.length).fill(false);
    let currentIdx = 0;
    visited[0] = true;

    result.push(...paths[0]);
    let currentPoint = paths[0][paths[0].length - 1];

    for (let count = 1; count < paths.length; count++) {
        let bestDist = Infinity;
        let bestIdx = -1;
        let bestReverse = false;

        // Instead of pure nearest neighbor, add a slight penalty for returning to the same area too quickly to encourage 'scribbling' across the face
        for (let i = 0; i < paths.length; i++) {
            if (visited[i]) continue;

            let pStart = paths[i][0];
            let pEnd = paths[i][paths[i].length - 1];

            let dStart = (currentPoint.x - pStart.x) ** 2 + (currentPoint.y - pStart.y) ** 2;
            let dEnd = (currentPoint.x - pEnd.x) ** 2 + (currentPoint.y - pEnd.y) ** 2;

            // Distances for finding the next stroke
            if (dStart < bestDist) {
                bestDist = dStart;
                bestIdx = i;
                bestReverse = false;
            }
            if (dEnd < bestDist) {
                bestDist = dEnd;
                bestIdx = i;
                bestReverse = true;
            }
        }

        if (bestIdx !== -1) {
            visited[bestIdx] = true;
            let nextPath = paths[bestIdx];

            // Introduce a subtle "bridge" point to make the connections more angular rather than straight lines cutting across empty space directly, though straight lines often look fine for the "glitch" aesthetic.
            if (bestReverse) {
                for (let k = nextPath.length - 1; k >= 0; k--) {
                    result.push(nextPath[k]);
                }
                currentPoint = nextPath[0];
            } else {
                for (let k = 0; k < nextPath.length; k++) {
                    result.push(nextPath[k]);
                }
                currentPoint = nextPath[nextPath.length - 1];
            }
        } else {
            break;
        }
    }

    return result;
}

function render(path) {
    // Fill white (Fixed Background)
    ctx.fillStyle = '#FFFFFF';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    if (path.length < 2) return;

    // Center and Scale the paths to focus more on the face/body
    // Usually the camera is centered, so we just scale up slightly
    const scaleFactor = 1.1;

    // Fix mobile aspect ratio: use uniform scaling (cover) instead of independent X/Y stretching
    const scaleX = canvas.width / state.width;
    const scaleY = canvas.height / state.height;
    const uniformScale = Math.max(scaleX, scaleY) * scaleFactor;

    const offsetX = (canvas.width - (state.width * uniformScale)) / 2;
    const offsetY = (canvas.height - (state.height * uniformScale)) / 2;

    ctx.beginPath();
    // Slightly off-black for a more natural ink look, but black is fine too.
    ctx.strokeStyle = '#111111';
    ctx.lineWidth = state.thickness;

    // Spikey aesthetic
    ctx.lineJoin = 'miter';
    ctx.miterLimit = state.spike;
    ctx.lineCap = 'butt'; // Keeps ends sharp

    ctx.moveTo(path[0].x * uniformScale + offsetX, path[0].y * uniformScale + offsetY);

    for (let i = 1; i < path.length; i++) {
        // Only draw long enough segments to avoid tiny dots looking like noise
        // This helps emphasize the angular strokes
        ctx.lineTo(path[i].x * uniformScale + offsetX, path[i].y * uniformScale + offsetY);
    }

    ctx.stroke();
}
