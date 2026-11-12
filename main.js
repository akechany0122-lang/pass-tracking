// ============================================================================
// SUBSTANTIALLY ── カメラ・パス化
//
// カメラ映像
//   ① 人物＋持ち物の切り抜き（MediaPipe: selfie_multiclass / Pose Landmarker）
//   ② マスク整形（しきい値・オープニング/クロージング・前フレームとのブレンド）
//   ③ 輪郭抽出（OpenCV findContours RETR_CCOMP ＝ 外側の輪郭＋内側の穴）
//   ④ 直線化（approxPolyDP ＝ 直線だけの多角形。角は丸めない）
//   ⑤ 小さな輪郭の除去
//   ⑥ 輪郭ごとに閉じたパスとして描画（一筆書きにしない）
// ============================================================================

const TV_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14';
const MODEL_URLS = {
    multiclass: 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite',
    pose_lite: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task',
    pose_full: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task',
    pose_heavy: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task',
};

// ----------------------------------------------------------------------------
// Parameters（調整パネルの値。localStorage に保存され、次回起動時に復元）
// ----------------------------------------------------------------------------

const DEFAULTS = {
    // ① 切り抜き
    segMode: 'hybrid',   // 'hybrid' | 'person' | 'merged'
    poseModel: 'full',   // 'lite' | 'full' | 'heavy'
    maxPeople: 4,
    procWidth: 320,      // 処理解像度（横幅 px）
    threshold: 0.5,      // 人物とみなす確率の境目
    // ② マスク整形
    openSize: 3,         // ノイズ除去（オープニング）の大きさ
    closeSize: 7,        // すき間埋め（クロージング）の大きさ
    smoothing: 0.55,     // 前フレームとのブレンド率（0 = ブレンドなし）
    // ③〜⑤ 輪郭・直線化
    epsilon: 3.0,        // 直線化の強さ（approxPolyDP の許容誤差, 320px 基準）
    minArea: 0.5,        // 描かない輪郭の大きさ（画面面積に対する %）
    showHoles: true,     // 内側の穴（中抜き）
    minHoleArea: 0.08,   // 穴の最小面積（%）
    // ⑥ 見た目
    drawMode: 'line',    // 'line' | 'fill' | 'both'
    lineWidth: 4,
    miterLimit: 40,
    lineColor: '#111111',
    bgColor: '#ffffff',
    mirror: true,
    zoom: 1.0,
};
const PARAMS_KEY = 'substantially.params.v1';

function loadParams() {
    const p = { ...DEFAULTS };
    try {
        const saved = JSON.parse(localStorage.getItem(PARAMS_KEY) || '{}');
        for (const k of Object.keys(DEFAULTS)) {
            if (k in saved && typeof saved[k] === typeof DEFAULTS[k]) p[k] = saved[k];
        }
    } catch (e) { }
    return p;
}
function saveParams() {
    try { localStorage.setItem(PARAMS_KEY, JSON.stringify(params)); } catch (e) { }
}

const params = loadParams();

// ----------------------------------------------------------------------------
// State
// ----------------------------------------------------------------------------

const state = {
    streaming: false,
    opencvReady: false,
    portraitMode: false,
    recording: false,
    frozen: false,
    lastVideoTime: -1,
    lastTs: 0,
    vw: 0, vh: 0,          // 最後に処理した映像のサイズ
    renderScale: 1,        // 最後の描画での 映像px → canvas px の倍率
};
const stats = { fps: 0, paths: 0 };

let wakeLock = null; // Wake Lock API for keeping screen awake

// Elements
const video = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d', { alpha: false });
const toggleBtn = document.getElementById('toggleBtn');
const loader = document.getElementById('loader');
const statusBadge = document.getElementById('statusBadge');

// 1フレーム分の結果: [{ outer: [[x,y],...], holes: [[[x,y],...], ...] }]（座標は 0〜1 に正規化）
let shapes = [];
// 人ごとの追跡（前フレームとのブレンド用）: { id, mask: Float32Array, anchor: {x,y}|null, missing }
let tracks = [];

// ----------------------------------------------------------------------------
// Status badge
// ----------------------------------------------------------------------------

let badgeTimer = null;
function showBadge(text, ms = 0) {
    if (!statusBadge) return;
    clearTimeout(badgeTimer);
    statusBadge.textContent = text;
    statusBadge.classList.remove('hidden');
    if (ms > 0) badgeTimer = setTimeout(updateBadge, ms);
}
function updateBadge() {
    if (!statusBadge) return;
    if (modelLoading) { statusBadge.textContent = 'モデル読み込み中…'; statusBadge.classList.remove('hidden'); }
    else if (state.frozen) { statusBadge.textContent = '■ FREEZE（Space で解除）'; statusBadge.classList.remove('hidden'); }
    else statusBadge.classList.add('hidden');
}

// ----------------------------------------------------------------------------
// OpenCV
// ----------------------------------------------------------------------------

function checkDependencies() {
    try {
        if (window.cv && typeof window.cv.Mat === 'function') {
            if (!state.opencvReady) {
                console.log("OpenCV Ready");
                state.opencvReady = true;
                loader.classList.add('hidden');
            }
            return;
        }
    } catch (e) {
        console.error("OpenCV Check Error: ", e);
    }
    setTimeout(checkDependencies, 200);
}
checkDependencies();

// 10秒経ってもロードが終わらない場合は画面にお知らせを出す
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

let binMat = null;          // 二値マスク（処理解像度）
const kernelCache = new Map();

function getKernel(size) {
    if (!kernelCache.has(size)) {
        kernelCache.set(size, cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(size, size)));
    }
    return kernelCache.get(size);
}

// ----------------------------------------------------------------------------
// ① MediaPipe models
// ----------------------------------------------------------------------------

let tv = null;              // tasks-vision module
let fileset = null;
let segmenter = null;       // selfie_multiclass（人物＋髪・服・アクセサリー）
let poseLandmarker = null;  // 人ごとのマスク
let poseModelLoaded = null;
let modelLoading = null;

function needsSegmenter() { return params.segMode !== 'person'; }
function needsPose() { return params.segMode !== 'merged'; }
function modelsReady() {
    if (modelLoading) return false;
    if (needsSegmenter() && !segmenter) return false;
    if (needsPose() && (!poseLandmarker || poseModelLoaded !== params.poseModel)) return false;
    return true;
}

async function createWithFallback(create, options, gpu = true) {
    if (!gpu) return await create({ ...options, baseOptions: { ...options.baseOptions, delegate: 'CPU' } });
    try {
        return await create({ ...options, baseOptions: { ...options.baseOptions, delegate: 'GPU' } });
    } catch (e) {
        console.warn('GPU で起動できなかったため CPU で再試行します', e);
        return await create({ ...options, baseOptions: { ...options.baseOptions, delegate: 'CPU' } });
    }
}

async function ensureModels() {
    while (modelLoading) {
        try { await modelLoading; } catch (e) { }
    }
    if (modelsReady()) return;

    modelLoading = (async () => {
        if (!tv) {
            tv = await import(`${TV_URL}/vision_bundle.mjs`);
            fileset = await tv.FilesetResolver.forVisionTasks(`${TV_URL}/wasm`);
        }
        if (needsSegmenter() && !segmenter) {
            segmenter = await createWithFallback(
                (o) => tv.ImageSegmenter.createFromOptions(fileset, o),
                {
                    baseOptions: { modelAssetPath: MODEL_URLS.multiclass },
                    runningMode: 'VIDEO',
                    outputConfidenceMasks: true,
                    outputCategoryMask: false,
                });
        }
        if (needsPose() && (!poseLandmarker || poseModelLoaded !== params.poseModel)) {
            if (poseLandmarker) { poseLandmarker.close(); poseLandmarker = null; }
            const model = params.poseModel;
            poseLandmarker = await createWithFallback(
                (o) => tv.PoseLandmarker.createFromOptions(fileset, o),
                {
                    baseOptions: { modelAssetPath: MODEL_URLS['pose_' + model] },
                    runningMode: 'VIDEO',
                    numPoses: params.maxPeople,
                    outputSegmentationMasks: true,
                    minPoseDetectionConfidence: 0.5,
                    minPosePresenceConfidence: 0.5,
                    minTrackingConfidence: 0.5,
                },
                // GPU だと人物マスクが全て 0 で返ってくる（tasks-vision 0.10.14 で確認）ため CPU で動かす
                false);
            poseModelLoaded = model;
        }
    })();
    updateBadge();
    try {
        await modelLoading;
    } catch (e) {
        console.error('モデルの読み込みに失敗しました', e);
        showBadge('モデルの読み込みに失敗しました（Console を確認）', 6000);
    } finally {
        modelLoading = null;
        updateBadge();
    }
}

// ----------------------------------------------------------------------------
// ①〜② Masks
// ----------------------------------------------------------------------------

const procCanvas = document.createElement('canvas');
const procCtx = procCanvas.getContext('2d');
let procW = 0, procH = 0;

// float マスクを (dw, dh) にバイリニアで拡大縮小（常にコピーを返す）
function resizeFloat(src, sw, sh, dw, dh) {
    if (sw === dw && sh === dh) return new Float32Array(src);
    const out = new Float32Array(dw * dh);
    const fx = sw / dw, fy = sh / dh;
    for (let y = 0; y < dh; y++) {
        const syf = Math.min(sh - 1, Math.max(0, (y + 0.5) * fy - 0.5));
        const y0 = Math.floor(syf), y1 = Math.min(sh - 1, y0 + 1), wy = syf - y0;
        for (let x = 0; x < dw; x++) {
            const sxf = Math.min(sw - 1, Math.max(0, (x + 0.5) * fx - 0.5));
            const x0 = Math.floor(sxf), x1 = Math.min(sw - 1, x0 + 1), wx = sxf - x0;
            const a = src[y0 * sw + x0], b = src[y0 * sw + x1];
            const c = src[y1 * sw + x0], d = src[y1 * sw + x1];
            out[y * dw + x] = (a + (b - a) * wx) * (1 - wy) + (c + (d - c) * wx) * wy;
        }
    }
    return out;
}

// 人の位置（肩と腰の中心）。フレーム間で同じ人を対応づけるのに使う
function anchorOf(landmarks) {
    if (!landmarks || !landmarks.length) return null;
    const ids = [11, 12, 23, 24].filter(i => landmarks[i]);
    const pts = ids.length ? ids.map(i => landmarks[i]) : landmarks;
    let x = 0, y = 0;
    for (const p of pts) { x += p.x; y += p.y; }
    return { x: x / pts.length, y: y / pts.length };
}

// しきい値以上の画素の連結成分ラベリング（4近傍）
function labelComponents(fg, thr, w, h) {
    const n = w * h;
    const map = new Int32Array(n);
    const stack = new Int32Array(n);
    let count = 0;
    for (let s = 0; s < n; s++) {
        if (map[s] || fg[s] < thr) continue;
        count++;
        let sp = 0;
        map[s] = count; stack[sp++] = s;
        while (sp) {
            const i = stack[--sp];
            const x = i % w;
            let j;
            if (x > 0 && !map[j = i - 1] && fg[j] >= thr) { map[j] = count; stack[sp++] = j; }
            if (x < w - 1 && !map[j = i + 1] && fg[j] >= thr) { map[j] = count; stack[sp++] = j; }
            if (i >= w && !map[j = i - w] && fg[j] >= thr) { map[j] = count; stack[sp++] = j; }
            if (i < n - w && !map[j = i + w] && fg[j] >= thr) { map[j] = count; stack[sp++] = j; }
        }
    }
    return { map, count };
}

// 統合マスク（人物＋持ち物）を、人ごとのマスクに振り分ける。
// - Pose のマスクがかかる画素はその人へ（重なる部分は両方の人に入る → 線が交差する）
// - 手に持った傘やバッグなど Pose が拾わない画素は、つながっている人へ
// - どの人ともつながっていない塊は、大きければ別の人として残し、小さければ（背景の誤検出として）捨てる
function composeHybrid(fg, people, w, h) {
    if (!fg) return people;
    if (!people.length) return [{ anchor: null, mask: fg }];
    const n = w * h, K = people.length, thr = params.threshold;
    const ASSIGN = 0.3;
    const { map, count } = labelComponents(fg, thr, w, h);
    const votes = new Float32Array((count + 1) * K);
    const out = people.map(p => ({ anchor: p.anchor, mask: new Float32Array(n) }));
    const assigned = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
        const f = fg[i], lab = map[i];
        for (let k = 0; k < K; k++) {
            const c = people[k].mask[i];
            if (c >= ASSIGN) {
                out[k].mask[i] = f > c ? f : c;
                assigned[i] = 1;
                if (lab) votes[lab * K + k]++;
            }
        }
    }
    const owner = new Int32Array(count + 1).fill(-1);
    for (let c = 1; c <= count; c++) {
        let best = 0;
        for (let k = 0; k < K; k++) {
            const v = votes[c * K + k];
            if (v > best) { best = v; owner[c] = k; }
        }
    }
    // Pose が検出できなかった人（遠い・横向き・重なりなど）の塊は、十分大きければ独立した人として残す
    const size = new Int32Array(count + 1), sx = new Float64Array(count + 1), sy = new Float64Array(count + 1);
    for (let i = 0; i < n; i++) {
        const c = map[i];
        if (c && owner[c] < 0) { size[c]++; sx[c] += i % w; sy[c] += (i / w) | 0; }
    }
    const minPx = params.minArea / 100 * n;
    const extra = new Map();
    for (let c = 1; c <= count; c++) {
        if (owner[c] < 0 && size[c] >= minPx) {
            const e = { anchor: { x: sx[c] / size[c] / w, y: sy[c] / size[c] / h }, mask: new Float32Array(n) };
            extra.set(c, e);
            out.push(e);
        }
    }
    for (let i = 0; i < n; i++) {
        if (assigned[i] || !map[i]) continue;
        const o = owner[map[i]];
        if (o >= 0) out[o].mask[i] = fg[i];
        else if (extra.has(map[i])) extra.get(map[i]).mask[i] = fg[i];
    }
    return out;
}

// 前フレームの人と対応づけて、マスクをブレンド（ちらつき防止）
let nextTrackId = 1;
function updateTracks(frameMasks) {
    const a = params.smoothing;
    const used = new Set();
    const next = [];
    for (const fm of frameMasks) {
        let best = null, bestD = Infinity;
        for (const t of tracks) {
            if (used.has(t) || t.mask.length !== fm.mask.length) continue;
            let d;
            if (!fm.anchor && !t.anchor) d = 0;
            else if (!fm.anchor || !t.anchor) d = Infinity;
            else d = Math.hypot(fm.anchor.x - t.anchor.x, fm.anchor.y - t.anchor.y);
            if (d < bestD) { bestD = d; best = t; }
        }
        if (best && bestD <= 0.25) {
            used.add(best);
            const m = best.mask, cur = fm.mask;
            for (let i = 0; i < m.length; i++) m[i] = m[i] * a + cur[i] * (1 - a);
            best.anchor = fm.anchor;
            best.missing = 0;
            next.push(best);
        } else {
            next.push({ id: nextTrackId++, mask: fm.mask, anchor: fm.anchor, missing: 0 });
        }
    }
    // 見失った人は数フレームかけてフェードアウト
    for (const t of tracks) {
        if (used.has(t)) continue;
        t.missing++;
        if (a > 0 && t.missing <= 6 && t.mask.length === procW * procH) {
            const m = t.mask;
            for (let i = 0; i < m.length; i++) m[i] *= a;
            next.push(t);
        }
    }
    tracks = next;
}

// ----------------------------------------------------------------------------
// ②〜⑤ Mask → closed straight polygons
// ----------------------------------------------------------------------------

function approxPoly(contour, eps, w, h) {
    const approx = new cv.Mat();
    cv.approxPolyDP(contour, approx, eps, true);
    const n = approx.rows;
    let pts = null;
    if (n >= 3) {
        const d = approx.data32S;
        pts = new Array(n);
        for (let j = 0; j < n; j++) pts[j] = [(d[j * 2] + 0.5) / w, (d[j * 2 + 1] + 0.5) / h];
    }
    approx.delete();
    return pts;
}

function oddSize(v) {
    const n = Math.round(v);
    return n <= 1 ? 0 : (n % 2 ? n : n + 1);
}

function maskToShapes(mask, w, h) {
    // ② 二値化 → オープニング（小さなノイズを消す）→ クロージング（小さな隙間を埋める）
    const thr = params.threshold;
    const bin = binMat.data;
    for (let i = 0; i < bin.length; i++) bin[i] = mask[i] >= thr ? 255 : 0;
    const s = w / 320;
    const ok = oddSize(params.openSize * s), ck = oddSize(params.closeSize * s);
    if (ok) cv.morphologyEx(binMat, binMat, cv.MORPH_OPEN, getKernel(ok));
    if (ck) cv.morphologyEx(binMat, binMat, cv.MORPH_CLOSE, getKernel(ck));

    // ③ 外側の輪郭と内側の穴（RETR_CCOMP: 2階層）
    const contours = new cv.MatVector();
    const hier = new cv.Mat();
    cv.findContours(binMat, contours, hier, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE);
    const H = hier.data32S;
    const total = w * h;
    const minA = params.minArea / 100 * total;
    const minHA = params.minHoleArea / 100 * total;
    const eps = params.epsilon * s;
    const out = [];

    for (let i = 0; i < contours.size(); i++) {
        if (H[i * 4 + 3] !== -1) continue; // 親がある＝穴。外側の輪郭から辿る
        const c = contours.get(i);
        const area = cv.contourArea(c);
        // ⑤ 小さすぎる輪郭は描かない
        const outer = area >= minA ? approxPoly(c, eps, w, h) : null; // ④ 直線化
        c.delete();
        if (!outer) continue;
        const holes = [];
        if (params.showHoles) {
            let ch = H[i * 4 + 2];
            while (ch !== -1) {
                const hc = contours.get(ch);
                if (cv.contourArea(hc) >= minHA) {
                    const p = approxPoly(hc, eps, w, h);
                    if (p) holes.push(p);
                }
                hc.delete();
                ch = H[ch * 4];
            }
        }
        out.push({ outer, holes });
    }
    contours.delete();
    hier.delete();
    return out;
}

// ----------------------------------------------------------------------------
// Frame processing
// ----------------------------------------------------------------------------

function processFrame() {
    const vw = video.videoWidth, vh = video.videoHeight;
    if (!vw || !vh) return;
    const w = Math.round(params.procWidth);
    const h = Math.max(2, Math.round(w * vh / vw));
    if (w !== procW || h !== procH) {
        procW = w; procH = h;
        procCanvas.width = w; procCanvas.height = h;
        if (binMat) binMat.delete();
        binMat = new cv.Mat(h, w, cv.CV_8UC1);
        tracks = [];
    }
    state.vw = vw; state.vh = vh;
    procCtx.drawImage(video, 0, 0, w, h);

    let ts = performance.now();
    if (ts <= state.lastTs) ts = state.lastTs + 1;
    state.lastTs = ts;

    // ① 切り抜き
    let fg = null;
    const people = [];
    if (needsSegmenter()) {
        segmenter.segmentForVideo(procCanvas, ts, (res) => {
            const masks = res.confidenceMasks;
            if (!masks || !masks.length) return;
            const bg = masks[0]; // 0 = background。1 - background = 人物（髪・肌・服・アクセサリー）
            fg = resizeFloat(bg.getAsFloat32Array(), bg.width, bg.height, w, h);
            for (let i = 0; i < fg.length; i++) fg[i] = 1 - fg[i];
        });
    }
    if (needsPose()) {
        poseLandmarker.detectForVideo(procCanvas, ts, (res) => {
            const masks = res.segmentationMasks || [];
            masks.forEach((m, i) => {
                people.push({
                    mask: resizeFloat(m.getAsFloat32Array(), m.width, m.height, w, h),
                    anchor: anchorOf(res.landmarks && res.landmarks[i]),
                });
            });
        });
    }

    let frameMasks;
    if (params.segMode === 'merged') frameMasks = fg ? [{ anchor: null, mask: fg }] : [];
    else if (params.segMode === 'person') frameMasks = people;
    else frameMasks = composeHybrid(fg, people, w, h);

    // ② 時間方向の平滑化 → ③〜⑤ 輪郭化
    updateTracks(frameMasks);
    const next = [];
    for (const t of tracks) next.push(...maskToShapes(t.mask, w, h));
    shapes = next;
    stats.paths = shapes.reduce((n, s) => n + 1 + s.holes.length, 0);
}

// Camera Selection Logic
const cameraSelect = document.getElementById('cameraSelect');
const CAMERA_LABEL_KEY = 'passpeople.cameraLabel';
const NIKON_LABEL_RE = /nikon|ニコン/i;

function loadSavedCameraLabel() {
    try { return localStorage.getItem(CAMERA_LABEL_KEY) || ''; } catch (e) { return ''; }
}
function saveCameraLabel(label) {
    try { localStorage.setItem(CAMERA_LABEL_KEY, label || ''); } catch (e) { }
}

async function getCameras() {
    if (!cameraSelect || !navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return [];
    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        // 権限未許可のブラウザでは deviceId が空のダミーが返るため除外する
        const videoDevices = devices.filter(d => d.kind === 'videoinput' && d.deviceId);

        // Preserve current selection if any (PCが変わってもラベル名で外部カメラを復元)
        const currentVal = cameraSelect.value;
        const savedLabel = loadSavedCameraLabel();
        cameraSelect.innerHTML = '<option value="">デフォルト</option>';

        let matched = false;
        videoDevices.forEach((device, index) => {
            const option = document.createElement('option');
            option.value = device.deviceId;
            option.text = device.label || `カメラ ${index + 1}`;
            if (!matched && device.deviceId === currentVal) {
                option.selected = true;
                matched = true;
            }
            cameraSelect.appendChild(option);
        });
        if (!matched && savedLabel) {
            const opt = Array.from(cameraSelect.options).find(o => o.value && o.text === savedLabel);
            if (opt) { opt.selected = true; matched = true; }
        }
        // 未選択ならニコンのカメラ（UVC接続 / Nikon Webcam Utility）を優先して選択
        const nikonOpt = Array.from(cameraSelect.options).find(o => o.value && NIKON_LABEL_RE.test(o.text));
        if (!matched && !state.streaming && nikonOpt) nikonOpt.selected = true;

        // ニコンが見つからない時だけ接続方法のヒントを表示
        const hint = document.getElementById('cameraHint');
        if (hint) hint.classList.toggle('hidden', !!nikonOpt);
        return videoDevices;
    } catch (e) {
        console.error('カメラ一覧の取得に失敗しました', e);
        return [];
    }
}

// Request permission to get labels / deviceIds, then list cameras
async function requestCameraPermissionAndList(force = false) {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) return;
    const list = await getCameras();
    // 既に許可済みでラベルが取れているなら追加の起動は不要
    if (!force && list.length > 0 && list.every(d => d.label)) return;
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        stream.getTracks().forEach(track => track.stop());
    } catch (e) {
        console.warn("Permission denied or error getting labels", e);
    }
    await getCameras();
}

// Initialize camera list
requestCameraPermissionAndList();

if (cameraSelect) {
    // プルダウンを開くたびに最新の接続状況を反映（後から挿した外部カメラ対応）
    cameraSelect.addEventListener('mousedown', () => { getCameras(); });
    cameraSelect.addEventListener('focus', () => { getCameras(); });

    // 再検索ボタン：権限を取り直して一覧を更新
    const rescanBtn = document.getElementById('cameraRescanBtn');
    if (rescanBtn) {
        rescanBtn.addEventListener('click', async () => {
            if (state.streaming) {
                await getCameras();
            } else {
                await requestCameraPermissionAndList(true);
            }
        });
    }

    // Restart camera if selection changes and streaming is active
    cameraSelect.addEventListener('change', () => {
        const opt = cameraSelect.options[cameraSelect.selectedIndex];
        saveCameraLabel(opt && opt.value ? opt.text : '');
        if (state.streaming) {
            stopCamera();
            startCamera();
        }
    });
}

// Also update list when devices change (e.g. plug in a camera)
if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) {
    navigator.mediaDevices.addEventListener('devicechange', getCameras);
}

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
            suggestedName: `record_${new Date().getTime()}.webm`,
            types: [{
                description: 'WebM Video',
                accept: { 'video/webm': ['.webm'] },
            }],
        });

        fileStream = await fileHandle.createWritable();

        // 30fpsでキャンバスから直接ストリームを取得（手動フレーム要求はバグの温床なため撤廃）
        const stream = canvas.captureStream(30);

        const options = { mimeType: 'video/webm; codecs=vp9', videoBitsPerSecond: 2500000 };
        try {
            mediaRecorder = new MediaRecorder(stream, options);
        } catch (e) {
            console.warn("VP9 not supported, falling back to default webm");
            mediaRecorder = new MediaRecorder(stream, { videoBitsPerSecond: 2500000 });
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

        // 1秒ごとにチャンクを書き出してデータ損失を防ぐ
        mediaRecorder.start(1000);
        
        state.recording = true;
        recordingModeBtn.textContent = "録画停止";
        recordingModeBtn.classList.add('active');
        
        // UI Indicator
        recText.textContent = "REC START";
        recordingIndicator.classList.remove('hidden');
        setTimeout(() => {
            if (state.recording) {
                recordingIndicator.classList.add('hidden');
            }
        }, 2000);

    } catch (error) {
        console.error("録画の開始に失敗したか、キャンセルされました:", error);
    }
}

async function stopRecording() {
    return new Promise((resolve) => {
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
// Wake Lock (Prevent screen from sleeping)
// ----------------------------------------------------------------------------

async function requestWakeLock() {
    try {
        if ('wakeLock' in navigator) {
            wakeLock = await navigator.wakeLock.request('screen');
            console.log('Screen Wake Lock acquired');
            wakeLock.addEventListener('release', () => {
                console.log('Screen Wake Lock released');
            });
        }
    } catch (err) {
        console.error(`Wake Lock error: ${err.name}, ${err.message}`);
    }
}

document.addEventListener('visibilitychange', async () => {
    if (wakeLock !== null && document.visibilityState === 'visible' && state.streaming) {
        await requestWakeLock();
    }
});

// ----------------------------------------------------------------------------
// Camera
// ----------------------------------------------------------------------------

async function startCamera() {
    if (!state.opencvReady) {
        alert('OpenCV の読み込みが完了していません。少し待ってから再度お試しください。');
        return;
    }
    let stream;
    let selectedDeviceId = cameraSelect ? cameraSelect.value : null;

    try {
        let videoConstraints;
        if (selectedDeviceId) {
            // 一眼カメラ等の仮想カメラの場合、解像度やfacingModeを指定するとエラーになりやすいため
            // deviceIdのみを厳格に指定し、解像度はデバイスのデフォルトに任せます。
            videoConstraints = { deviceId: { exact: selectedDeviceId } };
        } else {
            videoConstraints = {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: 'user'
            };
        }
        stream = await navigator.mediaDevices.getUserMedia({ video: videoConstraints, audio: false });
    } catch (e) {
        console.warn("1回目の起動に失敗しました。フォールバックを試みます。", e);
        try {
            let fallbackConstraints = selectedDeviceId ? { deviceId: { exact: selectedDeviceId } } : true;
            stream = await navigator.mediaDevices.getUserMedia({ video: fallbackConstraints, audio: false });
        } catch (fallbackError) {
            console.error(fallbackError);
            alert("カメラ接続エラー: " + fallbackError.name + "\n\n【よくある原因】\n1. 他のアプリ(Zoom, OBS, カメラメーカーの公式ソフト等)がカメラを使用中\n2. ブラウザでカメラのアクセスが許可されていない\n3. ケーブル接続やカメラ本体のモード設定(動画モード等)が正しくない");
            return;
        }
    }

    // 権限取得後に一覧を更新し、実際に使用中のカメラを選択状態にする
    try {
        await getCameras();
        const track = stream.getVideoTracks()[0];
        const activeId = track && track.getSettings ? track.getSettings().deviceId : null;
        if (activeId && cameraSelect && !cameraSelect.value) {
            const opt = Array.from(cameraSelect.options).find(o => o.value === activeId);
            if (opt) opt.selected = true;
        }
    } catch (e) {
        console.warn('カメラ一覧の更新に失敗しました', e);
    }

    try {
        video.srcObject = stream;
        await video.play();
        await new Promise((resolve) => {
            if (video.videoWidth > 0) resolve();
            else video.onloadeddata = () => resolve();
        });

        state.streaming = true;
        state.lastVideoTime = -1;
        tracks = [];
        toggleBtn.textContent = "CAMERA STOP";
        requestWakeLock();
        ensureModels();
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
    if (stream) stream.getTracks().forEach(t => t.stop());
    video.srcObject = null;

    if (wakeLock !== null) {
        wakeLock.release().then(() => { wakeLock = null; });
    }
    tracks = [];
    if (!state.frozen) shapes = [];
}

toggleBtn.addEventListener('click', () => {
    if (state.streaming) stopCamera();
    else startCamera();
});

// ----------------------------------------------------------------------------
// ⑥ Render — 輪郭ごとに閉じたパス（つなぎ線なし）
// ----------------------------------------------------------------------------

function resizeCanvas() {
    const dpr = window.devicePixelRatio || 1;
    let w = window.innerWidth, h = window.innerHeight;
    if (state.portraitMode) {
        h = w * 16 / 9;
        if (h > window.innerHeight) {
            h = window.innerHeight;
            w = h * 9 / 16;
        }
    }
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

function render() {
    const cw = canvas.width, ch = canvas.height;
    const dpr = window.devicePixelRatio || 1;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = params.bgColor;
    ctx.fillRect(0, 0, cw, ch);

    const vw = state.vw, vh = state.vh;
    if (!shapes.length || !vw) return;

    // 画面いっぱいに（cover）表示
    const sc = Math.max(cw / vw, ch / vh) * params.zoom;
    state.renderScale = sc / dpr;
    const ox = (cw - vw * sc) / 2, oy = (ch - vh * sc) / 2;
    const mirror = params.mirror;
    const trace = (pts) => {
        for (let j = 0; j < pts.length; j++) {
            const x = ((mirror ? 1 - pts[j][0] : pts[j][0]) * vw) * sc + ox;
            const y = pts[j][1] * vh * sc + oy;
            if (j === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.closePath();
    };

    ctx.lineJoin = 'miter';           // 角を尖らせる（トンガリ）
    ctx.miterLimit = params.miterLimit;
    ctx.lineCap = 'butt';
    ctx.lineWidth = params.lineWidth * dpr;

    if (params.drawMode !== 'line') {
        // 塗りつぶし（パネルと同じシルエット）。穴は evenodd で抜く
        ctx.fillStyle = params.lineColor;
        for (const s of shapes) {
            ctx.beginPath();
            trace(s.outer);
            for (const hole of s.holes) trace(hole);
            ctx.fill('evenodd');
        }
    }
    if (params.drawMode !== 'fill') {
        ctx.strokeStyle = params.drawMode === 'line' ? params.lineColor : params.bgColor;
        for (const s of shapes) {
            for (const pts of [s.outer, ...s.holes]) {
                ctx.beginPath();
                trace(pts);
                ctx.stroke();
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Main loop
// ----------------------------------------------------------------------------

let fpsFrames = 0, fpsStart = performance.now();
function loop() {
    requestAnimationFrame(loop);
    if (state.streaming && !state.frozen && state.opencvReady && video.readyState >= 2 &&
        video.currentTime !== state.lastVideoTime && modelsReady()) {
        state.lastVideoTime = video.currentTime;
        try {
            processFrame();
        } catch (e) {
            console.error('Processing Error:', e);
        }
        fpsFrames++;
    }
    const now = performance.now();
    if (now - fpsStart >= 1000) {
        stats.fps = Math.round(fpsFrames * 1000 / (now - fpsStart));
        fpsFrames = 0; fpsStart = now;
    }
    render();
}
requestAnimationFrame(loop);

// ----------------------------------------------------------------------------
// Freeze / SVG export / Fullscreen / UI toggle
// ----------------------------------------------------------------------------

const ui = { frozen: false };

function toggleFreeze(force) {
    state.frozen = typeof force === 'boolean' ? force : !state.frozen;
    ui.frozen = state.frozen;
    const btn = document.getElementById('freezeBtn');
    if (btn) btn.classList.toggle('active', state.frozen);
    if (!state.frozen) state.lastVideoTime = -1;
    updateBadge();
}

function timestamp() {
    const d = new Date();
    const p = (n) => String(n).padStart(2, '0');
    return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}_${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
}

// 表示中のパスを SVG で保存（Illustrator で開ける。外側＋穴は複合パスになる）
function exportSVG() {
    if (!shapes.length || !state.vw) {
        showBadge('保存できるパスがありません', 2000);
        return;
    }
    const vw = state.vw, vh = state.vh;
    const f = (n) => (Math.round(n * 100) / 100).toString();
    const pt = ([x, y]) => `${f((params.mirror ? 1 - x : x) * vw)} ${f(y * vh)}`;
    const sub = (pts) => 'M ' + pts.map(pt).join(' L ') + ' Z';
    const fill = params.drawMode === 'line' ? 'none' : params.lineColor;
    const stroke = params.drawMode === 'fill' ? 'none' : (params.drawMode === 'line' ? params.lineColor : params.bgColor);
    const sw = f(params.lineWidth / (state.renderScale || 1));
    const paths = shapes.map((s, i) =>
        `    <path id="path_${i + 1}" d="${[s.outer, ...s.holes].map(sub).join(' ')}"/>`).join('\n');
    const svg = `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${vw}" height="${vh}" viewBox="0 0 ${vw} ${vh}">
  <g id="SUBSTANTIALLY" fill="${fill}" fill-rule="evenodd" stroke="${stroke}" stroke-width="${sw}" stroke-linejoin="miter" stroke-miterlimit="${params.miterLimit}">
${paths}
  </g>
</svg>
`;
    const blob = new Blob([svg], { type: 'image/svg+xml' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `substantially_${timestamp()}.svg`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    showBadge(`SVG を保存しました（${stats.paths} パス）`, 2000);
}

function toggleFullscreen() {
    if (document.fullscreenElement) document.exitFullscreen();
    else document.documentElement.requestFullscreen().catch(err => console.error(err));
}

function toggleUI() {
    document.body.classList.toggle('ui-hidden');
}

const freezeBtn = document.getElementById('freezeBtn');
if (freezeBtn) freezeBtn.addEventListener('click', () => toggleFreeze());
const svgBtn = document.getElementById('svgBtn');
if (svgBtn) svgBtn.addEventListener('click', exportSVG);

document.addEventListener('keydown', (e) => {
    const tag = (e.target && e.target.tagName) || '';
    if (/INPUT|SELECT|TEXTAREA/.test(tag) || e.metaKey || e.ctrlKey || e.altKey) return;
    switch (e.key.toLowerCase()) {
        case 'h': toggleUI(); break;
        case ' ': e.preventDefault(); toggleFreeze(); break;
        case 's': exportSVG(); break;
        case 'f': toggleFullscreen(); break;
        case 'm': {
            const order = ['line', 'fill', 'both'];
            params.drawMode = order[(order.indexOf(params.drawMode) + 1) % order.length];
            saveParams();
            if (gui) gui.controllersRecursive().forEach(c => c.updateDisplay());
            break;
        }
        default: return;
    }
});

// ----------------------------------------------------------------------------
// 調整パネル（lil-gui）
// ----------------------------------------------------------------------------

let gui = null;

function onModelChange() {
    tracks = [];
    if (state.streaming) ensureModels();
}

function buildGUI() {
    if (!window.lil || !window.lil.GUI) return;
    gui = new lil.GUI({ title: '調整パネル（H で表示切替）' });

    const fSeg = gui.addFolder('① 切り抜き');
    fSeg.add(params, 'segMode', {
        '人物ごと＋持ち物（推奨）': 'hybrid',
        '人物ごと（Pose のみ）': 'person',
        '全体を1つ（Multiclass のみ）': 'merged',
    }).name('方式').onChange(onModelChange);
    fSeg.add(params, 'poseModel', { '軽量 lite': 'lite', '標準 full': 'full', '高精度 heavy': 'heavy' })
        .name('Pose モデル').onChange(onModelChange);
    fSeg.add(params, 'maxPeople', 1, 6, 1).name('最大人数').onFinishChange((v) => {
        if (poseLandmarker) poseLandmarker.setOptions({ numPoses: v });
    });
    fSeg.add(params, 'procWidth', 160, 640, 16).name('処理解像度');
    fSeg.add(params, 'threshold', 0.05, 0.95, 0.01).name('マスクのしきい値');

    const fMask = gui.addFolder('② マスク整形');
    fMask.add(params, 'openSize', 0, 15, 1).name('ノイズ除去');
    fMask.add(params, 'closeSize', 0, 25, 1).name('すき間埋め');
    fMask.add(params, 'smoothing', 0, 0.95, 0.01).name('平滑化（ちらつき防止）');

    const fPoly = gui.addFolder('③ 直線化・輪郭');
    fPoly.add(params, 'epsilon', 0.5, 15, 0.1).name('直線化の強さ');
    fPoly.add(params, 'minArea', 0, 5, 0.05).name('最小面積（%）');
    fPoly.add(params, 'showHoles').name('内側の穴（中抜き）');
    fPoly.add(params, 'minHoleArea', 0, 2, 0.01).name('穴の最小面積（%）');

    const fLook = gui.addFolder('④ 見た目');
    fLook.add(params, 'drawMode', { '線のみ': 'line', '塗りつぶし': 'fill', '塗り＋線': 'both' }).name('表示モード（M）');
    fLook.add(params, 'lineWidth', 0.5, 20, 0.5).name('線の太さ');
    fLook.add(params, 'miterLimit', 1, 100, 1).name('トンガリ（miterLimit）');
    fLook.addColor(params, 'lineColor').name('線の色');
    fLook.addColor(params, 'bgColor').name('背景色');
    fLook.add(params, 'mirror').name('左右反転');
    fLook.add(params, 'zoom', 0.5, 2, 0.01).name('拡大');

    const actions = {
        svg: exportSVG,
        fullscreen: toggleFullscreen,
        reset: () => {
            Object.assign(params, DEFAULTS);
            gui.controllersRecursive().forEach(c => c.updateDisplay());
            if (poseLandmarker) poseLandmarker.setOptions({ numPoses: params.maxPeople });
            saveParams();
            onModelChange();
        },
    };
    const fAct = gui.addFolder('操作');
    fAct.add(ui, 'frozen').name('静止（Space）').listen().onChange((v) => toggleFreeze(v));
    fAct.add(actions, 'svg').name('SVG 書き出し（S）');
    fAct.add(actions, 'fullscreen').name('フルスクリーン（F）');
    fAct.add(actions, 'reset').name('初期値に戻す');
    fAct.add(stats, 'fps').name('処理 fps').listen().disable();
    fAct.add(stats, 'paths').name('パス数').listen().disable();

    gui.onFinishChange(saveParams);
}
buildGUI();
