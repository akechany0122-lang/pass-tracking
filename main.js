// ============================================================================
// SUBSTANTIALLY ── カメラ・パス化
// 見た目の基準：reference/ の参考画像（Illustrator で作った人のパス）
//
// カメラ映像
//   ① 人物＋持ち物の切り抜き（MediaPipe: selfie_multiclass / Pose Landmarker）
//   ② マスク整形（しきい値・オープニング/クロージング・前フレームとのブレンド）
//   ③ 輪郭抽出（OpenCV findContours RETR_CCOMP ＝ 外側の輪郭＋内側の穴）
//   ④ 直線化（approxPolyDP ＝ 直線だけの多角形。角は丸めない）
//   ⑤ 小さな輪郭の除去
//   ⑤' 内側の線（髪・顔・手・持ち物の部位の境目＋人物の内側のエッジ）
//   ⑥ 輪郭・線ごとに独立したパスとして描画（一筆書きにしない）
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
    segMode: 'hybrid',     // 'hybrid' | 'person' | 'merged'
    poseModel: 'full',     // 'lite' | 'full' | 'heavy'
    maxPeople: 4,
    procWidth: 480,        // 処理解像度（横幅 px）
    threshold: 0.5,        // 人物とみなす確率の境目
    // ② マスク整形・ちらつき
    openSize: 2,           // ノイズ除去（オープニング）の大きさ
    closeSize: 7,          // すき間埋め（クロージング）。大きすぎると形がなめらかになりすぎる
    smoothing: 0.5,        // 前フレームとのブレンド率（0 = ブレンドなし）
    shapeFps: 12,          // パスの形を更新する頻度（一枚ずつ描いたような見え方）
    // ③ 外側の輪郭
    outerEpsPct: 1.8,      // 直線化の強さ（人物の高さに対する %）
    minArea: 0.3,          // 描かない輪郭の大きさ（画面面積に対する %）
    showHoles: true,       // 内側の穴（中抜き）
    minHoleArea: 0.05,     // 穴の最小面積（%）
    // ④ 内側の線
    showParts: true,       // 部位の境目（髪・顔・手・持ち物の多角形）
    showEdges: true,       // 人物の内側のエッジ（服の境目・靴など）
    innerMax: 22,          // 1人あたりの内側の線の上限
    faceMax: 6,            // 顔の中の線の上限
    innerEpsPct: 1.4,      // 内側の線の直線化（人物の高さに対する %）
    edgeMinLenPct: 16,     // これより短いエッジは捨てる（人物の高さに対する %）
    edgeLow: 60,           // Canny のしきい値
    edgeHigh: 160,
    edgeInsetPct: 2,       // 外側の輪郭からどれだけ内側でエッジを取るか（%）
    // ⑤ 見た目（参考画像20枚の実測：線の太さ＝画面の高さの約1.35%で一定、人物は高さの約83%）
    //   背景は白・線は黒で固定。1人＝1本の線。
    lineWidthPct: 1.8,     // 線の太さ（画面の高さに対する %）
    minSegPct: 1.5,        // これより短い線分はまとめて、長い直線だけにする（画面の高さに対する %）
    spikeAngle: 8,         // これより鈍い角でトンガリ（miter の針）が出る。8° ≒ miterLimit 14（11.5° = 10 が Illustrator 標準）
    spikeRate: 0.7,        // 外側の輪郭の出っ張った角のうち、トンガリを生やす割合
    spikeLen: 4,           // 生やすトンガリの長さ（線の太さの何倍か）
    spikeCorner: 30,       // この角度以上曲がる角にだけ生やす（°）
    mergeAngle: 9,         // これより小さい折れは1本の直線にまとめる（°）
    wedgeRate: 0.8,        // 内側の線のうち、くさび（細い三角＋先端のトンガリ）にする割合
    mirror: true,
    zoom: 1.0,
    // ⑥ 参考画像の重ね表示
    refShow: false,
    refIndex: 21,
    refOpacity: 0.35,
};
const PARAMS_KEY = 'substantially.params.v5';

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
    frozen: false,
    lastVideoTime: -1,
    lastTs: 0,
    vw: 0, vh: 0,          // 最後に処理した映像のサイズ
    renderScale: 1,        // 最後の描画での 映像px → canvas px の倍率
};
const stats = { fps: 0, paths: 0, parts: 0, ink: 0 };

let wakeLock = null; // Wake Lock API for keeping screen awake

// Elements
const video = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d', { alpha: false });
const startBtn = document.getElementById('startBtn');
const introNote = document.getElementById('introNote');
const statusBadge = document.getElementById('statusBadge');

// 1フレーム分の結果（座標は 0〜1 に正規化）:
// [{ h: 人物の高さ, path: [[x,y],...]（1人分を1本につないだ線）, parts: つないだ要素の数 }]
let figures = [];
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
    if (modelLoading) { statusBadge.textContent = 'LOADING MODEL'; statusBadge.classList.remove('hidden'); }
    else if (state.frozen) { statusBadge.textContent = 'FREEZE'; statusBadge.classList.remove('hidden'); }
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
                startBtn.disabled = false;
                startBtn.textContent = 'START';
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
        if (introNote) introNote.innerHTML = '読み込みが完了しません。インターネット接続を確認し、ローカルサーバーか公開URLで開いてください（file:// では動きません）。';
    }
}, 10000);

let binMat = null, tmpMat = null, edgeMat = null, grayMat = null, insetMat = null; // 処理解像度の作業用画像
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
const procCtx = procCanvas.getContext('2d', { willReadFrequently: true });
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
// ②〜⑤ Mask → 直線の多角形（外側の輪郭・中抜き・内側の線）
// ----------------------------------------------------------------------------

// selfie_multiclass のクラス
const CLS = { bg: 0, hair: 1, body: 2, face: 3, clothes: 4, others: 5 };
const PART_CLASSES = [CLS.hair, CLS.face, CLS.body, CLS.others]; // 髪・顔・手（肌）・持ち物

let classConf = null;   // クラスごとの確率（時間方向に平滑化）
let classMap = null;    // 画素ごとのクラス
let edgesReady = false;

function allocMats(w, h) {
    for (const m of [binMat, tmpMat, edgeMat, grayMat, insetMat]) if (m) m.delete();
    binMat = new cv.Mat(h, w, cv.CV_8UC1);
    insetMat = new cv.Mat(h, w, cv.CV_8UC1);
    tmpMat = new cv.Mat(h, w, cv.CV_8UC1);
    edgeMat = new cv.Mat(h, w, cv.CV_8UC1);
    grayMat = new cv.Mat(h, w, cv.CV_8UC1);
    classConf = null;
    classMap = null;
}

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

// 開いた線のダグラス・ポイカー法（直線化）
function simplifyOpen(pts, eps) {
    const n = pts.length;
    if (n <= 2) return pts.slice();
    const keep = new Uint8Array(n);
    keep[0] = keep[n - 1] = 1;
    const stack = [[0, n - 1]];
    const e2 = eps * eps;
    while (stack.length) {
        const [a, b] = stack.pop();
        const [ax, ay] = pts[a], [bx, by] = pts[b];
        const dx = bx - ax, dy = by - ay;
        const L2 = dx * dx + dy * dy || 1;
        let best = -1, bestD = e2;
        for (let i = a + 1; i < b; i++) {
            const t = ((pts[i][0] - ax) * dx + (pts[i][1] - ay) * dy) / L2;
            const px = ax + t * dx - pts[i][0], py = ay + t * dy - pts[i][1];
            const d = px * px + py * py;
            if (d > bestD) { bestD = d; best = i; }
        }
        if (best > 0) { keep[best] = 1; stack.push([a, best], [best, b]); }
    }
    return pts.filter((_, i) => keep[i]);
}

function contourPoints(c) {
    const d = c.data32S, n = c.rows, pts = new Array(n);
    for (let j = 0; j < n; j++) pts[j] = [d[j * 2], d[j * 2 + 1]];
    return pts;
}

function oddSize(v) {
    const n = Math.round(v);
    return n <= 1 ? 0 : (n % 2 ? n : n + 1);
}

// 髪・顔・肌（手）・持ち物の部位の境目を内側の線にする。
// 外側の輪郭と重なる部分は捨て、人物の内側を通る部分（髪と顔、袖と手、服と持ち物の境目など）だけ残す。
// 全体が内側にある形（顔の中の目・口、服の上のバッグなど）は閉じた多角形のまま残す。
function classParts(w, h, hPx) {
    const out = [];
    if (!classMap) return out;
    const eps = Math.max(0.5, hPx * params.innerEpsPct / 100);
    const minA = Math.pow(hPx * 0.015, 2); // 消すのは点状のノイズだけ（細長い形は残す）
    const minLen = hPx * params.edgeMinLenPct / 100;
    const tb = binMat.data, td = tmpMat.data, inside = insetMat.data;
    for (const c of PART_CLASSES) {
        let any = false;
        for (let i = 0; i < td.length; i++) {
            const on = tb[i] && classMap[i] === c;
            td[i] = on ? 255 : 0;
            if (on) any = true;
        }
        if (!any) continue;
        const contours = new cv.MatVector();
        const hier = new cv.Mat();
        cv.findContours(tmpMat, contours, hier, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE);
        const H = hier.data32S;
        for (let i = 0; i < contours.size(); i++) {
            const ct = contours.get(i);
            if (Math.abs(cv.contourArea(ct)) < minA) { ct.delete(); continue; }
            const isHole = H[i * 4 + 3] !== -1;
            const face = c === CLS.face && isHole; // 顔の中の穴＝目・眉・口など（顔の線の上限に数える）
            const all = contourPoints(ct);
            const n = all.length;
            const isIn = all.map(([x, y]) => inside[y * w + x] > 0);
            let start = isIn.indexOf(false);
            if (start < 0) {
                // 全体が人物の内側 → 閉じた多角形
                const pts = approxPoly(ct, eps, w, h);
                if (pts) out.push({ pts, closed: true, len: cv.arcLength(ct, true), face, part: true });
            } else {
                // 内側を通る区間だけを開いた線にする
                let run = [];
                const flush = () => {
                    if (run.length >= Math.max(3, minLen)) {
                        const pts = simplifyOpen(run, eps);
                        if (pts.length >= 2) out.push({
                            pts: pts.map(([x, y]) => [(x + 0.5) / w, (y + 0.5) / h]),
                            closed: false, len: run.length, face, part: true,
                        });
                    }
                    run = [];
                };
                for (let k = 1; k <= n; k++) {
                    const j = (start + k) % n;
                    if (isIn[j]) run.push(all[j]); else flush();
                }
                flush();
            }
            ct.delete();
        }
        contours.delete();
        hier.delete();
    }
    return out;
}

// 人物の内側（少し縮めたマスクの中）だけでエッジを取り、長い線だけを直線化する
function edgeLines(w, h, hPx) {
    const out = [];
    if (!edgesReady) return out;
    cv.bitwise_and(edgeMat, insetMat, tmpMat);
    const contours = new cv.MatVector();
    const hier = new cv.Mat();
    cv.findContours(tmpMat, contours, hier, cv.RETR_LIST, cv.CHAIN_APPROX_NONE);
    const minLen = hPx * params.edgeMinLenPct / 100;
    const eps = Math.max(0.5, hPx * params.innerEpsPct / 100);
    const d2 = (p, q) => (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2;
    for (let i = 0; i < contours.size(); i++) {
        const ct = contours.get(i);
        const perim = cv.arcLength(ct, true);
        if (perim / 2 < minLen) { ct.delete(); continue; }
        const area = Math.abs(cv.contourArea(ct));
        const closed = area > perim; // 本当に輪になっている線（往復ではない）
        let pts, len;
        if (closed) {
            const p = approxPoly(ct, eps, 1, 1);
            pts = p ? p.map(([x, y]) => [x - 0.5, y - 0.5]) : null;
            len = perim;
        } else {
            // 細い線の輪郭は行って戻る形なので、端から端までの片道だけにする
            const all = contourPoints(ct);
            let a = 0, b = 0, best = -1;
            for (let j = 0; j < all.length; j++) { const v = d2(all[j], all[0]); if (v > best) { best = v; a = j; } }
            best = -1;
            for (let j = 0; j < all.length; j++) { const v = d2(all[j], all[a]); if (v > best) { best = v; b = j; } }
            const seg = [];
            for (let j = a; ; j = (j + 1) % all.length) { seg.push(all[j]); if (j === b) break; }
            pts = simplifyOpen(seg, eps);
            len = perim / 2;
        }
        ct.delete();
        if (!pts || pts.length < 2) continue;
        let mx = 0, my = 0;
        for (const p of pts) { mx += p[0]; my += p[1]; }
        const mi = Math.round(my / pts.length) * w + Math.round(mx / pts.length);
        const face = !!(classMap && classMap[mi] === CLS.face);
        out.push({
            pts: pts.map(([x, y]) => [(x + 0.5) / w, (y + 0.5) / h]),
            closed, len, face, part: false,
        });
    }
    contours.delete();
    hier.delete();
    return out;
}

// 内側の線を、部位の形を優先し、長い順に上位 N 本だけ残す（顔は別の上限）
function selectInner(items) {
    items.sort((a, b) => (b.part - a.part) || (b.len - a.len));
    const out = [];
    let face = 0;
    for (const it of items) {
        if (out.length >= params.innerMax) break;
        if (it.face) {
            if (face >= params.faceMax) continue;
            face++;
        }
        out.push(it);
    }
    return out;
}

// 線ごとに決まった値（座標から作る）で、くさびにするかを決める。毎フレーム同じ線は同じ結果になりやすい
function wedgePick(p, rate) {
    const h = Math.sin(Math.round(p[0] * 60) * 12.9898 + Math.round(p[1] * 60) * 78.233) * 43758.5453;
    return h - Math.floor(h) < rate;
}

// 外側の輪郭の出っ張った角から、針のようなトンガリを生やす。
// 角の先へ伸ばした点まで行き、少し角度をつけて戻る → 細い三角が黒く潰れ、先端は miter で尖る
function addOutlineSpikes(pts, ax) {
    const n = pts.length;
    if (params.spikeRate <= 0 || n < 3) return pts;
    let area = 0;
    for (let i = 0; i < n; i++) {
        const p = pts[i], q = pts[(i + 1) % n];
        area += p[0] * ax * q[1] - q[0] * ax * p[1];
    }
    const sgn = Math.sign(area) || 1;
    const lw = params.lineWidthPct / 100;
    const th = Math.max(params.spikeAngle * 1.6, params.spikeAngle + 4) * Math.PI / 180;
    const minTurn = params.spikeCorner * Math.PI / 180;
    const out = [];
    for (let i = 0; i < n; i++) {
        const A = pts[(i - 1 + n) % n], V = pts[i], B = pts[(i + 1) % n];
        out.push(V);
        const ix = (V[0] - A[0]) * ax, iy = V[1] - A[1], ox = (B[0] - V[0]) * ax, oy = B[1] - V[1];
        const la = Math.hypot(ix, iy), lb = Math.hypot(ox, oy);
        if (la < lw * 1.5 || lb < lw * 1.5) continue;
        const turn = Math.atan2(ix * oy - iy * ox, ix * ox + iy * oy);
        if (Math.sign(turn) !== sgn || Math.abs(turn) < minTurn) continue; // 出っ張った角だけ
        if (!wedgePick(V, params.spikeRate)) continue;
        let dx = ix / la - ox / lb, dy = iy / la - oy / lb;
        const dl = Math.hypot(dx, dy) || 1;
        dx /= dl; dy /= dl;
        const h = Math.abs(Math.sin(V[0] * 91.7 + V[1] * 47.3) * 9301.7) % 1;
        const L = lw * params.spikeLen * (0.6 + 0.8 * h);
        const e = Math.min(lb * 0.45, Math.max(L * Math.tan(th), params.minSegPct / 100 * 1.3));
        out.push([V[0] + dx * L / ax, V[1] + dy * L], [V[0] + (ox / lb) * e / ax, V[1] + (oy / lb) * e]);
    }
    return out;
}

// 1人分の輪郭・穴・内側の線を、1本の連続した線につなぐ。
// いちばん長い外側の輪郭を土台にし、残りの要素を「いま描いている線のいちばん近い点」から枝分かれさせて差し込む。
// 枝は、近い点 → 要素（閉じた形は一周、開いた線は往復）→ 近い点に戻る、の順に通るので、
// つなぎ線は短く、人物の内側を通る。曲がり角が鋭いところでは miter のトンガリになる。
function linkFigure(contours, lines, ax) {
    const outers = contours.map(c => addOutlineSpikes(c.outer, ax)).sort((a, b) => b.length - a.length);
    if (!outers.length) return null;
    const base = outers[0];
    const path = base.concat([base[0].slice()]);
    const elems = [];
    for (const o of outers.slice(1)) elems.push({ pts: o, closed: true });
    for (const c of contours) for (const hole of c.holes) elems.push({ pts: hole, closed: true });
    for (const l of lines) elems.push({ pts: l.pts, closed: l.closed });
    const d2 = (p, q) => ((p[0] - q[0]) * ax) ** 2 + (p[1] - q[1]) ** 2;

    // 各要素について、線上のいちばん近い点を覚えておき、線が伸びたら新しい点とだけ比べる
    const check = (el, pts) => {
        const cand = el.closed ? el.pts.map((_, k) => k) : [0, el.pts.length - 1];
        for (const p of pts) {
            for (const vi of cand) {
                const v = d2(p, el.pts[vi]);
                if (v < el.d) { el.d = v; el.at = p; el.vi = vi; }
            }
        }
    };
    for (const el of elems) { el.d = Infinity; check(el, path); }

    while (elems.length) {
        let bi = 0;
        for (let k = 1; k < elems.length; k++) if (elems[k].d < elems[bi].d) bi = k;
        const el = elems.splice(bi, 1)[0];
        const i = path.indexOf(el.at);
        const n = el.pts.length;
        let seq;
        if (el.closed) {
            seq = [];
            for (let k = 0; k <= n; k++) seq.push(el.pts[(el.vi + k) % n].slice());
        } else {
            const fwd = el.vi === 0 ? el.pts : el.pts.slice().reverse();
            seq = fwd.map(q => q.slice());
            const tip = fwd[fwd.length - 1], st = fwd[0];
            const len = Math.hypot((tip[0] - st[0]) * ax, tip[1] - st[1]);
            if (len > 0 && wedgePick(st, params.wedgeRate)) {
                // くさび：先端で少し角度をつけて折り返す → 細い三角が黒く潰れ、先端にトンガリが出る。
                // 三角の幅は線の太さ程度に抑え、先端の近くだけに作る（大きな三角が体の外に出ないように）
                const ang = Math.max(params.spikeAngle * 1.6, params.spikeAngle + 4) * Math.PI / 180;
                const w = params.lineWidthPct / 100 * 1.2;          // 幅（画面の高さ基準）
                const k = Math.min(len * 0.8, w / Math.tan(ang));  // 先端から戻る距離
                const prev = fwd[fwd.length - 2];
                const pl = Math.hypot((tip[0] - prev[0]) * ax, tip[1] - prev[1]) || 1;
                const ux = (tip[0] - prev[0]) * ax / pl, uy = (tip[1] - prev[1]) / pl;
                const side = wedgePick(tip, 0.5) ? 1 : -1;
                const wk = k * Math.tan(ang);
                seq.push([tip[0] + (-ux * k - side * uy * wk) / ax, tip[1] - uy * k + side * ux * wk]);
                // 残りは元の線をなぞって根元へ
                const back = fwd.slice(0, -1).reverse().filter(q => Math.hypot((q[0] - tip[0]) * ax, q[1] - tip[1]) > k);
                seq.push(...back.map(q => q.slice()));
                if (!back.length || back[back.length - 1] !== st) seq.push(st.slice());
            } else {
                seq.push(...fwd.slice(0, -1).reverse().map(q => q.slice())); // 往復（同じ線をなぞって戻る）
            }
        }
        seq.push(el.at.slice());
        path.splice(i + 1, 0, ...seq);
        for (const other of elems) check(other, seq);
    }
    // 短すぎる線分をまとめ、ほぼ一直線に並ぶ点も消す → 長い直線だけで描く
    // （長さ 0 の線分も消える。角の向きが不安定になるのを防ぐ）
    const minSeg = params.minSegPct / 100;
    const len = (p, q) => Math.hypot((p[0] - q[0]) * ax, p[1] - q[1]);
    let out = [path[0]];
    for (let k = 1; k < path.length; k++) {
        if (len(path[k], out[out.length - 1]) >= minSeg || k === path.length - 1) out.push(path[k]);
    }
    const straight = [out[0]];
    for (let k = 1; k < out.length - 1; k++) {
        const a = straight[straight.length - 1], b = out[k], c = out[k + 1];
        const cross = ((b[0] - a[0]) * ax) * (c[1] - b[1]) - (b[1] - a[1]) * ((c[0] - b[0]) * ax);
        const dot = ((b[0] - a[0]) * ax) * ((c[0] - b[0]) * ax) + (b[1] - a[1]) * (c[1] - b[1]);
        if (dot > 0 && Math.abs(Math.atan2(cross, dot)) < params.mergeAngle * Math.PI / 180) continue; // ゆるい折れは直線にまとめる
        straight.push(b);
    }
    straight.push(out[out.length - 1]);
    return straight;
}

// 1人分のマスク → { h: 人物の高さ(0〜1), path: 1本の線, parts: 要素の数 }
function trackToFigure(mask, w, h) {
    // ② 二値化 → オープニング（ノイズを消す）→ クロージング（隙間を埋める）
    const thr = params.threshold;
    const bin = binMat.data;
    for (let i = 0; i < bin.length; i++) bin[i] = mask[i] >= thr ? 255 : 0;
    const s = w / 320;
    const ok = oddSize(params.openSize * s), ck = oddSize(params.closeSize * s);
    if (ok) cv.morphologyEx(binMat, binMat, cv.MORPH_OPEN, getKernel(ok));
    if (ck) cv.morphologyEx(binMat, binMat, cv.MORPH_CLOSE, getKernel(ck));

    // 人物の高さ（線の太さ・直線化の強さをこれに比例させる）
    let minY = h, maxY = -1;
    for (let y = 0; y < h; y++) {
        const row = y * w;
        for (let x = 0; x < w; x++) {
            if (bin[row + x]) { if (y < minY) minY = y; maxY = y; break; }
        }
    }
    if (maxY < 0) return null;
    const hPx = maxY - minY + 1;

    // ③ 外側の輪郭と内側の穴（RETR_CCOMP）→ ④ 直線化 → ⑤ 小さな輪郭を捨てる
    const contours = new cv.MatVector();
    const hier = new cv.Mat();
    cv.findContours(binMat, contours, hier, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE);
    const H = hier.data32S;
    const total = w * h;
    const minA = params.minArea / 100 * total;
    const minHA = params.minHoleArea / 100 * total;
    const eps = Math.max(0.5, hPx * params.outerEpsPct / 100);
    const fig = { h: hPx / h, contours: [], lines: [] };

    for (let i = 0; i < contours.size(); i++) {
        if (H[i * 4 + 3] !== -1) continue;
        const c = contours.get(i);
        const outer = Math.abs(cv.contourArea(c)) >= minA ? approxPoly(c, eps, w, h) : null;
        c.delete();
        if (!outer) continue;
        const holes = [];
        if (params.showHoles) {
            let ch = H[i * 4 + 2];
            while (ch !== -1) {
                const hc = contours.get(ch);
                if (Math.abs(cv.contourArea(hc)) >= minHA) {
                    const p = approxPoly(hc, eps, w, h);
                    if (p) holes.push(p);
                }
                hc.delete();
                ch = H[ch * 4];
            }
        }
        fig.contours.push({ outer, holes });
    }
    contours.delete();
    hier.delete();
    if (!fig.contours.length) return null;

    // 内側の線（部位の境目＋人物の内側のエッジ）。外側の輪郭から少し内側の範囲だけを使う
    cv.erode(binMat, insetMat, getKernel(oddSize(Math.max(3, hPx * params.edgeInsetPct / 100))));
    const inner = [];
    if (params.showParts) inner.push(...classParts(w, h, hPx));
    if (params.showEdges) inner.push(...edgeLines(w, h, hPx));
    fig.lines = selectInner(inner).map(({ pts, closed }) => ({ pts, closed }));
    const path = linkFigure(fig.contours, fig.lines, state.vw && state.vh ? state.vw / state.vh : 16 / 9);
    if (!path) return null;
    return { h: fig.h, path, parts: fig.contours.reduce((m, c) => m + 1 + c.holes.length, 0) + fig.lines.length };
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
        allocMats(w, h);
        tracks = [];
    }
    state.vw = vw; state.vh = vh;
    procCtx.drawImage(video, 0, 0, w, h);

    let ts = performance.now();
    if (ts <= state.lastTs) ts = state.lastTs + 1;
    state.lastTs = ts;
    const a = params.smoothing;

    // ① 切り抜き
    let fg = null;
    const people = [];
    if (needsSegmenter()) {
        segmenter.segmentForVideo(procCanvas, ts, (res) => {
            const masks = res.confidenceMasks;
            if (!masks || !masks.length) return;
            const arrs = masks.map(m => resizeFloat(m.getAsFloat32Array(), m.width, m.height, w, h));
            // 0 = background。1 - background = 人物（髪・肌・服・アクセサリー）
            fg = new Float32Array(arrs[0].length);
            for (let i = 0; i < fg.length; i++) fg[i] = 1 - arrs[0][i];
            if (!classConf || classConf.length !== arrs.length) classConf = arrs;
            else {
                for (let k = 0; k < arrs.length; k++) {
                    const cc = classConf[k], cur = arrs[k];
                    for (let i = 0; i < cc.length; i++) cc[i] = cc[i] * a + cur[i] * (1 - a);
                }
            }
        });
    } else {
        classConf = null;
    }
    // 画素ごとのクラス
    if (classConf && classConf.length >= 6) {
        const n = w * h;
        if (!classMap || classMap.length !== n) classMap = new Uint8Array(n);
        for (let i = 0; i < n; i++) {
            let best = 0, bv = classConf[0][i];
            for (let k = 1; k < classConf.length; k++) {
                const v = classConf[k][i];
                if (v > bv) { bv = v; best = k; }
            }
            classMap[i] = best;
        }
    } else {
        classMap = null;
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

    // 内側の線用のエッジ（映像全体で1回。人ごとにマスクの内側だけ使う）
    edgesReady = false;
    if (params.showEdges) {
        const rgba = cv.matFromImageData(procCtx.getImageData(0, 0, w, h));
        cv.cvtColor(rgba, grayMat, cv.COLOR_RGBA2GRAY);
        rgba.delete();
        cv.GaussianBlur(grayMat, grayMat, new cv.Size(3, 3), 0);
        cv.Canny(grayMat, edgeMat, params.edgeLow, params.edgeHigh);
        edgesReady = true;
    }

    let frameMasks;
    if (params.segMode === 'merged') frameMasks = fg ? [{ anchor: null, mask: fg }] : [];
    else if (params.segMode === 'person') frameMasks = people;
    else frameMasks = composeHybrid(fg, people, w, h);

    // ② 時間方向の平滑化 → ③〜⑤ 輪郭化
    updateTracks(frameMasks);
    const next = [];
    for (const t of tracks) {
        const f = trackToFigure(t.mask, w, h);
        if (f) next.push(f);
    }
    figures = next;
    stats.paths = figures.length;
    stats.parts = figures.reduce((n, f) => n + f.parts, 0);
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
        document.body.classList.add('running');
        pokeDock();
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
    document.body.classList.remove('running');

    const stream = video.srcObject;
    if (stream) stream.getTracks().forEach(t => t.stop());
    video.srcObject = null;

    if (wakeLock !== null) {
        wakeLock.release().then(() => { wakeLock = null; });
    }
    tracks = [];
    if (!state.frozen) figures = [];
}

startBtn.addEventListener('click', () => { if (!state.streaming) startCamera(); });
document.getElementById('stopBtn').addEventListener('click', () => {
    toggleFreeze(false);
    stopCamera();
});

// ----------------------------------------------------------------------------
// ⑥ Render — 輪郭・線ごとに独立したパス（つなぎ線なし）
// ----------------------------------------------------------------------------

function resizeCanvas() {
    const dpr = window.devicePixelRatio || 1;
    const w = window.innerWidth, h = window.innerHeight;
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

const INK = '#000000';    // 線は黒で固定
const PAPER = '#ffffff';  // 背景は白で固定

// トンガリが出る角度 → miterLimit（miter の長さ / 線の太さ = 1 / sin(角度/2)）
function miterLimitFor(angleDeg) {
    return 1 / Math.sin(Math.max(0.5, angleDeg) * Math.PI / 360);
}

// 参考画像（reference/ref_01〜20.png：1回目の資料、ref_21〜40.webp：2回目の資料）
const refImages = new Map();
function getRefImage(i) {
    const key = String(i).padStart(2, '0');
    if (!refImages.has(key)) {
        const img = new Image();
        img.src = `reference/ref_${key}.${i > 20 ? 'webp' : 'png'}`;
        refImages.set(key, img);
    }
    return refImages.get(key);
}

// インク量（黒い部分の割合）を測る
const inkCanvas = document.createElement('canvas');
inkCanvas.width = 160; inkCanvas.height = 90;
const inkCtx = inkCanvas.getContext('2d', { willReadFrequently: true });
let lastInk = 0;
function measureInk() {
    inkCtx.drawImage(canvas, 0, 0, inkCanvas.width, inkCanvas.height);
    const d = inkCtx.getImageData(0, 0, inkCanvas.width, inkCanvas.height).data;
    let ink = 0;
    for (let i = 0; i < d.length; i += 4) {
        if (0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2] < 128) ink++;
    }
    stats.ink = Math.round(ink / (d.length / 4) * 1000) / 10;
}

function render() {
    const cw = canvas.width, ch = canvas.height;
    const dpr = window.devicePixelRatio || 1;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.globalAlpha = 1;
    ctx.fillStyle = PAPER;
    ctx.fillRect(0, 0, cw, ch);

    const vw = state.vw, vh = state.vh;
    if (figures.length && vw) {
        // 画面いっぱいに（cover）表示
        const sc = Math.max(cw / vw, ch / vh) * params.zoom;
        state.renderScale = sc / dpr;
        const ox = (cw - vw * sc) / 2, oy = (ch - vh * sc) / 2;
        const mirror = params.mirror;

        ctx.strokeStyle = INK;
        ctx.lineJoin = 'miter';           // 鋭角は黒いトゲ（針）になる
        ctx.miterLimit = miterLimitFor(params.spikeAngle);
        ctx.lineCap = 'butt';
        // 線の太さは画面の高さに対して一定（参考画像と同じ。人が小さくても線は細くならない）
        ctx.lineWidth = Math.max(1, ch * params.lineWidthPct / 100);

        // 1人＝1本の線。beginPath から stroke まで、途中で途切れない
        for (const f of figures) {
            const pts = f.path;
            ctx.beginPath();
            for (let j = 0; j < pts.length; j++) {
                const x = ((mirror ? 1 - pts[j][0] : pts[j][0]) * vw) * sc + ox;
                const y = pts[j][1] * vh * sc + oy;
                if (j === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
            }
            ctx.closePath();
            ctx.stroke();
        }
    }

    const now = performance.now();
    if (now - lastInk >= 1000) { lastInk = now; measureInk(); }

    // 参考画像を半透明で重ねる（見比べ用）
    if (params.refShow) {
        const img = getRefImage(params.refIndex);
        if (img.complete && img.naturalWidth) {
            const s = Math.min(cw / img.naturalWidth, ch / img.naturalHeight);
            const iw = img.naturalWidth * s, ih = img.naturalHeight * s;
            ctx.globalAlpha = params.refOpacity;
            ctx.drawImage(img, (cw - iw) / 2, (ch - ih) / 2, iw, ih);
            ctx.globalAlpha = 1;
        }
    }
}

// ----------------------------------------------------------------------------
// Main loop（映像は毎フレーム、パスの形は「形の更新 fps」で間引いて更新）
// ----------------------------------------------------------------------------

let fpsFrames = 0, fpsStart = performance.now(), lastShape = 0;
function loop() {
    requestAnimationFrame(loop);
    const now = performance.now();
    if (state.streaming && !state.frozen && state.opencvReady && video.readyState >= 2 &&
        video.currentTime !== state.lastVideoTime && modelsReady() &&
        now - lastShape >= 1000 / params.shapeFps - 4) {
        state.lastVideoTime = video.currentTime;
        lastShape = now;
        try {
            processFrame();
        } catch (e) {
            console.error('Processing Error:', e);
        }
        fpsFrames++;
    }
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

// 表示中のパスを SVG で保存（Illustrator で開ける。1人＝1本の閉じたパス）
function exportSVG() {
    if (!figures.length || !state.vw) {
        showBadge('保存できるパスがありません', 2000);
        return;
    }
    const vw = state.vw, vh = state.vh;
    const f = (n) => (Math.round(n * 100) / 100).toString();
    const pt = ([x, y]) => `${f((params.mirror ? 1 - x : x) * vw)} ${f(y * vh)}`;
    // 画面で見えている太さを映像の座標に換算
    const sw = f((canvas.height * params.lineWidthPct / 100) / (state.renderScale * (window.devicePixelRatio || 1)));
    const paths = figures.map((fig, i) =>
        `    <path id="person_${i + 1}" d="M ${fig.path.map(pt).join(' L ')} Z"/>`).join('\n');
    const svg = `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${vw}" height="${vh}" viewBox="0 0 ${vw} ${vh}">
  <g id="SUBSTANTIALLY" fill="none" stroke="${INK}" stroke-width="${sw}" stroke-linejoin="miter" stroke-miterlimit="${f(miterLimitFor(params.spikeAngle))}" stroke-linecap="butt">
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

// H：展示モード（操作バーも調整パネルも出さない、カーソルも消す）
function toggleUI() {
    document.body.classList.toggle('ui-hidden');
    if (document.body.classList.contains('ui-hidden') && gui) gui.hide();
}

function toggleTune(force) {
    if (!gui) return;
    const show = typeof force === 'boolean' ? force : gui._hidden;
    gui.show(show);
    document.getElementById('tuneBtn').classList.toggle('active', show);
    if (show) document.body.classList.remove('ui-hidden');
}

function refreshGUI() {
    if (gui) gui.controllersRecursive().forEach(c => c.updateDisplay());
}

// 操作バー：マウスを動かすと現れ、しばらく動かさないと消える
const dock = document.getElementById('dock');
let dockTimer = null;
function pokeDock() {
    document.body.classList.add('dock-on');
    clearTimeout(dockTimer);
    dockTimer = setTimeout(() => {
        if (dock.matches(':hover')) return pokeDock();
        document.body.classList.remove('dock-on');
    }, 2500);
}
document.addEventListener('mousemove', pokeDock);
document.addEventListener('touchstart', pokeDock, { passive: true });

document.getElementById('freezeBtn').addEventListener('click', () => toggleFreeze());
document.getElementById('svgBtn').addEventListener('click', exportSVG);
document.getElementById('fullBtn').addEventListener('click', toggleFullscreen);
document.getElementById('tuneBtn').addEventListener('click', () => toggleTune());

document.addEventListener('keydown', (e) => {
    const tag = (e.target && e.target.tagName) || '';
    if (/INPUT|SELECT|TEXTAREA/.test(tag) || e.metaKey || e.ctrlKey || e.altKey) return;
    switch (e.key.toLowerCase()) {
        case 'h': toggleUI(); break;
        case 'g': toggleTune(); break;
        case ' ': e.preventDefault(); toggleFreeze(); break;
        case 's': exportSVG(); break;
        case 'f': toggleFullscreen(); break;
        case 'r': params.refShow = !params.refShow; saveParams(); refreshGUI(); break;
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
    gui = new lil.GUI({ title: 'TUNE' });
    gui.hide();

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

    const fMask = gui.addFolder('② マスク整形・ちらつき');
    fMask.add(params, 'openSize', 0, 15, 1).name('ノイズ除去');
    fMask.add(params, 'closeSize', 0, 25, 1).name('すき間埋め');
    fMask.add(params, 'smoothing', 0, 0.95, 0.01).name('平滑化');
    fMask.add(params, 'shapeFps', 2, 30, 1).name('形の更新（fps）');

    const fOuter = gui.addFolder('③ 外側の輪郭');
    fOuter.add(params, 'outerEpsPct', 0.1, 3, 0.05).name('直線化（身長の%）');
    fOuter.add(params, 'minArea', 0, 5, 0.05).name('最小面積（画面の%）');
    fOuter.add(params, 'showHoles').name('内側の穴（中抜き）');
    fOuter.add(params, 'minHoleArea', 0, 2, 0.01).name('穴の最小面積（%）');

    const fInner = gui.addFolder('④ 内側の線');
    fInner.add(params, 'showParts').name('部位の境目（髪・顔・手・持ち物）');
    fInner.add(params, 'showEdges').name('内側のエッジ（服の境目など）');
    fInner.add(params, 'innerMax', 0, 150, 1).name('線の本数（上限）');
    fInner.add(params, 'faceMax', 0, 20, 1).name('顔の線（上限）');
    fInner.add(params, 'innerEpsPct', 0.1, 2, 0.05).name('直線化（身長の%）');
    fInner.add(params, 'edgeMinLenPct', 0.5, 20, 0.5).name('最短の線（身長の%）');
    fInner.add(params, 'edgeLow', 5, 200, 1).name('エッジ感度（下）');
    fInner.add(params, 'edgeHigh', 10, 400, 1).name('エッジ感度（上）');
    fInner.add(params, 'edgeInsetPct', 0, 8, 0.5).name('輪郭からの内側幅（%）');

    const fLook = gui.addFolder('⑤ 見た目');
    fLook.add(params, 'lineWidthPct', 0.3, 6, 0.05).name('線の太さ（画面の高さの%）');
    fLook.add(params, 'minSegPct', 0, 4, 0.05).name('最短の線分（%）');
    fLook.add(params, 'spikeAngle', 2, 60, 0.5).name('トンガリが出る角度（°）');
    fLook.add(params, 'spikeRate', 0, 1, 0.05).name('輪郭のトンガリの割合');
    fLook.add(params, 'spikeLen', 0, 12, 0.5).name('トンガリの長さ（×太さ）');
    fLook.add(params, 'spikeCorner', 0, 120, 1).name('トンガリを生やす角（°）');
    fLook.add(params, 'wedgeRate', 0, 1, 0.05).name('くさびの割合');
    fLook.add(params, 'mergeAngle', 0, 30, 0.5).name('直線にまとめる折れ（°）');
    fLook.add(params, 'mirror').name('左右反転');
    fLook.add(params, 'zoom', 0.5, 2, 0.01).name('拡大');

    const fRef = gui.addFolder('⑥ 参考画像と見比べる');
    fRef.add(params, 'refShow').name('重ねて表示（R）');
    fRef.add(params, 'refIndex', 1, 40, 1).name('参考画像の番号');
    fRef.add(params, 'refOpacity', 0, 1, 0.01).name('不透明度');
    fRef.close();

    const actions = {
        svg: exportSVG,
        fullscreen: toggleFullscreen,
        reset: () => {
            Object.assign(params, DEFAULTS);
            refreshGUI();
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
    fAct.add(stats, 'fps').name('形の更新 fps').listen().disable();
    fAct.add(stats, 'paths').name('人数（＝線の本数）').listen().disable();
    fAct.add(stats, 'parts').name('つないだ要素の数').listen().disable();
    fAct.add(stats, 'ink').name('インク量（%）').listen().disable();

    gui.onFinishChange(saveParams);
}
buildGUI();
