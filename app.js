import {
  FilesetResolver,
  HandLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const statusEl = document.getElementById("status");
const hudEl = document.getElementById("hud");
const stageEl = document.getElementById("stage");
const videoEl = document.getElementById("video");
const overlayEl = document.getElementById("overlay");
const ctx = overlayEl.getContext("2d");
const CLONE_COUNT = 10;
const KAGE_EFFECT_MS = 3200;
const CHIDORI_EFFECT_MS = 2500;
const RASENGAN_EFFECT_MS = 2700;
const JUTSU_COOLDOWN_MS = 4300;

const CLONE_OFFSETS = [
  { x: -0.45, y: -0.03, scale: 1.0, tilt: -0.02 },
  { x: 0.45, y: -0.03, scale: 1.0, tilt: 0.02 },
  { x: -0.3, y: -0.17, scale: 0.98, tilt: -0.012 },
  { x: 0.3, y: -0.17, scale: 0.98, tilt: 0.012 },
  { x: -0.2, y: 0.05, scale: 1.02, tilt: -0.008 },
  { x: 0.2, y: 0.05, scale: 1.02, tilt: 0.008 },
  { x: -0.39, y: 0.2, scale: 1.0, tilt: -0.018 },
  { x: 0.39, y: 0.2, scale: 1.0, tilt: 0.018 },
  { x: -0.1, y: 0.24, scale: 1.01, tilt: -0.01 },
  { x: 0.1, y: 0.24, scale: 1.01, tilt: 0.01 },
];

const HAND_CONNECTIONS = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [5, 9],
  [9, 10],
  [10, 11],
  [11, 12],
  [9, 13],
  [13, 14],
  [14, 15],
  [15, 16],
  [13, 17],
  [17, 18],
  [18, 19],
  [19, 20],
  [0, 17],
];

let handLandmarker = null;
let stream = null;
let rafId = null;
let lastVideoTime = -1;
let viewWidth = 1;
let viewHeight = 1;
let effectUntil = 0;
let cooldownUntil = 0;
let audioCtx = null;
let noiseBuffer = null;
let latestHands = [];
let selfieSegmentation = null;
let segmentationBusy = false;
let lastSegmentationTime = -1;
let hasPersonCutout = false;
let activeJutsu = null;
let activeChidoriEmitter = null;
let activeRasenganData = null;
const personCutoutCanvas = document.createElement("canvas");
const personCutoutCtx = personCutoutCanvas.getContext("2d");

function setStatus(message) {
  statusEl.textContent = message;
}

function setHud(message) {
  hudEl.textContent = message;
}

function resizeCanvas() {
  const rect = stageEl.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;

  viewWidth = Math.max(1, rect.width);
  viewHeight = Math.max(1, rect.height);

  overlayEl.width = Math.floor(viewWidth * ratio);
  overlayEl.height = Math.floor(viewHeight * ratio);
  overlayEl.style.width = `${viewWidth}px`;
  overlayEl.style.height = `${viewHeight}px`;

  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
}

function handleResize() {
  resizeCanvas();
}

function clearPersonCutout() {
  hasPersonCutout = false;
  segmentationBusy = false;
  lastSegmentationTime = -1;

  if (!personCutoutCtx) {
    return;
  }

  const width = personCutoutCanvas.width || 1;
  const height = personCutoutCanvas.height || 1;
  personCutoutCtx.clearRect(0, 0, width, height);
}

function ensurePersonCutoutSize() {
  if (!personCutoutCtx) {
    return false;
  }

  const width = Math.max(1, videoEl.videoWidth);
  const height = Math.max(1, videoEl.videoHeight);

  if (personCutoutCanvas.width !== width || personCutoutCanvas.height !== height) {
    personCutoutCanvas.width = width;
    personCutoutCanvas.height = height;
  }

  return true;
}

function updatePersonCutout(segmentationMask) {
  if (!stream || !segmentationMask || !ensurePersonCutoutSize() || !personCutoutCtx) {
    hasPersonCutout = false;
    return;
  }

  const width = personCutoutCanvas.width;
  const height = personCutoutCanvas.height;

  personCutoutCtx.clearRect(0, 0, width, height);
  personCutoutCtx.save();
  personCutoutCtx.drawImage(segmentationMask, 0, 0, width, height);
  personCutoutCtx.globalCompositeOperation = "source-in";
  personCutoutCtx.drawImage(videoEl, 0, 0, width, height);
  personCutoutCtx.restore();

  hasPersonCutout = true;
}

function distance(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}

function averagePoint(hand, indices) {
  let x = 0;
  let y = 0;
  for (const index of indices) {
    x += hand[index].x;
    y += hand[index].y;
  }
  return { x: x / indices.length, y: y / indices.length };
}

function normalizeVector(x, y, fallbackX = 0, fallbackY = -1) {
  const len = Math.hypot(x, y);
  if (len < 0.00001) {
    return { x: fallbackX, y: fallbackY };
  }
  return { x: x / len, y: y / len };
}

function rotateVector(vec, radians) {
  const c = Math.cos(radians);
  const s = Math.sin(radians);
  return {
    x: vec.x * c - vec.y * s,
    y: vec.x * s + vec.y * c,
  };
}

function toScreenPoint(point) {
  return {
    x: (1 - point.x) * viewWidth,
    y: point.y * viewHeight,
  };
}

function isFingerExtended(hand, tip, pip, mcp) {
  return hand[mcp].y - hand[tip].y;
}

function isFingerFolded(hand, tip, pip, mcp) {
  return hand[tip].y - hand[mcp].y;
}

function rangeScore(value, low, high) {
  if (value <= low) {
    return 0;
  }
  if (value >= high) {
    return 1;
  }
  return (value - low) / (high - low);
}

function handSealScore(hand) {
  const indexUp = isFingerExtended(hand, 8, 6, 5);
  const middleUp = isFingerExtended(hand, 12, 10, 9);
  const ringDown = isFingerFolded(hand, 16, 14, 13);
  const pinkyDown = isFingerFolded(hand, 20, 18, 17);

  const indexScore = rangeScore(indexUp, 0.01, 0.11);
  const middleScore = rangeScore(middleUp, 0.01, 0.11);
  const ringScore = rangeScore(ringDown, 0.015, 0.12);
  const pinkyScore = rangeScore(pinkyDown, 0.015, 0.12);
  const pairAlign = 1 - Math.min(1, Math.abs(hand[8].x - hand[12].x) / 0.16);

  return (
    indexScore * 0.26 +
    middleScore * 0.26 +
    ringScore * 0.22 +
    pinkyScore * 0.22 +
    pairAlign * 0.04
  );
}

function hasStrongRingPinkyFold(hand) {
  const ringFold = isFingerFolded(hand, 16, 14, 13);
  const pinkyFold = isFingerFolded(hand, 20, 18, 17);
  const ringTipBelowPip = hand[16].y - hand[14].y;
  const pinkyTipBelowPip = hand[20].y - hand[18].y;
  return (
    ringFold > 0.03 &&
    pinkyFold > 0.03 &&
    ringTipBelowPip > 0.008 &&
    pinkyTipBelowPip > 0.008
  );
}

function openPalmScore(hand) {
  const thumbOpen = rangeScore(Math.abs(hand[4].x - hand[2].x), 0.03, 0.16);
  const indexOpen = rangeScore(isFingerExtended(hand, 8, 6, 5), 0.02, 0.13);
  const middleOpen = rangeScore(isFingerExtended(hand, 12, 10, 9), 0.02, 0.13);
  const ringOpen = rangeScore(isFingerExtended(hand, 16, 14, 13), 0.02, 0.12);
  const pinkyOpen = rangeScore(isFingerExtended(hand, 20, 18, 17), 0.02, 0.12);
  const flatness = rangeScore(0.2 - Math.abs(hand[8].y - hand[20].y), 0, 0.14);

  return (
    thumbOpen * 0.16 +
    indexOpen * 0.22 +
    middleOpen * 0.24 +
    ringOpen * 0.18 +
    pinkyOpen * 0.16 +
    flatness * 0.04
  );
}

function wristGrabScore(holderHand, targetHand) {
  const holderPalm = averagePoint(holderHand, [0, 5, 9, 13, 17]);
  const targetWrist = targetHand[0];
  const holderPinch = {
    x: (holderHand[4].x + holderHand[8].x) / 2,
    y: (holderHand[4].y + holderHand[8].y) / 2,
  };

  const nearWristCenter = rangeScore(0.24 - distance(holderPalm, targetWrist), 0, 0.14);
  const nearWristPinch = rangeScore(0.2 - distance(holderPinch, targetWrist), 0, 0.12);
  const wrapWidth = rangeScore(0.2 - Math.abs(holderHand[4].x - holderHand[8].x), 0, 0.1);

  return nearWristCenter * 0.4 + nearWristPinch * 0.5 + wrapWidth * 0.1;
}

function detectChidoriSeal(landmarks) {
  if (!landmarks || landmarks.length < 2) {
    return null;
  }

  const [handA, handB] = landmarks;
  const aOpen = openPalmScore(handA);
  const bOpen = openPalmScore(handB);
  const aHeld = wristGrabScore(handB, handA);
  const bHeld = wristGrabScore(handA, handB);

  const candidateA = aOpen * 0.52 + aHeld * 0.48;
  const candidateB = bOpen * 0.52 + bHeld * 0.48;
  const chooseA = candidateA >= candidateB;
  const bestScore = chooseA ? candidateA : candidateB;
  const emitterHand = chooseA ? handA : handB;
  const holderHand = chooseA ? handB : handA;
  const palm = averagePoint(emitterHand, [0, 5, 9, 13, 17]);
  const tip = emitterHand[12];
  const direction = normalizeVector(tip.x - palm.x, tip.y - palm.y);
  const holderPalm = averagePoint(holderHand, [0, 5, 9, 13, 17]);
  const holderPinch = {
    x: (holderHand[4].x + holderHand[8].x) / 2,
    y: (holderHand[4].y + holderHand[8].y) / 2,
  };
  const wristToWrist = distance(holderHand[0], emitterHand[0]);
  const palmToWrist = distance(holderPalm, emitterHand[0]);
  const pinchToWrist = distance(holderPinch, emitterHand[0]);

  if (
    bestScore < 0.64 ||
    wristToWrist > 0.22 ||
    palmToWrist > 0.2 ||
    pinchToWrist > 0.17
  ) {
    return null;
  }

  return {
    score: bestScore,
    palm,
    direction,
  };
}

function detectRasenganSeal(landmarks) {
  if (!landmarks || landmarks.length < 2) {
    return null;
  }

  const [handA, handB] = landmarks;
  const openA = openPalmScore(handA);
  const openB = openPalmScore(handB);

  const palmA = averagePoint(handA, [0, 5, 9, 13, 17]);
  const palmB = averagePoint(handB, [0, 5, 9, 13, 17]);

  const gapX = Math.abs(palmA.x - palmB.x);
  const gapY = Math.abs(palmA.y - palmB.y);
  const gapDist = distance(palmA, palmB);

  const axisA = normalizeVector(handA[17].x - handA[5].x, handA[17].y - handA[5].y, 1, 0);
  const axisB = normalizeVector(handB[17].x - handB[5].x, handB[17].y - handB[5].y, 1, 0);
  const parallelScore = Math.abs(axisA.x * axisB.x + axisA.y * axisB.y);
  const horizontalA = rangeScore(
    Math.abs(handA[17].x - handA[5].x) - Math.abs(handA[17].y - handA[5].y),
    0.02,
    0.16
  );
  const horizontalB = rangeScore(
    Math.abs(handB[17].x - handB[5].x) - Math.abs(handB[17].y - handB[5].y),
    0.02,
    0.16
  );

  const openScore = (openA + openB) / 2;
  const sameColumn = rangeScore(0.2 - gapX, 0, 0.14);
  const verticalStack = rangeScore(gapY - gapX, 0.04, 0.22);
  const gapLower = rangeScore(gapDist, 0.1, 0.22);
  const gapUpper = rangeScore(0.48 - gapDist, 0, 0.18);
  const gapScore = Math.min(gapLower, gapUpper);
  const shapeScore = (horizontalA + horizontalB + parallelScore) / 3;

  const topHand = palmA.y <= palmB.y ? handA : handB;
  const bottomHand = topHand === handA ? handB : handA;
  const topFingerTowardCenter = rangeScore(topHand[9].y - topHand[0].y, 0.02, 0.13);
  const bottomFingerTowardCenter = rangeScore(bottomHand[0].y - bottomHand[9].y, 0.02, 0.13);
  const facingEachOther = (topFingerTowardCenter + bottomFingerTowardCenter) / 2;

  const score =
    openScore * 0.3 +
    sameColumn * 0.2 +
    verticalStack * 0.16 +
    gapScore * 0.13 +
    shapeScore * 0.09 +
    facingEachOther * 0.12;

  if (
    openScore < 0.56 ||
    sameColumn < 0.38 ||
    verticalStack < 0.45 ||
    gapScore < 0.35 ||
    shapeScore < 0.45 ||
    facingEachOther < 0.35 ||
    score < 0.6
  ) {
    return null;
  }

  return {
    score,
    center: {
      x: (palmA.x + palmB.x) / 2,
      y: (palmA.y + palmB.y) / 2,
    },
    gap: gapDist,
    axis: normalizeVector(palmB.x - palmA.x, palmB.y - palmA.y, 1, 0),
  };
}

function isKageBunshinSeal(landmarks) {
  if (!landmarks || landmarks.length < 2) {
    return false;
  }

  const [handA, handB] = landmarks;
  if (!hasStrongRingPinkyFold(handA) || !hasStrongRingPinkyFold(handB)) {
    return false;
  }

  const handScore = (handSealScore(handA) + handSealScore(handB)) / 2;

  const wristDistance = distance(handA[0], handB[0]);
  const tipIndexDistance = distance(handA[8], handB[8]);
  const tipMiddleDistance = distance(handA[12], handB[12]);
  if (
    wristDistance > 0.5 ||
    tipIndexDistance > 0.34 ||
    tipMiddleDistance > 0.34
  ) {
    return false;
  }

  const wristsClose = rangeScore(0.52 - wristDistance, 0, 0.22);
  const fingersClose =
    (rangeScore(0.38 - tipIndexDistance, 0, 0.21) +
      rangeScore(0.38 - tipMiddleDistance, 0, 0.21)) /
    2;
  const similarHeight = rangeScore(0.38 - Math.abs(handA[0].y - handB[0].y), 0, 0.22);

  const closenessScore =
    wristsClose * 0.42 + fingersClose * 0.38 + similarHeight * 0.2;
  const totalScore = handScore * 0.68 + closenessScore * 0.32;

  return totalScore > 0.58;
}

function drawHandLandmarks(landmarks) {
  ctx.lineWidth = 2;
  ctx.strokeStyle = "rgba(99, 211, 255, 0.85)";
  ctx.fillStyle = "rgba(255, 255, 255, 0.95)";

  for (const hand of landmarks) {
    for (const [from, to] of HAND_CONNECTIONS) {
      const a = hand[from];
      const b = hand[to];

      ctx.beginPath();
      ctx.moveTo((1 - a.x) * viewWidth, a.y * viewHeight);
      ctx.lineTo((1 - b.x) * viewWidth, b.y * viewHeight);
      ctx.stroke();
    }

    for (const point of hand) {
      ctx.beginPath();
      ctx.arc((1 - point.x) * viewWidth, point.y * viewHeight, 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

function drawMirroredClone(source, xOffset, yOffset, scale, tilt, alpha) {
  const drawW = viewWidth * scale;
  const drawH = viewHeight * scale;
  const cx = viewWidth * (0.5 + xOffset);
  const cy = viewHeight * (0.5 + yOffset);
  const sourceW = Math.max(1, source.width ?? source.videoWidth ?? 1);
  const sourceH = Math.max(1, source.height ?? source.videoHeight ?? 1);

  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.translate(cx, cy);
  ctx.rotate(tilt);
  ctx.scale(-1, 1);
  ctx.drawImage(
    source,
    0,
    0,
    sourceW,
    sourceH,
    -drawW / 2,
    -drawH / 2,
    drawW,
    drawH
  );
  ctx.restore();
}

function renderCloneEffect(now) {
  if (
    activeJutsu !== "kage" ||
    !stream ||
    videoEl.readyState < 2 ||
    effectUntil <= now ||
    !hasPersonCutout
  ) {
    return;
  }

  const source = personCutoutCanvas;

  for (let i = 0; i < CLONE_COUNT; i += 1) {
    const base = CLONE_OFFSETS[i];
    const drift = Math.sin((now / 190) + i * 0.75) * 0.012;
    const pulse = 1 + Math.sin((now / 230) + i) * 0.01;
    const alpha = 1;
    drawMirroredClone(
      source,
      base.x + drift,
      base.y + drift * 0.35,
      base.scale * pulse,
      base.tilt + drift * 0.15,
      alpha
    );
  }
}

function drawElectricBolt(start, end, now, seed, alpha = 1) {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const len = Math.hypot(dx, dy);
  if (len < 5) {
    return;
  }

  const ux = dx / len;
  const uy = dy / len;
  const px = -uy;
  const py = ux;
  const segments = 7;

  ctx.save();
  ctx.strokeStyle = `rgba(106, 225, 255, ${0.5 * alpha})`;
  ctx.lineWidth = 7;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.beginPath();
  ctx.moveTo(start.x, start.y);
  for (let i = 1; i < segments; i += 1) {
    const t = i / segments;
    const baseX = start.x + dx * t;
    const baseY = start.y + dy * t;
    const jitter = Math.sin((now / 42) + seed * 1.9 + i * 1.3) * (20 * (1 - t));
    ctx.lineTo(baseX + px * jitter, baseY + py * jitter);
  }
  ctx.lineTo(end.x, end.y);
  ctx.stroke();
  ctx.restore();

  ctx.save();
  ctx.strokeStyle = `rgba(220, 249, 255, ${0.92 * alpha})`;
  ctx.lineWidth = 2.3;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.beginPath();
  ctx.moveTo(start.x, start.y);
  for (let i = 1; i < segments; i += 1) {
    const t = i / segments;
    const baseX = start.x + dx * t;
    const baseY = start.y + dy * t;
    const jitter = Math.sin((now / 32) + seed * 2.2 + i * 1.65) * (11 * (1 - t));
    ctx.lineTo(baseX + px * jitter, baseY + py * jitter);
  }
  ctx.lineTo(end.x, end.y);
  ctx.stroke();
  ctx.restore();
}

function renderChidoriEffect(now) {
  if (activeJutsu !== "chidori" || effectUntil <= now || !activeChidoriEmitter) {
    return;
  }

  const start = toScreenPoint(activeChidoriEmitter.palm);
  const dir = normalizeVector(
    -activeChidoriEmitter.direction.x,
    activeChidoriEmitter.direction.y
  );
  const life = (effectUntil - now) / CHIDORI_EFFECT_MS;
  const radius = Math.max(75, Math.min(viewWidth, viewHeight) * 0.13);

  const glow = ctx.createRadialGradient(start.x, start.y, 2, start.x, start.y, radius * 1.7);
  glow.addColorStop(0, "rgba(234, 252, 255, 0.95)");
  glow.addColorStop(0.4, "rgba(99, 230, 255, 0.85)");
  glow.addColorStop(1, "rgba(0, 98, 255, 0)");
  ctx.save();
  ctx.fillStyle = glow;
  ctx.beginPath();
  ctx.arc(start.x, start.y, radius * 1.7, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();

  const boltCount = 8;
  const baseLen = Math.min(viewWidth, viewHeight) * 0.9;
  for (let i = 0; i < boltCount; i += 1) {
    const fan = (i - (boltCount - 1) / 2) * 0.21;
    const wobble = Math.sin((now / 125) + i) * 0.12;
    const vec = rotateVector(dir, fan + wobble);
    const len = baseLen * (0.74 + Math.sin((now / 140) + i * 1.3) * 0.2 + life * 0.2);
    const end = {
      x: start.x + vec.x * len,
      y: start.y + vec.y * len,
    };
    drawElectricBolt(start, end, now, i + 1, 0.55 + life * 0.45);
  }
}

function renderRasenganEffect(now) {
  if (activeJutsu !== "rasengan" || effectUntil <= now || !activeRasenganData) {
    return;
  }

  const center = toScreenPoint(activeRasenganData.center);
  const life = (effectUntil - now) / RASENGAN_EFFECT_MS;
  const baseSize = Math.min(viewWidth, viewHeight);
  const radius = Math.max(
    28,
    Math.min(baseSize * 0.26, baseSize * (0.06 + activeRasenganData.gap * 0.5))
  );

  const coreGlow = ctx.createRadialGradient(
    center.x,
    center.y,
    1,
    center.x,
    center.y,
    radius * 1.5
  );
  coreGlow.addColorStop(0, "rgba(241, 251, 255, 0.95)");
  coreGlow.addColorStop(0.35, "rgba(149, 226, 255, 0.92)");
  coreGlow.addColorStop(0.75, "rgba(52, 147, 255, 0.62)");
  coreGlow.addColorStop(1, "rgba(34, 96, 255, 0)");
  ctx.save();
  ctx.fillStyle = coreGlow;
  ctx.beginPath();
  ctx.arc(center.x, center.y, radius * 1.5, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();

  for (let i = 0; i < 7; i += 1) {
    const phase = now / 340 + i * 0.73;
    const ringR = radius * (0.58 + i * 0.085);
    ctx.save();
    ctx.translate(center.x, center.y);
    ctx.rotate(phase);
    ctx.scale(1, 0.62 + Math.sin(now / 620 + i) * 0.05);
    ctx.strokeStyle = `rgba(189, 241, 255, ${0.66 - i * 0.07 + life * 0.14})`;
    ctx.lineWidth = 2.2 - i * 0.2;
    ctx.beginPath();
    ctx.arc(0, 0, ringR, 0.1 * Math.PI, 1.8 * Math.PI);
    ctx.stroke();
    ctx.restore();
  }

  const particles = 18;
  for (let i = 0; i < particles; i += 1) {
    const angle = (i / particles) * Math.PI * 2 + now / 290;
    const orbit = radius * (0.6 + Math.sin(now / 270 + i) * 0.15);
    const px = center.x + Math.cos(angle) * orbit;
    const py = center.y + Math.sin(angle) * orbit * 0.78;
    ctx.save();
    ctx.fillStyle = `rgba(220, 250, 255, ${0.58 + Math.sin(now / 180 + i) * 0.2})`;
    ctx.beginPath();
    ctx.arc(px, py, 1.8 + Math.sin(now / 210 + i) * 0.6, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }
}

function getAudioContext() {
  if (!audioCtx) {
    audioCtx = new AudioContext();
  }

  if (audioCtx.state === "suspended") {
    audioCtx.resume();
  }

  return audioCtx;
}

function pickNarutoStyleVoice() {
  if (!("speechSynthesis" in window)) {
    return null;
  }

  const voices = window.speechSynthesis.getVoices();
  if (!voices || voices.length === 0) {
    return null;
  }

  const japaneseVoices = voices.filter((voice) => /ja/i.test(voice.lang || ""));
  if (japaneseVoices.length === 0) {
    return null;
  }

  const preferred = japaneseVoices.find((voice) =>
    /(kyoko|otoya|google|japanese)/i.test(voice.name || "")
  );
  return preferred || japaneseVoices[0];
}

function speakNarutoStyle(text, rate, pitch, fallbackText = text) {
  if (!("speechSynthesis" in window)) {
    return;
  }

  const synth = window.speechSynthesis;
  // Warm voice list access improves reliability on some browsers.
  synth.getVoices();

  const utterance = new SpeechSynthesisUtterance(text);
  const voice = pickNarutoStyleVoice();
  if (voice) {
    utterance.voice = voice;
    utterance.lang = voice.lang;
  } else {
    // Fallback for browsers without Japanese voices.
    utterance.text = fallbackText;
    utterance.lang = "en-US";
  }
  utterance.rate = rate;
  utterance.pitch = pitch;
  utterance.volume = 1;

  utterance.onerror = () => {
    if (!fallbackText || fallbackText === text) {
      return;
    }
    const backup = new SpeechSynthesisUtterance(fallbackText);
    backup.lang = "en-US";
    backup.rate = 1;
    backup.pitch = 1;
    backup.volume = 1;
    synth.cancel();
    synth.speak(backup);
  };

  synth.cancel();
  // Small delay avoids cancel/speak race in some engines.
  window.setTimeout(() => synth.speak(utterance), 35);
}

function primeSpeechSynthesis() {
  if (!("speechSynthesis" in window)) {
    return;
  }
  const synth = window.speechSynthesis;
  synth.getVoices();
  const primer = new SpeechSynthesisUtterance(" ");
  primer.volume = 0;
  synth.speak(primer);
  synth.cancel();
}

function playKageBunshinSound() {
  const ac = getAudioContext();
  const now = ac.currentTime;

  const osc = ac.createOscillator();
  const gain = ac.createGain();
  osc.type = "sawtooth";
  osc.frequency.setValueAtTime(170, now);
  osc.frequency.exponentialRampToValueAtTime(70, now + 0.42);

  gain.gain.setValueAtTime(0.0001, now);
  gain.gain.exponentialRampToValueAtTime(0.18, now + 0.03);
  gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.45);

  osc.connect(gain);
  gain.connect(ac.destination);
  osc.start(now);
  osc.stop(now + 0.46);

  speakNarutoStyle("影分身の術!", 1.06, 0.96, "Kage Bunshin no Jutsu");
}

function ensureNoiseBuffer(ac) {
  if (noiseBuffer && noiseBuffer.sampleRate === ac.sampleRate) {
    return noiseBuffer;
  }
  const buffer = ac.createBuffer(1, Math.floor(ac.sampleRate * 1.1), ac.sampleRate);
  const data = buffer.getChannelData(0);
  for (let i = 0; i < data.length; i += 1) {
    data[i] = (Math.random() * 2 - 1) * 0.9;
  }
  noiseBuffer = buffer;
  return buffer;
}

function playChidoriSound() {
  const ac = getAudioContext();
  const now = ac.currentTime;

  const noise = ac.createBufferSource();
  noise.buffer = ensureNoiseBuffer(ac);
  const band = ac.createBiquadFilter();
  band.type = "bandpass";
  band.frequency.setValueAtTime(2200, now);
  band.frequency.exponentialRampToValueAtTime(1200, now + 0.75);
  band.Q.value = 0.9;
  const noiseGain = ac.createGain();
  noiseGain.gain.setValueAtTime(0.0001, now);
  noiseGain.gain.exponentialRampToValueAtTime(0.25, now + 0.06);
  noiseGain.gain.exponentialRampToValueAtTime(0.0001, now + 0.78);
  noise.connect(band);
  band.connect(noiseGain);
  noiseGain.connect(ac.destination);
  noise.start(now);
  noise.stop(now + 0.8);

  const arcOsc = ac.createOscillator();
  const arcGain = ac.createGain();
  arcOsc.type = "square";
  arcOsc.frequency.setValueAtTime(560, now);
  arcOsc.frequency.exponentialRampToValueAtTime(240, now + 0.62);
  arcGain.gain.setValueAtTime(0.0001, now);
  arcGain.gain.exponentialRampToValueAtTime(0.08, now + 0.03);
  arcGain.gain.exponentialRampToValueAtTime(0.0001, now + 0.66);
  arcOsc.connect(arcGain);
  arcGain.connect(ac.destination);
  arcOsc.start(now);
  arcOsc.stop(now + 0.67);

  speakNarutoStyle("千鳥!", 1.12, 1.02, "Chidori!");
}

function playRasenganSound() {
  const ac = getAudioContext();
  const now = ac.currentTime;

  const swirl = ac.createOscillator();
  const swirlGain = ac.createGain();
  swirl.type = "sine";
  swirl.frequency.setValueAtTime(180, now);
  swirl.frequency.linearRampToValueAtTime(420, now + 0.48);
  swirl.frequency.linearRampToValueAtTime(260, now + 0.95);
  swirlGain.gain.setValueAtTime(0.0001, now);
  swirlGain.gain.exponentialRampToValueAtTime(0.1, now + 0.05);
  swirlGain.gain.exponentialRampToValueAtTime(0.0001, now + 0.98);
  swirl.connect(swirlGain);
  swirlGain.connect(ac.destination);
  swirl.start(now);
  swirl.stop(now + 1);

  const hiss = ac.createBufferSource();
  hiss.buffer = ensureNoiseBuffer(ac);
  const hissFilter = ac.createBiquadFilter();
  hissFilter.type = "highpass";
  hissFilter.frequency.setValueAtTime(900, now);
  hissFilter.frequency.exponentialRampToValueAtTime(1700, now + 0.6);
  const hissGain = ac.createGain();
  hissGain.gain.setValueAtTime(0.0001, now);
  hissGain.gain.exponentialRampToValueAtTime(0.07, now + 0.04);
  hissGain.gain.exponentialRampToValueAtTime(0.0001, now + 0.82);
  hiss.connect(hissFilter);
  hissFilter.connect(hissGain);
  hissGain.connect(ac.destination);
  hiss.start(now);
  hiss.stop(now + 0.84);

  speakNarutoStyle("螺旋丸!", 1.1, 1, "Rasengan!");
}

async function createHandLandmarker() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm"
  );

  return HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate: "GPU",
    },
    numHands: 2,
    runningMode: "VIDEO",
    minHandDetectionConfidence: 0.62,
    minHandPresenceConfidence: 0.62,
    minTrackingConfidence: 0.58,
  });
}

async function createSelfieSegmentation() {
  const SelfieSegmentation = window.SelfieSegmentation;
  if (!SelfieSegmentation) {
    throw new Error("SelfieSegmentation failed to load.");
  }

  const segmenter = new SelfieSegmentation({
    locateFile(file) {
      return `https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation/${file}`;
    },
  });

  segmenter.setOptions({
    modelSelection: 1,
  });

  segmenter.onResults((results) => {
    updatePersonCutout(results?.segmentationMask);
  });

  return segmenter;
}

function queueSegmentation() {
  if (!selfieSegmentation || segmentationBusy || videoEl.readyState < 2) {
    return;
  }

  if (lastSegmentationTime === videoEl.currentTime) {
    return;
  }

  lastSegmentationTime = videoEl.currentTime;
  segmentationBusy = true;

  Promise.resolve(selfieSegmentation.send({ image: videoEl }))
    .catch((error) => {
      console.error("Segmentation frame failed", error);
      hasPersonCutout = false;
    })
    .finally(() => {
      segmentationBusy = false;
    });
}

function stopLoop() {
  if (rafId) {
    cancelAnimationFrame(rafId);
    rafId = null;
  }
}

function stopStream() {
  if (!stream) {
    return;
  }
  for (const track of stream.getTracks()) {
    track.stop();
  }
  stream = null;
}

function triggerKageBunshin(now) {
  activeJutsu = "kage";
  activeChidoriEmitter = null;
  activeRasenganData = null;
  effectUntil = now + KAGE_EFFECT_MS;
  cooldownUntil = now + JUTSU_COOLDOWN_MS;
  setHud("Kage Bunshin detected! 10 clones deployed in-frame.");
  setStatus("Jutsu active");
  playKageBunshinSound();
}

function triggerChidori(now, emitter) {
  activeJutsu = "chidori";
  activeChidoriEmitter = emitter;
  activeRasenganData = null;
  effectUntil = now + CHIDORI_EFFECT_MS;
  cooldownUntil = now + JUTSU_COOLDOWN_MS;
  setHud("Chidori detected! Lightning unleashed from your palm.");
  setStatus("Jutsu active");
  playChidoriSound();
}

function triggerRasengan(now, rasenganData) {
  activeJutsu = "rasengan";
  activeRasenganData = rasenganData;
  activeChidoriEmitter = null;
  effectUntil = now + RASENGAN_EFFECT_MS;
  cooldownUntil = now + JUTSU_COOLDOWN_MS;
  setHud("Rasengan detected! Orb formed between your palms.");
  setStatus("Jutsu active");
  playRasenganSound();
}

function frameLoop() {
  const now = performance.now();

  ctx.clearRect(0, 0, viewWidth, viewHeight);

  if (activeJutsu && now > effectUntil) {
    activeJutsu = null;
    activeChidoriEmitter = null;
    activeRasenganData = null;
  }

  if (videoEl.readyState >= 2 && lastVideoTime !== videoEl.currentTime) {
    lastVideoTime = videoEl.currentTime;
    queueSegmentation();

    if (handLandmarker) {
      const results = handLandmarker.detectForVideo(videoEl, now);
      latestHands = results.landmarks ?? [];
      const chidoriCandidate = detectChidoriSeal(latestHands);
      const rasenganCandidate = detectRasenganSeal(latestHands);

      if (hasPersonCutout) {
        if (activeJutsu === "rasengan" && rasenganCandidate) {
          activeRasenganData = rasenganCandidate;
        }

        if (activeJutsu === "chidori" && chidoriCandidate) {
          activeChidoriEmitter = chidoriCandidate;
        }

        if (now > cooldownUntil) {
          if (isKageBunshinSeal(latestHands)) {
            triggerKageBunshin(now);
          } else if (rasenganCandidate) {
            triggerRasengan(now, rasenganCandidate);
          } else if (chidoriCandidate) {
            triggerChidori(now, chidoriCandidate);
          }
        }
      }
    }
  }

  renderCloneEffect(now);
  renderChidoriEffect(now);
  renderRasenganEffect(now);

  const sawHands = latestHands.length > 0;

  if (activeJutsu === "kage" && effectUntil > now) {
    setHud("Kage Bunshin active: 10 in-frame clones.");
    setStatus("Jutsu active");
  } else if (activeJutsu === "chidori" && effectUntil > now) {
    setHud("Chidori active: lightning flowing from your palm.");
    setStatus("Jutsu active");
  } else if (activeJutsu === "rasengan" && effectUntil > now) {
    setHud("Rasengan active: rotating blue orb between palms.");
    setStatus("Jutsu active");
  } else if (sawHands) {
    setHud("Hands found. Try Kage Bunshin, Chidori, or Rasengan seal.");
    setStatus("Tracking hands");
  } else if (!hasPersonCutout) {
    setHud("Camera running. Building person mask...");
    setStatus("Preparing clones");
  } else {
    setHud("No hands detected. Keep both hands in frame.");
    setStatus("Camera running");
  }
  rafId = requestAnimationFrame(frameLoop);
}

async function startCamera() {
  startBtn.disabled = true;
  setStatus("Loading models…");
  primeSpeechSynthesis();

  try {
    if (!handLandmarker) {
      handLandmarker = await createHandLandmarker();
    }
    if (!selfieSegmentation) {
      selfieSegmentation = await createSelfieSegmentation();
    }

    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
      audio: false,
    });

    videoEl.srcObject = stream;
    await videoEl.play();

    handleResize();
    window.addEventListener("resize", handleResize);

    stopBtn.disabled = false;
    setHud("Camera ready. Form Kage Bunshin, Chidori, or Rasengan hand seal.");
    setStatus("Camera running");

    lastVideoTime = -1;
    latestHands = [];
    clearPersonCutout();
    activeJutsu = null;
    activeChidoriEmitter = null;
    activeRasenganData = null;
    effectUntil = 0;
    cooldownUntil = 0;
    stopLoop();
    frameLoop();
  } catch (error) {
    console.error(error);
    setStatus("Error starting camera");
    setHud("Failed to start camera or model. Check camera permissions.");
    startBtn.disabled = false;
  }
}

function stopCamera() {
  stopLoop();
  stopStream();
  window.removeEventListener("resize", handleResize);
  latestHands = [];
  clearPersonCutout();
  activeJutsu = null;
  activeChidoriEmitter = null;
  activeRasenganData = null;
  effectUntil = 0;
  cooldownUntil = 0;

  ctx.clearRect(0, 0, viewWidth, viewHeight);
  videoEl.srcObject = null;

  setHud("Stopped.");
  setStatus("Idle");

  startBtn.disabled = false;
  stopBtn.disabled = true;
}

startBtn.addEventListener("click", startCamera);
stopBtn.addEventListener("click", stopCamera);

window.addEventListener("beforeunload", () => {
  stopLoop();
  stopStream();
});
