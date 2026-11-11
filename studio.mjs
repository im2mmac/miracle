import { FilesetResolver, HandLandmarker, ImageSegmenter } from "./vendor/vision_bundle.mjs";

const byId = (id) => document.getElementById(id);
const stage = byId("stage");
const ocean = byId("ocean");
const webcam = byId("webcam");
const performer = byId("performer");
const landmarks = byId("landmarks");
const guide = byId("guide");
const threshold = byId("threshold");
const tracking = byId("tracking-status");
const sound = byId("sound");
const showLandmarks = byId("show-landmarks");
const personContext = performer.getContext("2d");
const landmarkContext = landmarks.getContext("2d");
const frame = document.createElement("canvas");
const frameContext = frame.getContext("2d");
const alpha = document.createElement("canvas");
const alphaContext = alpha.getContext("2d");

const WIDTH = performer.width;
const HEIGHT = performer.height;
const HOLD_MS = 220;
const RELEASE_MARGIN = 0.06;
const RESET_AFTER_MS = 1250;
const CLIP_SECONDS = 8;

let taskPromise;
let camera = "off";
let cameraSession = 0;
let animation = 0;
let stream;
let scene = "idle";
let playbackSession = 0;
let openedAt = 0;
let placement;
let maskPixels;
let currentHands = [];
let lastCameraFrame = -1;
let lastSegmentation = -Infinity;
let lastHandCheck = -Infinity;
let handRaisedSince;
let handLoweredSince;
let armed = false;

function asset(path) {
  return new URL(path, import.meta.url).href;
}

function loadTasks() {
  taskPromise ??= (async () => {
    const files = await FilesetResolver.forVisionTasks(asset("./vendor/wasm"));
    const segmenter = await ImageSegmenter.createFromOptions(files, {
      baseOptions: { modelAssetPath: asset("./vendor/models/selfie.tflite"), delegate: "CPU" },
      runningMode: "VIDEO",
      outputCategoryMask: false,
      outputConfidenceMasks: true,
    });
    try {
      const hands = await HandLandmarker.createFromOptions(files, {
        baseOptions: { modelAssetPath: asset("./vendor/models/hands.task"), delegate: "CPU" },
        runningMode: "VIDEO",
        numHands: 2,
        minHandDetectionConfidence: 0.55,
      });
      return { segmenter, hands };
    } catch (error) {
      segmenter.close();
      throw error;
    }
  })().catch((error) => {
    taskPromise = undefined;
    throw error;
  });
  return taskPromise;
}

function setNotice(message) {
  const notice = byId("notice");
  notice.textContent = message;
  notice.hidden = !message;
}

function setTracking(message, active = false, progress = 0) {
  if (tracking.textContent !== message) tracking.textContent = message;
  tracking.parentElement.dataset.active = String(active);
  byId("gesture-progress").style.width = Math.round(100 * progress) + "%";
}

function updatePrompt() {
  const labels = { idle: "재생 대기", playing: "장면 재생 중", opened: "바닷길 열림" };
  byId("scene-status").textContent = labels[scene];
  if (scene === "playing") {
    byId("stage-prompt").hidden = true;
    return;
  }

  const prompt = byId("stage-prompt");
  prompt.hidden = false;
  if (scene === "opened") {
    prompt.textContent = camera === "on" ? "손을 내리면 다음 장면을 준비해요." : "바닷길이 열렸어요.";
  } else {
    prompt.textContent = {
      off: "카메라를 시작하면 이 장면에 내 모습이 나타나요.",
      loading: "카메라와 손 인식 모델을 준비하고 있어요.",
      on: "손목을 점선 위로 올리고 잠시 유지해보세요.",
    }[camera];
  }
}

function updateGuide() {
  guide.hidden = camera !== "on" || scene !== "idle" || !placement;
  if (guide.hidden) return;
  const stageHeight = stage.clientHeight;
  const contentHeight = Math.min(stageHeight, stage.clientWidth * HEIGHT / WIDTH);
  const fraction = (placement.y + Number(threshold.value) / 100 * placement.height) / HEIGHT;
  guide.style.top = (stageHeight - contentHeight) / 2 + fraction * contentHeight + "px";
}

function setCamera(next) {
  camera = next;
  const label = {
    off: ["카메라 꺼짐", "카메라 시작"],
    loading: ["카메라 준비 중", "시작 취소"],
    on: ["카메라 켜짐", "카메라 끄기"],
  }[next];
  byId("camera-badge").dataset.state = next;
  byId("camera-toggle").dataset.state = next;
  byId("camera-status").textContent = label[0];
  byId("camera-button-text").textContent = label[1];
  if (next === "off") setTracking("카메라 대기 중");
  if (next === "loading") setTracking("인식 모델 준비 중");
  if (next === "on") setTracking("손을 화면 안에 보여주세요");
  updatePrompt();
  updateGuide();
}

function updateTime() {
  const duration = Number.isFinite(ocean.duration) ? ocean.duration : CLIP_SECONDS;
  const position = scene === "idle" ? 0 : scene === "opened" ? duration : ocean.currentTime;
  const format = (seconds) => "0:" + String(Math.floor(seconds + 0.01)).padStart(2, "0");
  byId("clip-time").textContent = format(position) + " / " + format(duration);
  byId("clip-progress").style.width = Math.min(100, position / duration * 100) + "%";
}

function setScene(next) {
  scene = next;
  stage.dataset.scene = next;
  byId("preview").disabled = next === "playing";
  byId("preview-text").textContent = next === "opened" ? "다시 재생" : "영상 미리보기";
  byId("reset").disabled = next === "idle";
  updatePrompt();
  updateGuide();
  updateTime();
}

function resetGesture() {
  armed = false;
  handRaisedSince = undefined;
  handLoweredSince = undefined;
}

function resetScene() {
  playbackSession += 1;
  ocean.pause();
  if (ocean.readyState >= 1) ocean.currentTime = 0;
  resetGesture();
  setScene("idle");
}

async function playScene() {
  if (scene === "playing") return;
  const session = ++playbackSession;
  resetGesture();
  setNotice("");
  if (ocean.readyState >= 1) ocean.currentTime = 0;
  ocean.muted = !sound.checked;
  setScene("playing");
  try {
    await ocean.play();
  } catch (error) {
    if (session !== playbackSession) return;
    if (error.name === "NotAllowedError" && !ocean.muted) {
      sound.checked = false;
      ocean.muted = true;
      try {
        await ocean.play();
        if (session === playbackSession) setNotice("브라우저가 소리 자동 재생을 제한해 영상만 재생했어요. 소리를 켜고 미리보기 버튼으로 다시 재생할 수 있어요.");
        return;
      } catch (retryError) {
        error = retryError;
      }
    }
    if (session !== playbackSession) return;
    console.error(error);
    resetScene();
    setNotice("영상을 재생할 수 없어요. 페이지를 새로고침하거나 영상 미리보기를 다시 눌러보세요.");
  }
}

function positionFrame() {
  const width = webcam.videoWidth;
  const height = webcam.videoHeight;
  const scale = Math.min(640 / width, 480 / height, 1);
  const nextWidth = Math.round(width * scale);
  const nextHeight = Math.round(height * scale);
  if (frame.width !== nextWidth || frame.height !== nextHeight) {
    frame.width = nextWidth;
    frame.height = nextHeight;
  }
  const fit = 0.94 * Math.min(WIDTH / width, HEIGHT / height);
  const next = {
    x: (WIDTH - width * fit) / 2,
    y: HEIGHT - height * fit,
    width: width * fit,
    height: height * fit,
  };
  if (!placement || placement.width !== next.width || placement.height !== next.height) {
    placement = next;
    updateGuide();
  }
}

function composite(result) {
  const mask = result.confidenceMasks?.[1] ?? result.confidenceMasks?.[0];
  personContext.clearRect(0, 0, WIDTH, HEIGHT);
  if (!mask) return;

  if (!maskPixels || alpha.width !== mask.width || alpha.height !== mask.height) {
    alpha.width = mask.width;
    alpha.height = mask.height;
    maskPixels = alphaContext.createImageData(mask.width, mask.height);
  }
  const confidence = mask.getAsFloat32Array();
  const pixels = maskPixels.data;
  for (let index = 0; index < confidence.length; index += 1) {
    const edge = Math.min(1, Math.max(0, (confidence[index] - 0.33) / 0.39));
    pixels[index * 4 + 3] = 255 * edge * edge * (3 - 2 * edge);
  }
  alphaContext.putImageData(maskPixels, 0, 0);

  const { x, y, width, height } = placement;
  personContext.save();
  personContext.translate(x + width, y);
  personContext.scale(-1, 1);
  personContext.drawImage(alpha, 0, 0, width, height);
  personContext.globalCompositeOperation = "source-in";
  personContext.drawImage(frame, 0, 0, width, height);
  personContext.restore();
}

function drawHands() {
  landmarkContext.clearRect(0, 0, WIDTH, HEIGHT);
  if (!placement || !showLandmarks.checked) return;
  const { x, y, width, height } = placement;
  const ctx = landmarkContext;
  ctx.save();
  ctx.lineWidth = 2.5;
  ctx.lineCap = "round";
  ctx.strokeStyle = "#a9f4d9";
  ctx.shadowColor = "#063d3a";
  ctx.shadowBlur = 5;
  for (const hand of currentHands) {
    const points = hand.map((point) => [x + (1 - point.x) * width, y + point.y * height]);
    ctx.beginPath();
    for (const { start, end } of HandLandmarker.HAND_CONNECTIONS) {
      ctx.moveTo(...points[start]);
      ctx.lineTo(...points[end]);
    }
    ctx.stroke();
    points.forEach(([px, py], index) => {
      ctx.beginPath();
      ctx.fillStyle = index === 0 ? "#ffd591" : "#e7fff5";
      ctx.arc(px, py, index === 0 ? 6 : 3.1, 0, Math.PI * 2);
      ctx.fill();
    });
  }
  ctx.restore();
}

function checkGesture(now) {
  const limit = Number(threshold.value) / 100;
  const wrists = currentHands.map((hand) => hand[0].y);
  const raised = wrists.some((height) => height < limit);
  const lowered = wrists.every((height) => height > limit + RELEASE_MARGIN);
  if (lowered) {
    handLoweredSince ??= now;
  } else {
    handLoweredSince = undefined;
  }
  const released = handLoweredSince !== undefined &&
    now - handLoweredSince >= (wrists.length ? 170 : 560);
  if (scene === "opened" && released && now - openedAt >= RESET_AFTER_MS) resetScene();
  if (scene !== "idle") {
    setTracking(scene === "playing" ? "재생 중" : "손을 내리면 다시 대기", wrists.length > 0);
    return;
  }

  if (released) armed = true;
  if (!armed || !raised) {
    handRaisedSince = undefined;
    const message = !wrists.length ? "손을 화면 안에 보여주세요" :
      !armed ? "손을 기준선 아래로 내려주세요" : "손목을 기준선 위로 올려주세요";
    setTracking(message, wrists.length > 0);
    return;
  }

  handRaisedSince ??= now;
  const progress = Math.min(1, (now - handRaisedSince) / HOLD_MS);
  setTracking("손을 잠시 유지하세요", true, progress);
  if (progress === 1) void playScene();
}

function cameraLoop(tasks, now) {
  if (camera !== "on") return;
  try {
    if (webcam.readyState >= 2 && webcam.currentTime !== lastCameraFrame) {
      const segment = now - lastSegmentation >= 39;
      const detect = now - lastHandCheck >= 52;
      if (segment || detect) {
        lastCameraFrame = webcam.currentTime;
        positionFrame();
        frameContext.drawImage(webcam, 0, 0, frame.width, frame.height);
        if (segment) {
          lastSegmentation = now;
          tasks.segmenter.segmentForVideo(frame, now, composite);
        }
        if (detect) {
          if (now - lastHandCheck > 270) {
            handRaisedSince = undefined;
            handLoweredSince = undefined;
          }
          lastHandCheck = now;
          currentHands = tasks.hands.detectForVideo(frame, now).landmarks;
          drawHands();
          checkGesture(now);
        }
      }
    }
    animation = requestAnimationFrame((time) => cameraLoop(tasks, time));
  } catch (error) {
    console.error(error);
    stopCamera();
    setNotice("카메라 프레임을 처리하지 못했어요. 카메라를 다시 시작해보세요.");
  }
}

function stopCamera() {
  cameraSession += 1;
  cancelAnimationFrame(animation);
  stream?.getTracks().forEach((track) => track.stop());
  stream = undefined;
  webcam.srcObject = null;
  currentHands = [];
  placement = undefined;
  personContext.clearRect(0, 0, WIDTH, HEIGHT);
  drawHands();
  resetGesture();
  setCamera("off");
}

function cameraError(error, hasStream) {
  if (hasStream) return "카메라 또는 MediaPipe 모델을 준비하지 못했어요. 카메라를 다시 시작해보세요.";
  if (error.name === "NotAllowedError") return "브라우저 주소창에서 카메라 권한을 허용한 뒤 다시 시작해보세요.";
  if (error.name === "NotFoundError") return "연결된 카메라를 찾지 못했어요. 웹캠을 연결한 뒤 다시 시작해보세요.";
  if (error.name === "NotReadableError") return "카메라를 열 수 없어요. 다른 앱에서 카메라를 사용 중인지 확인해보세요.";
  return "카메라 연결에 실패했어요. 카메라를 확인하고 다시 시작해보세요.";
}

async function startCamera() {
  setNotice("");
  if (!navigator.mediaDevices?.getUserMedia) {
    setNotice("카메라를 사용하려면 HTTPS 또는 localhost 주소로 페이지를 열어주세요.");
    return;
  }
  const session = ++cameraSession;
  setCamera("loading");
  try {
    const nextStream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { width: { ideal: 960 }, height: { ideal: 540 }, facingMode: "user" },
    });
    if (session !== cameraSession) {
      nextStream.getTracks().forEach((track) => track.stop());
      return;
    }
    stream = nextStream;
    stream.getVideoTracks()[0].addEventListener("ended", () => {
      if (session === cameraSession) {
        stopCamera();
        setNotice("카메라 연결이 끊겼어요. 카메라를 다시 시작해보세요.");
      }
    }, { once: true });
    webcam.srcObject = stream;
    await webcam.play();
    if (session !== cameraSession) return;
    const tasks = await loadTasks();
    if (session !== cameraSession) return;
    lastCameraFrame = -1;
    lastSegmentation = -Infinity;
    lastHandCheck = -Infinity;
    setCamera("on");
    animation = requestAnimationFrame((now) => cameraLoop(tasks, now));
  } catch (error) {
    if (session !== cameraSession) return;
    console.error(error);
    const message = cameraError(error, Boolean(stream));
    stopCamera();
    setNotice(message);
  }
}

byId("camera-toggle").addEventListener("click", () => camera === "off" ? startCamera() : stopCamera());
byId("preview").addEventListener("click", playScene);
byId("reset").addEventListener("click", resetScene);
byId("fullscreen").hidden = !document.fullscreenEnabled;
byId("fullscreen").addEventListener("click", async () => {
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    else await stage.requestFullscreen();
  } catch (error) {
    console.error(error);
    setNotice("이 브라우저에서는 전체 화면을 열 수 없어요.");
  }
});
threshold.addEventListener("input", () => {
  byId("threshold-value").value = threshold.value + "%";
  resetGesture();
  updateGuide();
});
showLandmarks.addEventListener("change", drawHands);
sound.addEventListener("change", () => { ocean.muted = !sound.checked; });
ocean.addEventListener("loadedmetadata", updateTime);
ocean.addEventListener("timeupdate", updateTime);
ocean.addEventListener("ended", () => {
  if (scene !== "playing") return;
  openedAt = performance.now();
  setScene("opened");
});
window.addEventListener("pagehide", stopCamera);
new ResizeObserver(updateGuide).observe(stage);
