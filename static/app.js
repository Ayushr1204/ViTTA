/* ══════════════════════════════════════════════════════════════════
   ViTTA — Frontend Controller with ROI Selection
   ══════════════════════════════════════════════════════════════════ */

let currentJobId = null;
let pollingTimer = null;
let prevValues = {};
let uploadToken = null;
let roiPoints = [];
let roiImage = null;          // Image object for the first frame
let roiImageWidth = 0;        // actual video frame width
let roiImageHeight = 0;       // actual video frame height

// ── DOM Helpers ──────────────────────────────────────────────────

function $(id) {
  return document.getElementById(id);
}

function setStatus(text) {
  $("statusText").textContent = text;
}

function setProgress(ratio) {
  const pct = Math.max(0, Math.min(100, ratio * 100));
  $("progressBar").style.width = `${pct}%`;
  $("progressPct").textContent = `${Math.round(pct)}%`;
}

function setMetric(id, value) {
  const el = $(id);
  const numVal = Number(value);
  const prev = prevValues[id] || 0;
  el.textContent = numVal.toLocaleString();
  if (numVal !== prev) {
    el.classList.remove("updated");
    void el.offsetWidth;
    el.classList.add("updated");
  }
  prevValues[id] = numVal;
}

function setDownloads(csvUrl, videoUrl) {
  const csvLink = $("csvLink");
  const videoLink = $("videoLink");
  if (csvUrl) { csvLink.href = csvUrl; csvLink.classList.remove("disabled"); }
  if (videoUrl) { videoLink.href = videoUrl; videoLink.classList.remove("disabled"); }
}

function resetDownloads() {
  const csvLink = $("csvLink");
  const videoLink = $("videoLink");
  csvLink.href = "#"; videoLink.href = "#";
  csvLink.classList.add("disabled"); videoLink.classList.add("disabled");
}

// ── Drag & Drop ─────────────────────────────────────────────────

const dropZone = $("dropZone");
const fileInput = $("videoFile");
const fileNameDisplay = $("fileNameDisplay");

dropZone.addEventListener("click", () => fileInput.click());

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("drag-over");
});

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  if (e.dataTransfer.files.length > 0) {
    fileInput.files = e.dataTransfer.files;
    showFileName(e.dataTransfer.files[0]);
  }
});

fileInput.addEventListener("change", () => {
  if (fileInput.files.length > 0) {
    showFileName(fileInput.files[0]);
  }
});

function showFileName(file) {
  const sizeMB = (file.size / (1024 * 1024)).toFixed(1);
  fileNameDisplay.textContent = `📎 ${file.name} (${sizeMB} MB)`;
}

// ══════════════════════════════════════════════════════════════════
// ROI Selection — Canvas-based quadrilateral
// ══════════════════════════════════════════════════════════════════

const roiCanvas = $("roiCanvas");
const roiCtx = roiCanvas.getContext("2d");

function showRoiSection(imageSrc, w, h) {
  roiPoints = [];
  roiImageWidth = w;
  roiImageHeight = h;
  uploadToken = uploadToken; // already set

  roiImage = new Image();
  roiImage.onload = () => {
    // Size canvas to fit within container, preserving aspect ratio
    const maxW = Math.min(860, window.innerWidth - 80);
    const scale = maxW / w;
    const canvasW = Math.round(w * scale);
    const canvasH = Math.round(h * scale);
    roiCanvas.width = canvasW;
    roiCanvas.height = canvasH;
    drawRoi();
  };
  roiImage.src = imageSrc;

  $("roiSection").classList.remove("hidden");
  $("heroSection").style.display = "none";
  $("roiHint").textContent = "Click on the image to place 4 corner points of your ROI quadrilateral.";
}

function drawRoi() {
  const cw = roiCanvas.width;
  const ch = roiCanvas.height;
  roiCtx.clearRect(0, 0, cw, ch);

  // Draw image
  roiCtx.drawImage(roiImage, 0, 0, cw, ch);

  // Semi-transparent overlay outside ROI
  if (roiPoints.length >= 3) {
    // Draw filled polygon with translucent highlight
    roiCtx.save();
    roiCtx.fillStyle = "rgba(53, 103, 255, 0.15)";
    roiCtx.beginPath();
    const scaleX = cw / roiImageWidth;
    const scaleY = ch / roiImageHeight;
    roiCtx.moveTo(roiPoints[0][0] * scaleX, roiPoints[0][1] * scaleY);
    for (let i = 1; i < roiPoints.length; i++) {
      roiCtx.lineTo(roiPoints[i][0] * scaleX, roiPoints[i][1] * scaleY);
    }
    roiCtx.closePath();
    roiCtx.fill();
    roiCtx.restore();
  }

  // Draw polygon outline
  if (roiPoints.length >= 2) {
    const scaleX = cw / roiImageWidth;
    const scaleY = ch / roiImageHeight;
    roiCtx.save();
    roiCtx.strokeStyle = "#00FFFF";
    roiCtx.lineWidth = 2;
    roiCtx.setLineDash([]);
    roiCtx.beginPath();
    roiCtx.moveTo(roiPoints[0][0] * scaleX, roiPoints[0][1] * scaleY);
    for (let i = 1; i < roiPoints.length; i++) {
      roiCtx.lineTo(roiPoints[i][0] * scaleX, roiPoints[i][1] * scaleY);
    }
    if (roiPoints.length >= 3) {
      roiCtx.closePath();
    }
    roiCtx.stroke();
    roiCtx.restore();
  }

  // Draw points
  const scaleX = cw / roiImageWidth;
  const scaleY = ch / roiImageHeight;
  roiPoints.forEach((pt, i) => {
    const x = pt[0] * scaleX;
    const y = pt[1] * scaleY;

    // Outer ring
    roiCtx.beginPath();
    roiCtx.arc(x, y, 8, 0, Math.PI * 2);
    roiCtx.fillStyle = "rgba(0, 0, 0, 0.5)";
    roiCtx.fill();
    roiCtx.strokeStyle = "#fff";
    roiCtx.lineWidth = 2;
    roiCtx.stroke();

    // Inner dot
    roiCtx.beginPath();
    roiCtx.arc(x, y, 4, 0, Math.PI * 2);
    roiCtx.fillStyle = "#00FFFF";
    roiCtx.fill();

    // Point number
    roiCtx.fillStyle = "#fff";
    roiCtx.font = "bold 11px Inter, sans-serif";
    roiCtx.fillText(`${i + 1}`, x + 12, y + 4);
  });

  // Update hint
  const remaining = 4 - roiPoints.length;
  if (remaining > 0) {
    $("roiHint").textContent = `Click ${remaining} more point${remaining > 1 ? "s" : ""} to complete the quadrilateral.`;
  } else {
    $("roiHint").textContent = "✓ ROI defined! Click 'Confirm & Run' to start, or reset to try again.";
  }
}

roiCanvas.addEventListener("click", (e) => {
  if (roiPoints.length >= 4) return; // max 4 points

  const rect = roiCanvas.getBoundingClientRect();
  const canvasX = e.clientX - rect.left;
  const canvasY = e.clientY - rect.top;

  // Convert canvas coords to actual image coords
  const imgX = Math.round((canvasX / roiCanvas.width) * roiImageWidth);
  const imgY = Math.round((canvasY / roiCanvas.height) * roiImageHeight);

  roiPoints.push([imgX, imgY]);
  drawRoi();
});

$("roiResetBtn").addEventListener("click", () => {
  roiPoints = [];
  drawRoi();
});

$("roiSkipBtn").addEventListener("click", () => {
  roiPoints = []; // empty = full frame
  startJobWithRoi();
});

$("roiConfirmBtn").addEventListener("click", () => {
  if (roiPoints.length < 3) {
    $("roiHint").textContent = "⚠ Please place at least 3 points to define a region.";
    return;
  }
  startJobWithRoi();
});

// ══════════════════════════════════════════════════════════════════
// Job Management — Two-step flow
// ══════════════════════════════════════════════════════════════════

function uploadForRoi() {
  const file = fileInput.files[0];
  if (!file) {
    setStatus("Please upload a video file first.");
    return;
  }

  // Show upload progress bar
  const progressEl = $("uploadProgress");
  const fillEl = $("uploadProgressFill");
  const pctEl = $("uploadProgressPct");
  const nameEl = $("uploadFileName");
  const sizeEl = $("uploadFileSize");

  const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
  const displayName = file.name.length > 30 ? file.name.substring(0, 27) + " ..." : file.name;
  nameEl.textContent = displayName;
  sizeEl.textContent = `${sizeMB} MB`;
  fillEl.style.width = "0%";
  pctEl.textContent = "Progress: 0%";
  progressEl.classList.remove("hidden");

  setStatus("Uploading video...");

  const formData = new FormData();
  formData.append("video", file);

  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/upload-for-roi", true);

  xhr.upload.addEventListener("progress", (e) => {
    if (e.lengthComputable) {
      const pct = ((e.loaded / e.total) * 100).toFixed(1);
      fillEl.style.width = `${pct}%`;
      pctEl.textContent = `Progress: ${pct}%`;
    }
  });

  xhr.addEventListener("load", () => {
    progressEl.classList.add("hidden");
    try {
      const data = JSON.parse(xhr.responseText);
      if (xhr.status >= 200 && xhr.status < 300) {
        uploadToken = data.token;
        setStatus("Select your ROI on the first frame below.");
        showRoiSection(data.first_frame, data.width, data.height);
      } else {
        setStatus(data.detail || "Failed to upload video.");
      }
    } catch (err) {
      setStatus(`Upload error: ${String(err)}`);
    }
  });

  xhr.addEventListener("error", () => {
    progressEl.classList.add("hidden");
    setStatus("Upload failed. Check your connection.");
  });

  xhr.send(formData);
}

async function startJobWithRoi() {
  if (!uploadToken) {
    setStatus("No upload token. Please re-upload.");
    return;
  }

  const formData = new FormData();
  formData.append("token", uploadToken);
  formData.append("interval", $("interval").value);
  formData.append("roi_points", JSON.stringify(roiPoints));

  // Hide ROI section, show hero
  $("roiSection").classList.add("hidden");
  $("heroSection").style.display = "";

  setStatus("Starting job...");
  setProgress(0);
  resetDownloads();
  setMetric("processedVal", 0);
  setMetric("activeVal", 0);
  setMetric("uniqueVal", 0);
  setMetric("rowsVal", 0);
  $("outputHint").textContent = "Processing in progress...";

  // Hide placeholder, clear preview
  const preview = $("previewImg");
  preview.removeAttribute("src");
  preview.style.display = "none";
  $("previewPlaceholder").classList.remove("hidden");

  try {
    const response = await fetch("/api/jobs", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();

    if (!response.ok) {
      setStatus(data.detail || "Failed to create job.");
      return;
    }

    currentJobId = data.job_id;
    setStatus("Job started. Processing...");
    pollJob();
  } catch (err) {
    setStatus(`Error: ${String(err)}`);
  }
}

async function pollJob() {
  if (!currentJobId) return;

  try {
    const response = await fetch(`/api/jobs/${currentJobId}`);
    const data = await response.json();

    if (!response.ok) {
      setStatus("Error while fetching job status.");
      return;
    }

    setStatus(data.message);
    setProgress(data.progress || 0);
    setMetric("processedVal", data.processed_frames || 0);
    setMetric("activeVal", data.active_tracks || 0);
    setMetric("uniqueVal", data.unique_tracks || 0);
    setMetric("rowsVal", data.csv_rows || 0);

    // Elapsed time
    const elapsed = Number(data.elapsed_sec || 0);
    $("elapsedTime").textContent = `Elapsed: ${elapsed.toFixed(1)}s`;

    // FPS counter
    const processed = data.processed_frames || 0;
    if (elapsed > 0 && processed > 0) {
      const fps = (processed / elapsed).toFixed(1);
      $("fpsCounter").textContent = `${fps} frames/sec`;
    }

    // Live preview — every processed frame
    if (data.latest_preview_url) {
      const preview = $("previewImg");
      const newSrc = `${data.latest_preview_url}?t=${Date.now()}`;
      preview.src = newSrc;
      preview.style.display = "block";
      $("previewPlaceholder").classList.add("hidden");
    }

    if (data.status === "done") {
      const totalTime = Number(data.elapsed_sec || 0).toFixed(1);
      setStatus(
        `✓ Done in ${totalTime}s — ${data.processed_frames} frames processed`
      );
      setProgress(1);
      setDownloads(data.csv_download_url, data.video_download_url);
      $("outputHint").textContent = "Processing complete! Download your results below.";
      clearTimeout(pollingTimer);
      pollingTimer = null;
      return;
    }

    if (data.status === "failed") {
      setStatus(`✗ Failed: ${data.error || "unknown error"}`);
      $("outputHint").textContent = "Processing failed. Please try again.";
      clearTimeout(pollingTimer);
      pollingTimer = null;
      return;
    }

    pollingTimer = setTimeout(pollJob, 800);
  } catch (err) {
    setStatus(`Polling error: ${String(err)}`);
    pollingTimer = setTimeout(pollJob, 2000);
  }
}

// ── Event Binding ───────────────────────────────────────────────

$("uploadBtn").addEventListener("click", () => {
  uploadForRoi();
});
