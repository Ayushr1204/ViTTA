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

// ── Calibration State ────────────────────────────────────────────
let calibPoints = [];         // [{x, y}, {x, y}]  in image coords
let calibImage = null;        // reuses roiImage
let calibPixelsPerMetre = null;

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

function setDownloads(csvUrl, videoUrl, pdfUrl) {
  const csvLink = $("csvLink");
  const videoLink = $("videoLink");
  const pdfLink = $("pdfLink");
  if (csvUrl) {
    csvLink.href = csvUrl;
    csvLink.setAttribute("download", "tracks.csv");
    csvLink.classList.remove("disabled");
  }
  if (videoUrl) {
    videoLink.href = videoUrl;
    videoLink.setAttribute("download", "tracked_output.mp4");
    videoLink.classList.remove("disabled");
  }
  if (pdfUrl) {
    pdfLink.href = pdfUrl;
    pdfLink.setAttribute("download", "vitta_report.pdf");
    pdfLink.classList.remove("disabled");
  }
}

function resetDownloads() {
  const csvLink = $("csvLink");
  const videoLink = $("videoLink");
  const pdfLink = $("pdfLink");
  [csvLink, videoLink, pdfLink].forEach(link => {
    link.href = "#";
    link.removeAttribute("download");
    link.classList.add("disabled");
  });
}

// Prevent clicks on disabled download links
document.addEventListener("click", (e) => {
  const link = e.target.closest(".btn-download.disabled");
  if (link) {
    e.preventDefault();
    e.stopPropagation();
  }
});

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

  // Scroll ROI section into view so it's not lost below the fold
  setTimeout(() => {
    $("roiSection").scrollIntoView({ behavior: "smooth", block: "start" });
  }, 100);
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
    $("roiHint").textContent = "✓ ROI defined! Click 'Confirm & Run' to start, or 'Calibrate' to set a road-length reference.";
  }
}

roiCanvas.addEventListener("click", (e) => {
  if (roiPoints.length >= 4) return; // max 4 points

  const rect = roiCanvas.getBoundingClientRect();
  const canvasX = e.clientX - rect.left;
  const canvasY = e.clientY - rect.top;

  // Convert CSS pixel coords to actual image coords using displayed size
  const imgX = Math.round((canvasX / rect.width) * roiImageWidth);
  const imgY = Math.round((canvasY / rect.height) * roiImageHeight);

  roiPoints.push([imgX, imgY]);
  drawRoi();
});

$("roiResetBtn").addEventListener("click", () => {
  roiPoints = [];
  drawRoi();
});

$("roiSkipBtn").addEventListener("click", () => {
  roiPoints = []; // empty = full frame
  calibPixelsPerMetre = null;
  startJobWithCalibration();
});

$("roiCalibrateBtn").addEventListener("click", () => {
  showCalibrationSection();
});

$("roiConfirmBtn").addEventListener("click", () => {
  if (roiPoints.length < 3) {
    $("roiHint").textContent = "⚠ Please place at least 3 points to define a region.";
    return;
  }
  calibPixelsPerMetre = null;
  startJobWithCalibration();
});

// ══════════════════════════════════════════════════════════════════
// Calibration — Two-point spatial reference
// ══════════════════════════════════════════════════════════════════

const calibCanvas = $("calibCanvas");
const calibCtx = calibCanvas.getContext("2d");

function showCalibrationSection() {
  calibPoints = [];
  calibPixelsPerMetre = null;
  calibImage = roiImage; // reuse the already-loaded first frame

  // Size canvas to fit within container, preserving aspect ratio
  const maxW = Math.min(860, window.innerWidth - 80);
  const scale = maxW / roiImageWidth;
  const canvasW = Math.round(roiImageWidth * scale);
  const canvasH = Math.round(roiImageHeight * scale);
  calibCanvas.width = canvasW;
  calibCanvas.height = canvasH;

  // Clear manual inputs
  $("calibAx").value = "";
  $("calibAy").value = "";
  $("calibBx").value = "";
  $("calibBy").value = "";
  $("calibRoadLength").value = "";
  $("calibPixelDist").textContent = "—";
  $("calibPxPerMetre").textContent = "—";
  $("calibHint").textContent = "Click two points on the video frame to define a reference distance, then enter the real-world length in metres.";

  drawCalibration();

  $("roiSection").classList.add("hidden");
  $("calibrationSection").classList.remove("hidden");
}

function drawCalibration() {
  const cw = calibCanvas.width;
  const ch = calibCanvas.height;
  calibCtx.clearRect(0, 0, cw, ch);

  // Draw the first frame
  calibCtx.drawImage(calibImage, 0, 0, cw, ch);

  const scaleX = cw / roiImageWidth;
  const scaleY = ch / roiImageHeight;

  // Draw connecting dashed line between two points
  if (calibPoints.length === 2) {
    const ax = calibPoints[0].x * scaleX;
    const ay = calibPoints[0].y * scaleY;
    const bx = calibPoints[1].x * scaleX;
    const by = calibPoints[1].y * scaleY;

    calibCtx.save();
    calibCtx.strokeStyle = "#F59E0B";
    calibCtx.lineWidth = 2;
    calibCtx.setLineDash([8, 4]);
    calibCtx.beginPath();
    calibCtx.moveTo(ax, ay);
    calibCtx.lineTo(bx, by);
    calibCtx.stroke();
    calibCtx.setLineDash([]);
    calibCtx.restore();

    // Pixel distance label at midpoint
    const midX = (ax + bx) / 2;
    const midY = (ay + by) / 2;
    const pixDist = Math.sqrt(
      Math.pow(calibPoints[1].x - calibPoints[0].x, 2) +
      Math.pow(calibPoints[1].y - calibPoints[0].y, 2)
    );
    calibCtx.save();
    calibCtx.font = "bold 13px Inter, sans-serif";
    calibCtx.fillStyle = "rgba(0,0,0,0.7)";
    const label = `${pixDist.toFixed(1)} px`;
    const tw = calibCtx.measureText(label).width;
    calibCtx.fillRect(midX - tw / 2 - 6, midY - 18, tw + 12, 22);
    calibCtx.fillStyle = "#F59E0B";
    calibCtx.fillText(label, midX - tw / 2, midY - 2);
    calibCtx.restore();
  }

  // Draw points
  calibPoints.forEach((pt, i) => {
    const x = pt.x * scaleX;
    const y = pt.y * scaleY;

    // Outer ring
    calibCtx.beginPath();
    calibCtx.arc(x, y, 9, 0, Math.PI * 2);
    calibCtx.fillStyle = "rgba(0, 0, 0, 0.5)";
    calibCtx.fill();
    calibCtx.strokeStyle = "#fff";
    calibCtx.lineWidth = 2;
    calibCtx.stroke();

    // Inner dot
    calibCtx.beginPath();
    calibCtx.arc(x, y, 5, 0, Math.PI * 2);
    calibCtx.fillStyle = "#F59E0B";
    calibCtx.fill();

    // Label
    calibCtx.fillStyle = "#fff";
    calibCtx.font = "bold 12px Inter, sans-serif";
    calibCtx.fillText(i === 0 ? "A" : "B", x + 14, y + 4);
  });
}

function updateCalibComputed() {
  if (calibPoints.length === 2) {
    const dx = calibPoints[1].x - calibPoints[0].x;
    const dy = calibPoints[1].y - calibPoints[0].y;
    const pixDist = Math.sqrt(dx * dx + dy * dy);
    $("calibPixelDist").textContent = pixDist.toFixed(1);

    const roadLen = parseFloat($("calibRoadLength").value);
    if (roadLen > 0) {
      calibPixelsPerMetre = pixDist / roadLen;
      $("calibPxPerMetre").textContent = calibPixelsPerMetre.toFixed(2);
      $("calibHint").textContent = `✓ Calibration ready: ${calibPixelsPerMetre.toFixed(2)} px/m — click 'Confirm & Run' to start processing.`;
    } else {
      calibPixelsPerMetre = null;
      $("calibPxPerMetre").textContent = "—";
    }
  } else {
    $("calibPixelDist").textContent = "—";
    $("calibPxPerMetre").textContent = "—";
    calibPixelsPerMetre = null;
  }
}

function syncInputsFromPoints() {
  if (calibPoints.length >= 1) {
    $("calibAx").value = calibPoints[0].x;
    $("calibAy").value = calibPoints[0].y;
  }
  if (calibPoints.length >= 2) {
    $("calibBx").value = calibPoints[1].x;
    $("calibBy").value = calibPoints[1].y;
  }
}

// Canvas click handler
calibCanvas.addEventListener("click", (e) => {
  if (calibPoints.length >= 2) {
    // Replace: start over
    calibPoints = [];
  }

  const rect = calibCanvas.getBoundingClientRect();
  const canvasX = e.clientX - rect.left;
  const canvasY = e.clientY - rect.top;

  // Convert CSS pixel coords to actual image coords using displayed size
  const imgX = Math.round((canvasX / rect.width) * roiImageWidth);
  const imgY = Math.round((canvasY / rect.height) * roiImageHeight);

  calibPoints.push({ x: imgX, y: imgY });
  syncInputsFromPoints();
  updateCalibComputed();
  drawCalibration();

  if (calibPoints.length === 1) {
    $("calibHint").textContent = "Click one more point to complete the reference segment.";
  }
});

// Manual coordinate input handlers
["calibAx", "calibAy", "calibBx", "calibBy"].forEach(id => {
  $(id).addEventListener("input", () => {
    const ax = parseFloat($("calibAx").value);
    const ay = parseFloat($("calibAy").value);
    const bx = parseFloat($("calibBx").value);
    const by = parseFloat($("calibBy").value);

    calibPoints = [];
    if (!isNaN(ax) && !isNaN(ay)) {
      calibPoints.push({ x: ax, y: ay });
    }
    if (!isNaN(bx) && !isNaN(by)) {
      calibPoints.push({ x: bx, y: by });
    }

    updateCalibComputed();
    drawCalibration();
  });
});

// Road length input
$("calibRoadLength").addEventListener("input", () => {
  updateCalibComputed();
});

// Buttons
$("calibResetBtn").addEventListener("click", () => {
  calibPoints = [];
  calibPixelsPerMetre = null;
  $("calibAx").value = "";
  $("calibAy").value = "";
  $("calibBx").value = "";
  $("calibBy").value = "";
  $("calibRoadLength").value = "";
  $("calibPixelDist").textContent = "—";
  $("calibPxPerMetre").textContent = "—";
  $("calibHint").textContent = "Click two points on the video frame to define a reference distance, then enter the real-world length in metres.";
  drawCalibration();
});

$("calibSkipBtn").addEventListener("click", () => {
  calibPixelsPerMetre = null;
  startJobWithCalibration();
});

$("calibConfirmBtn").addEventListener("click", () => {
  if (calibPoints.length < 2) {
    $("calibHint").textContent = "⚠ Please place 2 points to define the reference segment.";
    return;
  }
  const roadLen = parseFloat($("calibRoadLength").value);
  if (!roadLen || roadLen <= 0) {
    $("calibHint").textContent = "⚠ Please enter a valid road length in metres.";
    return;
  }
  // calibPixelsPerMetre is already computed
  startJobWithCalibration();
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

async function startJobWithCalibration() {
  if (!uploadToken) {
    setStatus("No upload token. Please re-upload.");
    return;
  }

  const formData = new FormData();
  formData.append("token", uploadToken);
  formData.append("interval", $("interval").value);
  formData.append("roi_points", JSON.stringify(roiPoints));
  const fpsVal = $("videoFps").value;
  if (fpsVal) {
    formData.append("video_fps", fpsVal);
  }
  if (calibPixelsPerMetre !== null && calibPixelsPerMetre > 0) {
    formData.append("pixels_per_metre", calibPixelsPerMetre.toString());
    // Send calibration point coordinates for perspective correction
    if (calibPoints.length >= 2) {
      formData.append("calib_points", JSON.stringify(
        calibPoints.map(p => [p.x, p.y])
      ));
    }
  }

  // Hide calibration & ROI sections, hide upload card, show New Upload button
  $("calibrationSection").classList.add("hidden");
  $("roiSection").classList.add("hidden");
  $("heroSection").style.display = "none";
  $("newUploadBtn").classList.remove("hidden");

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
      setDownloads(data.csv_download_url, data.video_download_url, data.pdf_download_url);
      $("outputHint").textContent = "Processing complete! Download your results below.";
      clearTimeout(pollingTimer);
      pollingTimer = null;
      fetchAnalytics();
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

// ══════════════════════════════════════════════════════════════════
// New Upload button — reload page to start fresh
// ══════════════════════════════════════════════════════════════════

$("newUploadBtn").addEventListener("click", () => {
  window.location.reload();
});

// ══════════════════════════════════════════════════════════════════
// Analytics — Plotly.js dashboard
// ══════════════════════════════════════════════════════════════════

const PLOTLY_COLORS = [
  "#6366F1","#10B981","#F59E0B","#EF4444",
  "#06B6D4","#EC4899","#84CC16","#8B5CF6",
  "#3B82F6","#D946EF","#14B8A6","#F97316",
];

const BEHAVIOUR_COLORS = {
  "Disciplined":"#10B981","Speeding":"#EF4444","Slow":"#F59E0B",
  "Erratic":"#8B5CF6","Aggressive Braking":"#DC2626",
  "Tailgating":"#F97316","Lane Weaving":"#06B6D4",
  "Stopped/Idling":"#6B7280",
};

const DIRECTION_COLORS = {
  "Northbound":"#3B82F6","Southbound":"#EF4444",
  "Eastbound":"#F59E0B","Westbound":"#8B5CF6","Stationary":"#6B7280",
};

const PLOTLY_LAYOUT_BASE = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: { family: "'Plus Jakarta Sans', sans-serif", color: "#4B5563", size: 12 },
  margin: { l: 50, r: 20, t: 10, b: 40 },
  hoverlabel: { bgcolor: "#1F2937", font: { color: "#F9FAFB", family: "'Plus Jakarta Sans', sans-serif" } },
};

const PLOTLY_CONFIG = {
  displayModeBar: "hover",
  modeBarButtonsToRemove: ["lasso2d","select2d"],
  responsive: true,
};

async function fetchAnalytics() {
  if (!currentJobId) return;
  try {
    const resp = await fetch(`/api/jobs/${currentJobId}/analytics`);
    if (!resp.ok) return;
    const data = await resp.json();
    if (data.error) return;
    renderAnalytics(data);
  } catch (e) {
    console.warn("Analytics fetch failed:", e);
  }
}

function renderAnalytics(data) {
  const s = data.summary;
  const u = s.speed_unit || "px/s";
  $("sumTotalVehicles").textContent = s.total_vehicles;
  $("sumAvgSpeed").textContent = `${s.avg_speed} ${u}`;
  $("sumPeakSpeed").textContent = `${s.peak_speed} ${u}`;
  $("sumAvgSpeedLabel").textContent = `Avg Speed (${u})`;
  $("sumPeakSpeedLabel").textContent = `Peak Speed (${u})`;
  $("sumAvgDuration").textContent = `${s.avg_duration_sec}s`;
  $("sumDensity").textContent = s.vehicles_per_min;
  if ($("sumAvgHeadway")) $("sumAvgHeadway").textContent = `${s.avg_headway_sec || 0}s`;

  const LB = {...PLOTLY_LAYOUT_BASE, autosize: true};
  const CF = {...PLOTLY_CONFIG, responsive: true};
  const axStyle = {tickfont:{size:10,color:"#9CA3AF"},gridcolor:"rgba(0,0,0,0.04)",
    title:{font:{size:11,color:"#6B7280"}}};

  // 1. Vehicle Composition (Donut)
  Plotly.newPlot("chartClassDist",[{
    type:"pie",labels:data.class_distribution.labels,values:data.class_distribution.values,
    hole:0.55,marker:{colors:PLOTLY_COLORS,line:{color:"#fff",width:2.5}},
    textinfo:"label+percent",textposition:"outside",
    textfont:{size:11,family:"'Plus Jakarta Sans',sans-serif"},
    hoverinfo:"label+value+percent",pull:0.04,
    insidetextorientation:"radial",
  }],{...LB,margin:{l:10,r:10,t:10,b:10},showlegend:true,
    legend:{orientation:"h",y:-0.08,x:0.5,xanchor:"center",font:{size:10,color:"#6B7280"}}
  },CF);

  // 2. Speed Distribution (Histogram with 1 m/s bins)
  const sLabels = data.speed_histogram.labels;
  const sValues = data.speed_histogram.values;
  // Use bin midpoints for bar x positions
  const sMids = sLabels.map(label => {
    const parts = label.split("-");
    if (parts.length === 2) {
      return ((parseFloat(parts[0]) + parseFloat(parts[1])) / 2).toFixed(1);
    }
    return label;
  });
  Plotly.newPlot("chartSpeedHist",[{
    type:"bar",x:sMids,y:sValues,
    marker:{color:"#15803D",opacity:0.85,line:{color:"#0F5132",width:1}},
    hovertemplate:"<b>%{x} "+u+"</b><br>%{y} vehicles<extra></extra>",
  }],{...LB,
    margin:{l:50,r:20,t:10,b:50},
    xaxis:{title:{text:`Speed (${u})`},...axStyle,dtick:1},
    yaxis:{title:{text:"Vehicle Count"},...axStyle,rangemode:"tozero"},
    bargap:0.05,
  },CF);

  // 3. Direction (Polar) — North pointing UP
  const dL=data.direction_breakdown.labels, dV=data.direction_breakdown.values;
  const dC=dL.map(l=>DIRECTION_COLORS[l]||"#6B7280");
  // Map directions to compass bearings: North=0°(top), East=90°, South=180°(bottom), West=270°
  const dMap={"Northbound":0,"Eastbound":90,"Southbound":180,"Westbound":270,"Stationary":315};
  Plotly.newPlot("chartDirection",[{
    type:"barpolar",r:dV,theta:dL.map(l=>dMap[l]!==undefined?dMap[l]:0),
    text:dL,marker:{color:dC,opacity:0.85,line:{color:"#fff",width:1.5}},
    hovertemplate:"<b>%{text}</b><br>%{r} vehicles<extra></extra>",
    width:dV.map(()=>60),
  }],{...LB,margin:{l:50,r:50,t:40,b:40},
    polar:{bgcolor:"rgba(0,0,0,0)",
      radialaxis:{visible:true,tickfont:{size:9,color:"#9CA3AF"},gridcolor:"rgba(0,0,0,0.05)"},
      angularaxis:{tickfont:{size:10,color:"#4B5563"},direction:"clockwise",rotation:90,gridcolor:"rgba(0,0,0,0.05)",
        tickvals:[0,45,90,135,180,225,270,315],
        ticktext:["N","NE","E","SE","S","SW","W","NW"],
      }
    },showlegend:false,
  },CF);

  // 4. Class Speed Comparison (Horizontal Bar Chart)
  if (data.class_speed_comparison) {
    const csd = data.class_speed_comparison;
    const classNames = Object.keys(csd).sort();
    const avgSpeeds = classNames.map(cls => csd[cls].avg);
    const minSpeeds = classNames.map(cls => csd[cls].min);
    const maxSpeeds = classNames.map(cls => csd[cls].max);
    const counts = classNames.map(cls => csd[cls].count);
    Plotly.newPlot("chartClassSpeed",[
      {type:"bar",y:classNames,x:avgSpeeds,orientation:"h",name:"Avg Speed",
        marker:{color:classNames.map((_,i)=>PLOTLY_COLORS[i%PLOTLY_COLORS.length]),opacity:0.85,
          line:{color:"#fff",width:1}},
        text:avgSpeeds.map((v,i)=>`${v.toFixed(1)} ${u} (n=${counts[i]})`),
        textposition:"outside",textfont:{size:10},
        hovertemplate:classNames.map((cls,i)=>
          `<b>${cls}</b><br>Avg: ${avgSpeeds[i].toFixed(1)} ${u}<br>Min: ${minSpeeds[i].toFixed(1)}<br>Max: ${maxSpeeds[i].toFixed(1)}<br>Count: ${counts[i]}<extra></extra>`
        ),
      }
    ],{...LB,
      margin:{l:70,r:60,t:10,b:50},
      xaxis:{title:{text:`Avg Speed (${u})`},...axStyle,rangemode:"tozero"},
      yaxis:{...axStyle,automargin:true,type:"category",categoryorder:"array",categoryarray:classNames},
      showlegend:false,
    },CF);
  }

  // 5. Behaviour Distribution (Horizontal Bar)
  if (data.behaviour_distribution) {
    const bL=data.behaviour_distribution.labels, bV=data.behaviour_distribution.values;
    const total=bV.reduce((a,b)=>a+b,0);
    // Sort by count descending for clear visual hierarchy
    const sorted = bL.map((l,i)=>({label:l,value:bV[i]})).sort((a,b)=>a.value-b.value);
    const sLabels = sorted.map(s=>s.label);
    const sValues = sorted.map(s=>s.value);
    const sColors = sLabels.map(l=>BEHAVIOUR_COLORS[l]||"#6B7280");
    const sText = sorted.map(s=>{
      const pct = total > 0 ? ((s.value / total) * 100).toFixed(0) : 0;
      return `${s.value} (${pct}%)`;
    });
    Plotly.newPlot("chartBehaviour",[{
      type:"bar",y:sLabels,x:sValues,orientation:"h",
      marker:{color:sColors,opacity:0.9,line:{color:"#fff",width:1}},
      text:sText,textposition:"outside",textfont:{size:11,color:"#4B5563"},
      hovertemplate:"<b>%{y}</b><br>%{x} vehicles<extra></extra>",
    }],{...LB,
      margin:{l:110,r:50,t:8,b:40},
      xaxis:{title:{text:"Vehicle Count"},...axStyle,rangemode:"tozero"},
      yaxis:{...axStyle,automargin:true,type:"category",categoryorder:"array",categoryarray:sLabels},
      showlegend:false,
    },CF);
  }

  // 6. Speed vs Headway (Scatter) — with P95 outlier filtering
  if (data.speed_vs_headway && data.speed_vs_headway.speeds.length > 0) {
    const sv=data.speed_vs_headway;
    // Filter outliers: cap axes at P95 * 1.3 for clean view (user can zoom out)
    const sortedHw = [...sv.headways].sort((a,b)=>a-b);
    const sortedSp = [...sv.speeds].sort((a,b)=>a-b);
    const p95Hw = sortedHw[Math.floor(sortedHw.length * 0.95)] || sortedHw[sortedHw.length-1];
    const p95Sp = sortedSp[Math.floor(sortedSp.length * 0.95)] || sortedSp[sortedSp.length-1];
    const xMax = Math.max(p95Hw * 1.3, 2);
    const yMax = p95Sp * 1.3;
    const uniqueClasses=[...new Set(sv.classes)];
    const traces = uniqueClasses.map((cls,i)=>{
      const idx=sv.classes.map((c,j)=>c===cls?j:-1).filter(j=>j>=0);
      return {
        type:"scattergl",mode:"markers",name:cls,
        x:idx.map(j=>sv.headways[j]),y:idx.map(j=>sv.speeds[j]),
        marker:{size:6,color:PLOTLY_COLORS[i%PLOTLY_COLORS.length],opacity:0.65,
          line:{color:"#fff",width:0.5}},
        hovertemplate:`<b>${cls}</b><br>Headway: %{x:.1f}s<br>Speed: %{y:.1f} ${u}<extra></extra>`,
      };
    });
    Plotly.newPlot("chartSpeedHeadway",traces,{...LB,
      margin:{l:55,r:20,t:10,b:50},
      xaxis:{title:{text:"Time Headway (s)"},...axStyle,range:[0,xMax]},
      yaxis:{title:{text:`Speed (${u})`},...axStyle,range:[0,yMax]},
      legend:{font:{size:10,color:"#6B7280"},orientation:"h",y:1.05,x:0.5,xanchor:"center"},
    },CF);
  }

  // 7. Traffic Density (Area) — green theme
  let tdL=data.traffic_density.labels, tdV=data.traffic_density.values;
  if(tdL.length>200){const step=Math.ceil(tdL.length/200);tdL=tdL.filter((_,i)=>i%step===0);tdV=tdV.filter((_,i)=>i%step===0);}
  Plotly.newPlot("chartDensity",[{
    type:"scatter",mode:"lines",x:tdL,y:tdV,
    fill:"tozeroy",fillcolor:"rgba(21,128,61,0.08)",
    line:{color:"#15803D",width:2.5,shape:"spline"},
    hovertemplate:"<b>Time: %{x}s</b><br>Active Vehicles: %{y}<extra></extra>",
  }],{...LB,
    margin:{l:50,r:20,t:10,b:50},
    xaxis:{title:{text:"Time (seconds)"},...axStyle,nticks:15},
    yaxis:{title:{text:"Active Vehicles"},...axStyle,rangemode:"tozero"},
  },CF);

  // 8. Congestion chart removed

  // 9. Headway Distribution (Histogram)
  if (data.speed_vs_headway && data.speed_vs_headway.headways) {
    const hw = data.speed_vs_headway.headways;
    Plotly.newPlot("chartHeadway",[
      {type:"histogram",x:hw,xbins:{start:0,size:1},
       marker:{color:"#15803D",opacity:0.85,line:{color:"#0F5132",width:1}},
       hovertemplate:"Headway: %{x}s<br>Count: %{y}<extra></extra>"}
    ],{...LB,
      margin:{l:50,r:20,t:10,b:50},
      xaxis:{title:{text:"Time Headway (s)"},...axStyle},
      yaxis:{title:{text:"Vehicle Count"},...axStyle,rangemode:"tozero"}
    },CF);
  }

  $("analyticsSection").classList.remove("hidden");
  $("analyticsSection").scrollIntoView({behavior:"smooth",block:"start"});
}

