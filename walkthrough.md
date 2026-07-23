# ViTTA Dockerization — Walkthrough

## Goal

Dockerize the ViTTA web app so a non-technical organization can run it with a single command (`docker compose up`). GPU-enabled image with model weights baked in.

---

## Files Created

All files are in `C:\Users\ayush\Downloads\ViTTA\`:

### 1. [Dockerfile](file:///c:/Users/ayush/Downloads/ViTTA/Dockerfile)
- **Multi-stage build** for a leaner final image
- **Stage 1 (builder)**: Uses `nvidia/cuda:12.4.1-runtime-ubuntu22.04`, installs Python 3.10, creates a virtualenv, installs PyTorch 2.6.0+cu124 from the PyTorch wheel index, then `pip install .` for the rest of the project deps (FastAPI, Ultralytics, OpenCV, etc.)
- **Stage 2 (runtime)**: Same CUDA base, copies the pre-built venv from stage 1, installs only runtime system deps (`libgl1`, `libglib2.0-0` for OpenCV), copies app code + model weights
- **Baked-in model weights**: `IDD_YOLO/best.pt` (~84 MB) and `yolo26n.pt` (~5 MB)
- **Exposes** port 8000, runs `uvicorn web_app:app --host 0.0.0.0 --port 8000`
- Includes a health check endpoint

### 2. [docker-compose.yml](file:///c:/Users/ayush/Downloads/ViTTA/docker-compose.yml)
- Single service `vitta` that builds from the Dockerfile
- **GPU reservation**: Requests 1 NVIDIA GPU via `deploy.resources.reservations.devices`
- **Named volume** `vitta-data` mounted at `/app/output` for persistent results
- **Port mapping**: `8000:8000`
- `restart: unless-stopped` for auto-restart on crash/reboot
- Sets `NVIDIA_VISIBLE_DEVICES=all` and `NVIDIA_DRIVER_CAPABILITIES=compute,utility,video`

### 3. [.dockerignore](file:///c:/Users/ayush/Downloads/ViTTA/.dockerignore)
- Excludes: `.venv/`, `.git/`, `__pycache__/`, `*.egg-info/`, `output/`, `frames/`, `runs/`, `.benchmarks/`, `.pytest_cache/`, IDE files, OS files, `uv.lock`, `*.pdf`, test files, and Docker config files themselves
- **Does NOT exclude** `.pt` files (model weights need to be in the build context)

### 4. [DOCKER_README.md](file:///c:/Users/ayush/Downloads/ViTTA/DOCKER_README.md)
- Step-by-step guide for non-technical users
- Covers: NVIDIA driver installation, Docker Desktop installation (with WSL2), getting the ViTTA files, running `docker compose up`, using the app, stopping, restarting, and troubleshooting
- Troubleshooting sections for: port conflicts, Linux containers mode, NVIDIA driver issues, slow processing, app not loading

---

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Base image | `nvidia/cuda:12.4.1-runtime-ubuntu22.04` | Matches `cu124` in pyproject.toml |
| Multi-stage build | Yes | Keeps final image smaller (no build tools) |
| Model weights | Baked into image | Zero-config for non-technical users |
| PyTorch install | Separate layer before `pip install .` | Large download cached independently |
| Output persistence | Named Docker volume | Survives container restarts |

---

## Current Status

- ✅ All 4 files created and verified
- ❌ Docker build not yet tested (Docker Desktop engine was returning 500 errors)
- ⏳ User restarting PC to fix Docker Desktop

---

## Next Steps (After PC Restart)

1. **Ensure Docker Desktop is running** (green "Engine running" indicator)
2. **Build the image**:
   ```powershell
   cd C:\Users\ayush\Downloads\ViTTA
   docker compose build
   ```
   Expected time: 10–20 minutes (downloads CUDA base ~3.5 GB + PyTorch wheels ~2.5 GB)
3. **Run the container**:
   ```powershell
   docker compose up
   ```
4. **Test**: Open `http://localhost:8000`, upload a test video, verify processing works
5. If build fails, check error output and troubleshoot

---

## Key Project Context

- **Project**: ViTTA (Video-based Traffic Trajectory extraction and Analysis)
- **Location**: `C:\Users\ayush\Downloads\ViTTA`
- **Web app entry**: `web_app.py` (FastAPI app)
- **Package**: `vitta/` (tracking, export, analysis modules)
- **Templates**: `templates/index.html`, **Static**: `static/` (app.js, styles.css, logo.png)
- **Models**: `IDD_YOLO/best.pt` (YOLOv8-Large, 84 MB), `yolo26n.pt` (5 MB)
- **Python**: 3.10+, managed with `uv`
- **GPU**: CUDA 12.4 (PyTorch cu124)
