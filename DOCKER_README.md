# ViTTA — Get Started in 3 Steps

**ViTTA** is an AI-powered traffic video analysis tool. It detects, tracks, and counts vehicles from any traffic video — right from your browser.

The easiest way to run ViTTA is through **Docker Hub**. No code to download, no dependencies to install — just one command.

> 🐳 **Docker Hub:** [hub.docker.com/r/cr1tikal/vitta](https://hub.docker.com/repository/docker/cr1tikal/vitta/general)

---

## Requirements

| Requirement | Details |
|---|---|
| **OS** | Windows 10/11 (64-bit), Linux, or macOS |
| **GPU** | NVIDIA GPU (GTX 1060 or newer recommended) |
| **RAM** | 8 GB minimum (16 GB recommended) |
| **Disk** | ~15 GB free space |
| **Internet** | Required for the first-time download (~5 GB) |

> **How to check if you have an NVIDIA GPU:** Search "Device Manager" in the Start menu → expand "Display adapters". If you see an NVIDIA card listed, you're good.

---

## Step 1 — Install Docker Desktop

1. Download **Docker Desktop** from [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/)
2. Run the installer
   - Make sure **"Use WSL 2 instead of Hyper-V"** is checked ✅
3. **Restart your computer**
4. Open **Docker Desktop** and wait until it shows **"Engine running"** (green indicator, bottom-left)

> 💡 Docker may ask you to create an account — click **"Continue without signing in"** to skip.

### NVIDIA GPU Drivers

> ⏩ **Already gaming on this PC?** Your GPU drivers are almost certainly fine — skip to Step 2.

If you've never installed GPU drivers:
1. Go to [nvidia.com/drivers](https://www.nvidia.com/drivers)
2. Download and install the driver for your GPU model
3. Restart your computer

Verify it works — open PowerShell and run:
```
nvidia-smi
```
If you see a table with your GPU name, you're ready ✅

---

## Step 2 — Run ViTTA

Open **PowerShell** and run this single command:

```
docker run -d --name vitta -p 8000:8000 --gpus all cr1tikal/vitta:latest
```

That's it. Docker will automatically download the image from Docker Hub (~5 GB) and start the app.

> ⏳ **First time only:** The download takes 10–30 minutes depending on your internet speed. Subsequent starts are instant.

To check if it's running:
```
docker logs vitta
```
Look for this line:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

## Step 3 — Open the App

Open your browser and go to:

### 👉 [http://localhost:8000](http://localhost:8000)

You should see the ViTTA upload page. **You're done!** 🎉

---

## How to Use ViTTA

1. **Upload** a traffic video (`.mp4`)
2. **Select Region of Interest** *(optional)* — draw a shape to focus on a specific road area
3. **Calibrate** *(optional)* — enter a known real-world distance for accurate speed measurement
4. **Process** — watch the live preview as ViTTA detects and tracks vehicles
5. **Download Results:**
   - 📊 **CSV** — raw tracking data
   - 🎬 **Video** — annotated video with vehicle trails
   - 📄 **PDF Report** — charts and traffic statistics

---

## Managing ViTTA

| Action | Command |
|---|---|
| **Stop** the app | `docker stop vitta` |
| **Start** it again | `docker start vitta` |
| **View logs** | `docker logs vitta` |
| **Remove** the container | `docker rm -f vitta` |
| **Update** to latest version | `docker rm -f vitta` then re-run the Step 2 command |

---

## Alternative: Using Docker Compose

If you prefer Docker Compose, create a file called `docker-compose.yml` with this content:

```yaml
services:
  vitta:
    image: cr1tikal/vitta:latest
    container_name: vitta-web
    ports:
      - "8000:8000"
    volumes:
      - vitta-data:/app/output
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

volumes:
  vitta-data:
```

Then run:
```
docker compose up
```

> The Compose method also creates a persistent volume (`vitta-data`) so your output files are saved between restarts.

---

## Troubleshooting

### "port is already allocated"
Port 8000 is in use. Use a different port:
```
docker rm -f vitta
docker run -d --name vitta -p 8080:8000 --gpus all cr1tikal/vitta:latest
```
Then open **http://localhost:8080** instead.

### "could not select device driver 'nvidia'"
NVIDIA drivers aren't installed or Docker can't see them:
1. Run `nvidia-smi` in PowerShell — if it errors, install drivers (see Step 1)
2. Make sure Docker Desktop is updated to the latest version

### "no matching manifest for windows"
Docker is in Windows container mode. Fix:
- Right-click the Docker whale icon in the system tray → **"Switch to Linux containers"**

### App doesn't load in the browser
- Run `docker logs vitta` — check for errors
- Try **http://127.0.0.1:8000** instead of localhost
- Make sure your firewall isn't blocking port 8000

### Processing is slow
- Confirm you have an NVIDIA GPU (not Intel/AMD integrated graphics)
- Close other GPU-heavy apps (games, video editors)
- Try a shorter or lower-resolution video

---

## Need Help?

- 🐳 **Docker Hub:** [hub.docker.com/r/cr1tikal/vitta](https://hub.docker.com/repository/docker/cr1tikal/vitta/general)
- 📦 **GitHub:** [github.com/Ayushr1204/ViTTA](https://github.com/Ayushr1204/ViTTA)
