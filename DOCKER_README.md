# ViTTA — Traffic Analysis App · Setup Guide

Welcome! This guide will walk you through setting up **ViTTA** — an AI-powered traffic analysis tool — on your Windows computer.  
No programming experience is needed. Just follow each step carefully.

---

## 🐳 Docker Hub

ViTTA is published as a pre-built Docker image on **Docker Hub** — no compilation or source code required.

> 🔗 **Docker Hub:** [https://hub.docker.com/r/cr1tikal/vitta](https://hub.docker.com/repository/docker/cr1tikal/vitta/general)

| Detail | Value |
|---|---|
| **Image** | `cr1tikal/vitta:latest` |
| **Size** | ~5 GB (compressed) |
| **Platform** | Linux/amd64 (NVIDIA GPU required) |

To pull the image manually:
```
docker pull cr1tikal/vitta:latest
```

> 💡 You don't need to pull manually — the `docker compose up` command in **Step 4** will download it automatically if it isn't already on your machine.

---

## 📋 Before You Start — What You'll Need

| Requirement | Details |
|---|---|
| **Computer** | Windows 10 or 11 (64-bit) |
| **Graphics Card (GPU)** | NVIDIA GPU — GTX 1060 or newer recommended |
| **Memory (RAM)** | At least 8 GB (16 GB recommended) |
| **Free Disk Space** | At least **15 GB** |
| **Internet Connection** | Required for the first-time download (~5 GB) |

> **Don't have an NVIDIA GPU?** ViTTA requires an NVIDIA graphics card for its AI processing. Most gaming laptops and desktops have one. You can check by searching "Device Manager" in the Start menu and looking under "Display adapters".

---

## Step 1 — Install NVIDIA GPU Drivers

> ⏩ **Already gaming or doing GPU work on this PC?** Your drivers are probably fine — skip to Step 2.

1. Go to **https://www.nvidia.com/drivers**
2. Select your GPU model and download the latest driver
3. Run the downloaded file and follow the on-screen prompts
4. **Restart your computer** after installation

**To verify:** Open PowerShell (search "PowerShell" in the Start menu) and type:
```
nvidia-smi
```
If you see a table with your GPU name and driver version, you're good! ✅

---

## Step 2 — Install Docker Desktop

Docker is the tool that runs ViTTA in an isolated environment on your computer.

1. Go to **https://www.docker.com/products/docker-desktop/**
2. Click **"Download for Windows"**
3. Run the installer
   - When asked, make sure **"Use WSL 2 instead of Hyper-V"** is checked ✅
4. **Restart your computer** when prompted
5. Open **Docker Desktop** from the Start menu
6. Wait until the bottom-left corner shows a **green** indicator that says **"Engine running"**

> 💡 Docker may ask you to create an account — you can skip this by clicking **"Continue without signing in"** if that option appears.

---

## Step 3 — Download the Setup File

You only need **one small file** to run ViTTA. This file tells Docker to download the ViTTA image from [Docker Hub](https://hub.docker.com/repository/docker/cr1tikal/vitta/general) and run it on your machine.

1. Create a new folder on your computer, for example: `C:\ViTTA`
2. Download the file linked below and save it inside that folder:

   👉 **[docker-compose.yml](https://raw.githubusercontent.com/Ayushr1204/ViTTA/main/docker-compose.hub.yml)**

   > If clicking the link opens text in your browser instead of downloading, right-click the link → **"Save link as..."** → save it as **`docker-compose.yml`** (not `.txt`) inside your `C:\ViTTA` folder.

   Alternatively, create a file named `docker-compose.yml` in your `C:\ViTTA` folder and paste the following content:

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

---

## Step 4 — Start ViTTA

1. Open **PowerShell** (search "PowerShell" in the Start menu)
2. Navigate to your ViTTA folder by typing:
   ```
   cd C:\ViTTA
   ```
3. Start the app:
   ```
   docker compose up
   ```
4. **Wait** — the first time, Docker will download the ViTTA image (~5 GB).  
   This may take **10–30 minutes** depending on your internet speed.  
   You'll see text scrolling in the window — this is normal!

5. ViTTA is ready when you see a line like:
   ```
   vitta-web  | INFO:     Uvicorn running on http://0.0.0.0:8000
   ```

---

## Step 5 — Open the App in Your Browser

1. Open **Google Chrome**, **Microsoft Edge**, or **Firefox**
2. In the address bar, type:
   ```
   http://localhost:8000
   ```
3. Press **Enter**
4. You should see the ViTTA upload page 🎉

---

## 🎬 How to Use ViTTA

1. **Upload a Video** — Click the upload area and select a traffic video file (`.mp4`)
2. **Select Region of Interest** *(optional)* — Draw a shape on the first frame to focus on a specific road area
3. **Set Calibration** *(optional)* — Enter a known real-world distance in metres for accurate speed measurement
4. **Start Processing** — Watch the live preview as ViTTA detects and tracks vehicles in real time
5. **Download Results** — Once processing is complete, download your results:
   - 📊 **CSV** — Raw tracking data (spreadsheet-compatible)
   - 🎬 **Video** — Annotated video showing detected vehicles with trails
   - 📄 **PDF Report** — Automatic traffic analysis report with charts and statistics

---

## ⏹ Stopping the App

To stop ViTTA:
- In the PowerShell window, press **`Ctrl + C`**
- Or open a new PowerShell window and run:
  ```
  cd C:\ViTTA
  docker compose down
  ```

---

## ▶ Starting Again Later

After the first download, starting ViTTA is much faster (~10 seconds):

```
cd C:\ViTTA
docker compose up
```

> You do **not** need to download anything again — Docker remembers the image.

---

## ❓ Troubleshooting

### "port is already allocated"
Another program is using port 8000. Fix:
1. Stop ViTTA: `docker compose down`
2. Open `docker-compose.yml` in Notepad
3. Change `"8000:8000"` to `"8080:8000"`
4. Save the file
5. Run `docker compose up`
6. Open **http://localhost:8080** instead

### "no matching manifest for windows"
Docker needs to use Linux mode:
- Right-click the **Docker whale icon** in the system tray (bottom-right of taskbar)
- Click **"Switch to Linux containers"**
- Try again

### "could not select device driver 'nvidia'"
Your NVIDIA drivers may not be installed:
1. Open PowerShell and run: `nvidia-smi`
2. If this command shows an error, go back to **Step 1** and install drivers
3. Also make sure Docker Desktop is updated to the latest version

### Processing is very slow
- Make sure you have an **NVIDIA GPU** (not Intel/AMD integrated graphics)
- Close other heavy apps (games, video editors, etc.)
- Try a shorter or lower-resolution video first

### App doesn't load in the browser
- Check the PowerShell window — make sure it says "Uvicorn running on http://0.0.0.0:8000"
- Try **http://127.0.0.1:8000** instead of localhost
- Check that your firewall or antivirus isn't blocking port 8000

### Need to update ViTTA?
If you're told a new version is available:
```
docker compose down
docker pull cr1tikal/vitta:latest
docker compose up
```

---

## 🆘 Need Help?

Contact the ViTTA development team for support.
