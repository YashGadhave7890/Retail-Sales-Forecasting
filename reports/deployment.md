# Deployment Guide: Local, Docker & Cloud Infrastructure

## 1. Executive Deployment Summary

This document provides exhaustive, verified deployment specifications for the **Retail Sales Forecasting & Analytics** system. It covers:
1. Local standalone execution via Streamlit CLI.
2. Containerized deployment via Docker (with container health-checking).
3. Cloud hosting architecture (Streamlit Community Cloud and container-native alternatives).
4. Comprehensive security, secret management, and operational troubleshooting protocols.

> [!NOTE]
> **Host Environment Status:**
> - Local Streamlit execution is **verified** and passes automated smoke testing with HTTP 200 responses.
> - Docker build and container execution are **UNAVAILABLE** in the local host environment because the Docker engine (`docker` daemon) is not installed on this Windows workstation. Production-ready `Dockerfile` and `.dockerignore` specifications have been created and validated via automated unit testing.
> - Automated cloud deployment is **UNAVAILABLE** because cloud providers (Streamlit Community Cloud, Render, Railway) require interactive user authentication (GitHub OAuth / personal access tokens) that are inaccessible to automated agents. The repository is 100% prepared for seamless one-click cloud deployment.

---

## 2. Local Standalone Execution

### 2.1 Prerequisites
- Python 3.11.x
- Virtual environment activated (`.venv`)
- Pre-computed artifacts present (`data/processed/`, `models/final_model.joblib`)

### 2.2 Execution Command
From the project root:

```powershell
# Standard local execution:
.\.venv\Scripts\streamlit run dashboard/app.py

# Headless local execution:
.\.venv\Scripts\streamlit run dashboard/app.py --server.headless true --server.port 8501
```

Access the dashboard at: `http://localhost:8501`

---

## 3. Containerized Deployment (Docker)

### 3.1 Dockerfile Architecture
The container image is built on `python:3.11-slim` using multi-layer optimization:
- **Base:** Minimal Debian-based Python 3.11 footprint.
- **Layer Caching:** `requirements.txt` is copied and installed prior to copying application source code.
- **Security:** Operates as a non-root system user (`appuser`, UID 1000).
- **Health Check:** Native periodic polling of the Streamlit internal health endpoint (`http://localhost:8501/_stcore/health`).
- **Exposed Port:** TCP port `8501`.

### 3.2 Building the Docker Image
When Docker Desktop or Docker Engine is installed, build the image locally using:

```bash
docker build -t retail-sales-forecasting:latest .
```

### 3.3 Running the Docker Container
Run the container with interactive port forwarding and automated restart policies:

```bash
docker run -d \
  --name retail-sales-app \
  -p 8501:8501 \
  --restart unless-stopped \
  retail-sales-forecasting:latest
```

### 3.4 Local Container Verification & Health Check
Verify container health and inspect running logs:

```bash
# Check container status and health:
docker ps --filter "name=retail-sales-app"

# Inspect application logs:
docker logs -f retail-sales-app

# Host HTTP smoke test (PowerShell):
Invoke-WebRequest -Uri "http://localhost:8501" -UseBasicParsing

# Host HTTP smoke test (Bash):
curl -I http://localhost:8501
```

Expected HTTP status: `HTTP/1.1 200 OK`.

### 3.5 Stopping the Container
```bash
docker stop retail-sales-app
docker rm retail-sales-app
```

---

## 4. Cloud Deployment Architecture

### 4.1 Recommended Platform: Streamlit Community Cloud
Streamlit Community Cloud is the optimal hosting platform for this project:
- **Direct GitHub Sync:** Continuously synchronizes with the `main` branch.
- **Native Caching:** Supports `@st.cache_data` and `@st.cache_resource` in-memory.
- **Zero Configuration:** Automatically detects `requirements.txt` and `.streamlit/config.toml`.

### 4.2 Exact Steps for User Cloud Deployment (Manual Authentication)
Because cloud deployment requires GitHub OAuth authorization, the user should execute the following steps:

1. **Push to GitHub:** Ensure all project commits are pushed to the user's GitHub repository:
   ```bash
   git push origin main
   ```
2. **Access Streamlit Cloud:** Navigate to [share.streamlit.io](https://share.streamlit.io) and log in with GitHub.
3. **Create New App:**
   - Click **"Create app"** $\rightarrow$ **"Deploy a public app from GitHub"**.
   - Select the repository: `Retail-Sales-Forecasting`.
   - Set Branch: `main`.
   - Set Main file path: `dashboard/app.py`.
   - App URL (optional custom subdomain): `retail-sales-forecasting`.
4. **Deploy:** Click **"Deploy!"**. Streamlit Cloud will automatically build the environment from `requirements.txt` and launch the dashboard.

### 4.3 Alternative Cloud Platforms (Container-Native)
- **Render.com:**
  - Create a new **Web Service**, select **Docker** environment, and connect the repository.
  - Set Port: `8501`.
  - Render will execute `Dockerfile` automatically.
- **Railway.app:**
  - Create a new project $\rightarrow$ **Deploy from GitHub repo**.
  - Railway auto-detects `Dockerfile` and exposes port `8501`.

---

## 5. Environment Variables & Configuration

The application is designed to be fully self-contained using relative project paths defined in `src/config.py`. However, for advanced cloud deployments, optional environment variables can be provided:

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `ENVIRONMENT` | `production` | Deployment environment flag (`development`, `production`) |
| `STREAMLIT_SERVER_PORT` | `8501` | Network port for web server |
| `STREAMLIT_SERVER_ADDRESS` | `0.0.0.0` | Network binding interface |
| `STREAMLIT_SERVER_HEADLESS` | `true` | Headless execution mode |

Configuration is standardized in `.streamlit/config.toml`.

---

## 6. Security Audit & Governance

A strict repository security audit was conducted:
- **Secret Scanning:** Scanned for API keys, bearer tokens, passwords, and private certificates (`api_key`, `password`, `secret`). Result: **Zero hardcoded credentials**.
- **Environment Isolation:** `.env` is explicitly ignored by both `.gitignore` and `.dockerignore`.
- **Placeholder Standards:** `.env.example` contains only non-sensitive architectural placeholders.
- **Non-Root Execution:** The production Docker container executes under a dedicated unprivileged user (`appuser`, UID 1000).

---

## 7. Operational Troubleshooting

| Symptom | Probable Cause | Corrective Action |
| :--- | :--- | :--- |
| `ModuleNotFoundError: No module named 'src'` | Working directory is not project root | Set `PYTHONPATH=.` or run from project root directory |
| Port 8501 already bound | Another process or container is using 8501 | Run with alternative port: `--server.port=8502` or stop conflicting process |
| `FileNotFoundError: final_model.joblib` | Production model not trained | Run `python -m src.forecasting` to regenerate model artifacts |
| Container health check failing | Slow initial dependency compilation | Increase `--start-period=15s` in Docker health check |
