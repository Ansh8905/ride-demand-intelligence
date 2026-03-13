# Deployment Guide

This project has two deploy targets:

1. Streamlit dashboard (`app.py`)
2. FastAPI service (`app_fastapi.py`)

## A) Streamlit Community Cloud

1. Push latest `main` branch to GitHub.
2. Open Streamlit Community Cloud.
3. Click **New app**.
4. Select:
   - Repository: `Ansh8905/ride-demand-intelligence`
   - Branch: `main`
   - Main file path: `app.py`
5. Click **Deploy**.

If first load is slow, wait for model/data bootstrap on initial run.

## B) Render (FastAPI)

This repo includes `render.yaml` for Blueprint deployment.

1. Open Render dashboard.
2. Click **New +** -> **Blueprint**.
3. Connect this repository.
4. Render auto-creates `ride-demand-api` using `render.yaml`.
5. After deploy, verify:

```bash
curl https://<your-render-domain>/health
```

## Common Fixes

- `ModuleNotFoundError`: ensure latest code is pushed and build logs show successful `pip install -r requirements.txt`.
- API 503 "Models not loaded": run local pipeline once and commit required artifacts if your deployment depends on prebuilt models.
- Streamlit slow startup: expected on first cold start due to bootstrap logic in `app.py`.
