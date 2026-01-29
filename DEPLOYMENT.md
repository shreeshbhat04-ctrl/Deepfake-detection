# Deployment Guide for Deepfake Detection Backend

This backend is containerized using `Dockerfile`, making it easy to deploy on any platform that supports Docker.

## Option 1: Hugging Face Spaces (Recommended for ML)
Hugging Face Spaces offers a generous free tier (2 vCPU, 16GB RAM) which is excellent for machine learning models.

1.  Create a new Space on [Hugging Face](https://huggingface.co/new-space).
2.  Select **Docker** as the Space SDK.
3.  Upload your project files (including `Dockerfile` and `saved_models/`) to the Space's repository.
4.  It will build automatically.

## Option 2: Render.com (Easiest for Web Apps)
Render is very easy to connect to GitHub.

1.  Push your code to a GitHub repository.
2.  Create a new **Web Service** on Render.
3.  Connect your repo.
4.  Select **Docker** as the Environment.
5.  Render will auto-detect the `Dockerfile` and build it.
    *   *Note: The free tier spins down after inactivity (slow first request).*

## Option 3: Google Cloud Run (Scalable)
Serverless container hosting.

1.  Install Google Cloud SDK (`gcloud`).
2.  Run:
    ```bash
    gcloud run deploy deepfake-backend --source .
    ```
3.  It will handle building and deploying to a public URL.

## Important: Frontend Connection
Once deployed, you will get a public URL (e.g., `https://my-backend.onrender.com`).
You must update your **Frontend** or **Chrome Extension** to point to this new URL instead of `http://localhost:8000`.

**For Chrome Extension:**
Update `chrome-extension/manifest.json` host_permissions and `background.js` API base URL.

**For Web App:**
Update the `.env` file or `VITE_API_URL` variable.
