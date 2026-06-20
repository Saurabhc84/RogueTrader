# Deploying to GitHub + Streamlit Community Cloud

Goal: push this project to GitHub, then deploy the dashboard so it has a public
shareable URL (no install needed for whoever opens it — ideal for the viva).

---

## Part A — One-time setup

1. **GitHub account**: sign up free at https://github.com if you don't have one.
2. **Install Git for Windows**: https://git-scm.com/download/win — run the
   installer, accept all defaults. Close and reopen Command Prompt afterwards.
3. Verify: in a fresh Command Prompt, `git --version` should print a version.

---

## Part B — Put your Colab CSVs in place (IMPORTANT)

The deployed app reads the five CSVs from the repo's `outputs/` folder.
Copy these from your Colab run into `rogue_trader_poc\outputs\`:

    benchmark.csv
    communications_scored.csv
    final_scored.csv
    labels.csv
    lead_time.csv

(They total ~7 MB — fine to commit to Git.)

---

## Part C — Push to GitHub

Open Command Prompt in the project folder:

    cd "E:\Saurabh\College\IIM_Sirmaur\MaterialShared by Prof\Term3\Capstone\Claude_rogue\rogue_trader_poc\rogue_trader_poc"

First time only — tell Git who you are:

    git config --global user.name "Your Name"
    git config --global user.email "your_email@example.com"

Initialise and commit:

    git init
    git add .
    git commit -m "Rogue trader early-warning POC: dashboard + pipeline"
    git branch -M main

Now create an EMPTY repo on GitHub:
  - Go to https://github.com/new
  - Name it e.g. `rogue-trader-poc`
  - Set it to **Public** (required for free Streamlit Cloud) or Private
  - Do NOT tick "Add a README" (you already have one)
  - Click "Create repository"

GitHub shows a URL like `https://github.com/<you>/rogue-trader-poc.git`.
Connect and push (replace the URL with yours):

    git remote add origin https://github.com/<you>/rogue-trader-poc.git
    git push -u origin main

You'll be asked to sign in — a browser window opens; authorise it.
Refresh your GitHub repo page; your files should appear.

---

## Part D — Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io and sign in **with GitHub**.
2. Click **"Create app"** → **"Deploy a public app from GitHub"**.
3. Fill in:
   - **Repository**: `<you>/rogue-trader-poc`
   - **Branch**: `main`
   - **Main file path**: `app/dashboard.py`
4. Click **Deploy**. First build takes 2–4 minutes (it installs from
   `requirements.txt` — the lightweight one, no torch).
5. You get a URL like `https://rogue-trader-poc.streamlit.app` — share it,
   open it on any device, put it in your capstone report.

---

## Updating later

After you change code or refresh the CSVs:

    git add .
    git commit -m "describe what changed"
    git push

Streamlit Cloud auto-redeploys within ~a minute of the push.

---

## Notes
- `requirements.txt` is the deploy/dashboard set (pandas, numpy, streamlit,
  plotly). For running the full training pipeline locally, use
  `requirements-full.txt` (adds torch, scikit-learn, shap).
- The dashboard's `E:\...` path is ignored on the cloud (folder doesn't exist
  there); it automatically reads the repo's `outputs/` instead.
- If the build fails, open the app's "Manage app" logs on Streamlit Cloud —
  usually a missing CSV in outputs/ or a package name typo.
