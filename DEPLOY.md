# Publishing v2

Same setup as v1 (Streamlit Community Cloud, main file `app.py`). Files needed:
`app.py`, `theme.py`, `requirements.txt`, `runtime.txt`, `.streamlit/config.toml`, `data/wifi_clean.parquet`
(`metrics.json`, `Dockerfile`, `README.md` are optional extras).

**Option A — update the existing app (same link):** replace these files in the repo
https://github.com/zzeyu628-eng/aisc-library-wifi-dashboard and push. Streamlit Cloud redeploys automatically
and the old link keeps working.

**Option B — separate new app:** push this folder to a new repo, then on share.streamlit.io choose
"Create app", pick the repo, branch `main`, main file `app.py`. You get a new link.
