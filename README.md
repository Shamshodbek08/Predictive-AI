---
title: AISC1000B GenZ Library Wi-Fi Dashboard
emoji: 👓
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: HKUST Library Wi-Fi Predictive Advanced explorer
---

# AISC1000B GenZ glasses — Library Wi-Fi dashboard (v2)

English-only, tabbed version of the dashboard:

1. **What we found** — why this data, why it matters, two non-obvious trends and two hidden patterns (each with a chart and a "why it matters" line).
2. **Explore the data** — date / floor / user-group filters, KPI row, heatmap, floor bar chart, time series.
3. **Forecast** — pick floor, user group, weekday and month; same model and numbers as v1.
4. **About the data** — source, data-quality notes, ethics.

**Numbers are unchanged from v1.** Everything above the "Presentation layer" line in `app.py`
(data loading, model, insights) is copied from v1 as-is; only the layout was rewritten.
Checked: same model scores (152.74 / 171.36 / 146.75 / 147.12), same four insights,
all 2,940 forecast combinations identical, same chart data.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Publish

See `DEPLOY.md`.
