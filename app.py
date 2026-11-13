"""AISC1000B GenZ glasses — HKUST Library Wi-Fi Predictive Advanced dashboard (v2, English-only layout).

The data, model and insight code (everything above the "Presentation layer" line) is copied
unchanged from v1, so every number matches v1. Only the page layout below it is new.
"""
from __future__ import annotations

from pathlib import Path

import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so `theme` imports under export_metrics too
import theme as T  # noqa: E402  (visual system only: tokens, CSS, Plotly template, bilingual labels)

ROOT = Path(__file__).resolve().parent  # click-open deploy pack: data/ beside app.py
DATA_PATH = ROOT / "data" / "wifi_clean.parquet"
DOW_LABELS = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
DOI = "doi:10.14711/dataset/RQXRQB"
SOURCE_URL = "https://dataspace.hkust.edu.hk/dataset.xhtml?persistentId=doi:10.14711/dataset/RQXRQB"

try:
    from sklearn.ensemble import HistGradientBoostingRegressor

    HAS_HGB = True
except Exception:  # pragma: no cover
    from sklearn.linear_model import Ridge

    HAS_HGB = False


@st.cache_data
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        st.error(f"Missing {DATA_PATH}. Run: python scripts/merge_wifi.py")
        st.stop()
    df = pd.read_parquet(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def filter_df(df: pd.DataFrame, date_range, floors, groups) -> pd.DataFrame:
    out = df
    if date_range and len(date_range) == 2:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        out = out[(out["Date"] >= start) & (out["Date"] <= end)]
    if floors:
        out = out[out["Floor"].isin(floors)]
    if groups:
        out = out[out["User-Group"].isin(groups)]
    return out


def _encode_features(df: pd.DataFrame, floor_cats, group_cats) -> pd.DataFrame:
    """Build model matrix: floor/group one-hots + dow, month, is_weekend."""
    x = pd.DataFrame(index=df.index)
    for f in floor_cats:
        x[f"floor_{f}"] = (df["Floor"] == f).astype(np.float64)
    for g in group_cats:
        safe = str(g).replace(" ", "_").replace("(", "").replace(")", "")
        x[f"group_{safe}"] = (df["User-Group"] == g).astype(np.float64)
    x["dow"] = df["dow"].astype(np.float64)
    x["month"] = df["month"].astype(np.float64)
    x["is_weekend"] = df["is_weekend"].astype(np.float64)
    return x


@st.cache_resource
def train_models(df: pd.DataFrame):
    """Time-aware 80/20 split; HGB (or Ridge) + same-dow baseline MAE."""
    work = df.sort_values("Date").reset_index(drop=True)
    dates = sorted(work["Date"].unique())
    cut = dates[int(len(dates) * 0.8)]
    train = work[work["Date"] < cut]
    hold = work[work["Date"] >= cut]

    # one-hot schema from the TRAIN window only (review 2026-11-03: was full timeline;
    # numerically identical here because every floor/group appears daily, but cleaner)
    floor_cats = sorted(train["Floor"].unique().tolist())
    group_cats = sorted(train["User-Group"].unique().tolist())

    X_train = _encode_features(train, floor_cats, group_cats)
    y_train = train["User-Count"].astype(np.float64)
    X_hold = _encode_features(hold, floor_cats, group_cats)
    y_hold = hold["User-Count"].astype(np.float64)

    if HAS_HGB:
        model = HistGradientBoostingRegressor(
            max_depth=6,
            learning_rate=0.08,
            max_iter=200,
            random_state=42,
        )
        model_name = "HistGradientBoostingRegressor"
    else:
        model = Ridge(alpha=1.0)
        model_name = "Ridge"

    model.fit(X_train, y_train)
    yhat = model.predict(X_hold)
    mae_model = float(np.mean(np.abs(y_hold - yhat)))

    # same-dow baseline keyed by Floor × User-Group × dow
    means = (
        train.groupby(["Floor", "User-Group", "dow"])["User-Count"]
        .mean()
        .rename("yhat_base")
    )
    hold_b = hold.merge(means, on=["Floor", "User-Group", "dow"], how="left")
    # fallback: Floor×Group mean, then global train mean
    fg = train.groupby(["Floor", "User-Group"])["User-Count"].mean().rename("yhat_fg")
    hold_b = hold_b.merge(fg, on=["Floor", "User-Group"], how="left")
    hold_b["yhat_base"] = hold_b["yhat_base"].fillna(hold_b["yhat_fg"]).fillna(y_train.mean())
    mae_base = float(np.mean(np.abs(hold_b["User-Count"] - hold_b["yhat_base"])))

    # Opus final review: two STRONGER simple baselines, both built only from data before `cut`.
    # (a) recency: Floor×Group×dow mean over the last 28 days of train with non-zero building
    #     total (train ends after an all-zero gap 2021-02-12..03-02); (b) seasonal: 2019
    #     Floor×Group×month×dow mean (last full pre-pandemic year).
    def _keyed_mae(src: pd.DataFrame, keys) -> float:
        m = src.groupby(keys)["User-Count"].mean().rename("b")
        hb = hold.merge(m, on=keys, how="left").merge(fg, on=["Floor", "User-Group"], how="left")
        pred = hb["b"].fillna(hb["yhat_fg"]).fillna(y_train.mean())
        return float(np.mean(np.abs(hb["User-Count"] - pred)))

    day_tot = train.groupby("Date")["User-Count"].transform("sum")
    nz = train[day_tot > 0]
    recent = nz[nz["Date"] >= nz["Date"].max() - pd.Timedelta(days=27)]
    mae_recent = _keyed_mae(recent, ["Floor", "User-Group", "dow"])
    mae_season = _keyed_mae(train[train["Date"].dt.year == 2019], ["Floor", "User-Group", "month", "dow"])

    meta = {
        "cut": str(pd.Timestamp(cut).date()),
        "n_train": int(len(train)),
        "n_hold": int(len(hold)),
        "mae_model": mae_model,
        "mae_baseline": mae_base,
        "mae_rel_improvement_pct": 100.0 * (mae_base - mae_model) / mae_base,
        "mae_baseline_recent28d_same_dow": mae_recent,
        "mae_baseline_2019_same_month_dow": mae_season,
        "train_mean": float(y_train.mean()),
        "hold_mean": float(y_hold.mean()),
        "model_name": model_name,
        "floor_cats": floor_cats,
        "group_cats": group_cats,
    }
    return model, meta, means


def predict_user_count(model, meta, floor: str, group: str, dow: int, month: int) -> float:
    row = pd.DataFrame(
        [
            {
                "Floor": floor,
                "User-Group": group,
                "dow": int(dow),
                "month": int(month),
                "is_weekend": bool(int(dow) in (5, 6)),
            }
        ]
    )
    X = _encode_features(row, meta["floor_cats"], meta["group_cats"])
    return float(model.predict(X)[0])


def compute_insights(df: pd.DataFrame) -> list[dict]:
    """≥3 auto-calculated non-trivial findings from the cleaned table.

    Each dict carries `values` (raw numbers) so metrics.json can be regenerated
    from exactly the same code that renders the dashboard text.
    """
    insights: list[dict] = []
    U = "User-Count"

    # 1) Teaching-term vs summer-break weekday load (Opus final review 2026-11-03).
    #    Data-quality fix: on Mon 2018-11-05 the daily building total doubles overnight
    #    (~5.3k -> ~10.8k weekday users) while WiFi-Conn per user jumps 3.0 -> 5.3 and
    #    Duration per user halves: a counting-definition change, not behaviour. Nov-Dec 2019
    #    is the campus-disruption period. So each year is compared WITHIN one counting
    #    regime: Feb-May + Sep-Oct weekdays vs Jul-Aug weekdays of the SAME year
    #    (2018: all before 2018-11-05; 2019: all after it and before the disruption).
    #    Month sets are calendar PROXIES, not the official HKUST calendar.
    wd_all = df[~df["is_weekend"]]
    term_m, summer_m = [2, 3, 4, 5, 9, 10], [7, 8]

    def _ratio(sub: pd.DataFrame, a, b) -> float:
        num = sub.loc[sub["month"].isin(a), U].mean()
        den = sub.loc[sub["month"].isin(b), U].mean()
        return float(num / den) if den and den > 0 else float("nan")

    w18 = wd_all[wd_all["year"] == 2018]
    w19 = wd_all[wd_all["year"] == 2019]
    if len(w18) and len(w19):
        r18 = _ratio(w18, term_m, summer_m)
        r19 = _ratio(w19, term_m, summer_m)
        by_floor = {
            fl: _ratio(w19[w19["Floor"] == fl], term_m, summer_m) for fl in sorted(df["Floor"].unique())
        }
        top = max(by_floor, key=by_floor.get)
        lo = min(by_floor, key=by_floor.get)
        may_vs_marapr = _ratio(w19, [5], [3, 4])
        insights.append(
            {
                "title_en": "Teaching-term vs summer-break weekday load",
                "title_zh": "学期 vs 暑假：工作日人数倍率",
                "body_en": (
                    f"Weekday mean User-Count, teaching months (Feb–May, Sep–Oct) vs Jul–Aug of the same year: "
                    f"{r18:.2f}× in 2018 and {r19:.2f}× in 2019. By floor (2019) it ranges from "
                    f"{by_floor[lo]:.2f}× on {lo} to {by_floor[top]:.2f}× on {top}. "
                    f"No exam-month surge: May 2019 weekdays were {may_vs_marapr:.2f}× Mar–Apr 2019; the busiest "
                    f"weeks fall inside teaching weeks, not exam periods. Years are compared separately because the counting "
                    f"definition changes on 2018-11-05; month sets are proxies, not the official calendar."
                ),
                "body_zh": (
                    f"工作日均值，同年教学月（2–5、9–10 月）对 7–8 月：2018 年 {r18:.2f} 倍，2019 年 {r19:.2f} 倍；"
                    f"按楼层（2019）由 {lo} 层 {by_floor[lo]:.2f}× 到 {top} 层 {by_floor[top]:.2f}×。"
                    f"无考试月激增：2019 年 5 月工作日仅为 3–4 月的 {may_vs_marapr:.2f} 倍，最忙的周落在授课周而非考试期。"
                    f"因 2018-11-05 起计数口径变化，各年分开比较；月份仅为代理，非官方校历。"
                ),
                "values": {
                    "term_vs_summer_weekday_2018": round(r18, 3),
                    "term_vs_summer_weekday_2019": round(r19, 3),
                    "term_vs_summer_weekday_by_floor_2019": {k: round(v, 3) for k, v in by_floor.items()},
                    "term_vs_summer_top_floor": top,
                    "term_vs_summer_low_floor": lo,
                    "exam_proxy_may_vs_marapr_weekday_2019": round(may_vs_marapr, 3),
                },
            }
        )

    # 2) 2020 vs 2018–19 same calendar months
    base = (
        df[df["year"].isin([2018, 2019])]
        .groupby(df.loc[df["year"].isin([2018, 2019]), "month"])[U]
        .mean()
    )
    y20 = df[df["year"] == 2020].groupby("month")[U].mean()
    common = sorted(set(base.index) & set(y20.index))
    if common:
        drop = (y20.loc[common] - base.loc[common]) / base.loc[common] * 100
        worst_m = int(drop.idxmin())
        tot20 = df.loc[df["year"] == 2020, U].sum()
        tot_base = df.loc[df["year"].isin([2018, 2019]), U].sum() / 2.0
        total_drop = float((tot20 / tot_base - 1) * 100) if tot_base > 0 else float("nan")
        insights.append(
            {
                "title_en": "2020 vs 2018–19 same-month drop",
                "title_zh": "2020 相对 2018–19 同月跌幅",
                "body_en": (
                    f"Average of the 12 same-calendar-month changes in mean User-Count: "
                    f"{drop.mean():.1f}% (total 2020 volume {total_drop:.1f}% vs the 2018–19 yearly average; "
                    f"worst month {worst_m}: {drop.loc[worst_m]:.1f}%; median month {drop.median():.1f}%; "
                    f"{int((drop < -50).sum())} of {len(drop)} months below −50%). "
                    f"Timing is consistent with the campus-closure / remote-teaching period; "
                    f"the 2018–19 baseline also includes late-2019 disruption and a counting change on "
                    f"2018-11-05, so the size is indicative (conservative), not exact."
                ),
                "body_zh": (
                    f"12 个同公历月份变化的平均值：{drop.mean():.1f}%"
                    f"（2020 全年总量较 2018–19 年均 {total_drop:.1f}%；最差月份 {worst_m} 月：{drop.loc[worst_m]:.1f}%；"
                    f"月度中位数 {drop.median():.1f}%；{int((drop < -50).sum())}/{len(drop)} 个月跌超 50%）。"
                    f"时间上与校园停摆/远程教学时段一致；2018–19 基线含 2019 年末扰动及 2018-11-05 计数口径变化，幅度仅供参考（偏保守）。"
                ),
                "values": {
                    "y2020_vs_1819_mean_drop_pct": round(float(drop.mean()), 2),
                    "y2020_vs_1819_total_drop_pct": round(total_drop, 2),
                    "y2020_worst_month": worst_m,
                    "y2020_worst_month_drop_pct": round(float(drop.loc[worst_m]), 2),
                    "y2020_vs_1819_median_drop_pct": round(float(drop.median()), 2),
                    "y2020_months_below_minus50": int((drop < -50).sum()),
                },
            }
        )

    # 3) UG vs RPG: weekend retention (stable) and peak floor (period-dependent)
    def peak_floor(g: str, years=None):
        sub = df[df["User-Group"] == g]
        if years is not None:
            sub = sub[sub["year"].isin(years)]
        s = sub.groupby("Floor")[U].mean()
        return (s.idxmax(), float(s.max())) if len(s) else (None, float("nan"))

    def we_ratio(g: str, year=None) -> float:
        sub = df[df["User-Group"] == g]
        if year is not None:
            sub = sub[sub["year"] == year]
        wd = sub.loc[~sub["is_weekend"], U].mean()
        we = sub.loc[sub["is_weekend"], U].mean()
        return float(we / wd) if wd and wd > 0 else float("nan")

    ug_peak, ug_peak_v = peak_floor("UG")
    rpg_peak, rpg_peak_v = peak_floor("RPG")
    if ug_peak and rpg_peak:
        ug_peak_pre, _ = peak_floor("UG", [2018, 2019])
        rpg_peak_pre, _ = peak_floor("RPG", [2018, 2019])
        ug_we, rpg_we = we_ratio("UG"), we_ratio("RPG")
        yrs = sorted(df["year"].unique())
        ug_y = {int(y): we_ratio("UG", y) for y in yrs}
        rpg_y = {int(y): we_ratio("RPG", y) for y in yrs}
        rpg_above_every_year = all(rpg_y[y] > ug_y[y] for y in ug_y)
        insights.append(
            {
                "title_en": "UG vs RPG: weekend retention gap (and peak floors)",
                "title_zh": "本科 vs 研究型研究生：周末留存差异（及峰值楼层）",
                "body_en": (
                    f"Weekend/weekday ratio over 2018–21: UG {ug_we:.1%} vs RPG {rpg_we:.1%}"
                    + (" — RPG is higher in every year" if rpg_above_every_year else "")
                    + f" (UG {min(ug_y.values()):.0%}–{max(ug_y.values()):.0%}, "
                    f"RPG {min(rpg_y.values()):.0%}–{max(rpg_y.values()):.0%} across years). "
                    f"Highest-mean floor over all years: UG {ug_peak} ({ug_peak_v:.1f}), "
                    f"RPG {rpg_peak} ({rpg_peak_v:.1f}); but in 2018–19 alone both peak on "
                    f"{ug_peak_pre}/{rpg_peak_pre}, so the floor gap is period-dependent."
                ),
                "body_zh": (
                    f"2018–21 周末/工作日比：UG {ug_we:.1%} vs RPG {rpg_we:.1%}"
                    + ("（逐年 RPG 都更高）" if rpg_above_every_year else "")
                    + f"。全期均值最高楼层：UG {ug_peak}（{ug_peak_v:.1f}）、RPG {rpg_peak}（{rpg_peak_v:.1f}）；"
                    f"但仅看 2018–19，两者峰值分别为 {ug_peak_pre}/{rpg_peak_pre}，楼层差异随时期而变。"
                ),
                "values": {
                    "ug_peak_floor": ug_peak,
                    "rpg_peak_floor": rpg_peak,
                    "ug_peak_floor_1819": ug_peak_pre,
                    "rpg_peak_floor_1819": rpg_peak_pre,
                    "ug_weekend_retention": round(ug_we, 4),
                    "rpg_weekend_retention": round(rpg_we, 4),
                    "ug_weekend_retention_by_year": {str(k): round(v, 3) for k, v in ug_y.items()},
                    "rpg_weekend_retention_by_year": {str(k): round(v, 3) for k, v in rpg_y.items()},
                },
            }
        )

    # 4) Guest (Non-HKUST) share: rises when members leave (summer), annual share is volatile.
    def _nh_share(sub: pd.DataFrame) -> float:
        tot = sub[U].sum()
        return float(100.0 * sub.loc[sub["User-Group"] == "Non-HKUST", U].sum() / tot) if tot > 0 else float("nan")

    shares = {int(y): _nh_share(df[df["year"] == int(y)]) for y in sorted(df["year"].unique())}
    shares = {k: v for k, v in shares.items() if pd.notna(v)}
    season = {}
    for y in (2018, 2019):
        s = wd_all[wd_all["year"] == y]
        if len(s):
            season[y] = (_nh_share(s[s["month"].isin(term_m)]), _nh_share(s[s["month"].isin(summer_m)]))
    if len(shares) >= 2 and len(season) == 2:
        y1 = max(shares)
        last = df[df["year"] == y1]
        nh_m = last[last["User-Group"] == "Non-HKUST"].groupby("month")[U].sum()
        pk_m = int(nh_m.idxmax())
        pk_frac = float(nh_m.max() / nh_m.sum() * 100) if nh_m.sum() > 0 else float("nan")
        share_ex = _nh_share(last[last["month"] != pk_m])
        share_q4 = _nh_share(last[last["month"] >= 10])
        (t18, s18), (t19, s19) = season[2018], season[2019]
        insights.append(
            {
                "title_en": "Guests (Non-HKUST) fill a bigger share when students are away",
                "title_zh": "学生离校时访客（Non-HKUST）占比上升",
                "body_en": (
                    f"Weekday Non-HKUST share of User-Count, teaching months vs Jul–Aug: "
                    f"{t18:.2f}% → {s18:.2f}% (2018) and {t19:.2f}% → {s19:.2f}% (2019). Guest use also falls in "
                    f"summer, but far less than member use, so the library's summer user mix is more external. "
                    f"Annual shares (" + ", ".join(f"{y}={v:.2f}%" for y, v in shares.items()) + ") are NOT a trend: "
                    f"month {pk_m} of {y1} alone holds {pk_frac:.0f}% of that year's guest volume (excl. it "
                    f"{share_ex:.2f}%; Oct–Dec {y1} {share_q4:.2f}%), in a period when counting definitions shift."
                ),
                "body_zh": (
                    f"工作日 Non-HKUST 占比，教学月 → 7–8 月：2018 年 {t18:.2f}% → {s18:.2f}%，"
                    f"2019 年 {t19:.2f}% → {s19:.2f}%。暑假访客也减少，但远少于校内用户，所以暑期用户构成更“外部”。"
                    f"年度占比（" + "，".join(f"{y}={v:.2f}%" for y, v in shares.items()) + "）不是趋势："
                    f"{y1} 年 {pk_m} 月一个月占全年访客量 {pk_frac:.0f}%（剔除后 {share_ex:.2f}%；10–12 月 {share_q4:.2f}%），"
                    f"且该时期计数口径在变。"
                ),
                "values": {
                    "non_hkust_weekday_share_term_vs_summer_2018": [round(t18, 3), round(s18, 3)],
                    "non_hkust_weekday_share_term_vs_summer_2019": [round(t19, 3), round(s19, 3)],
                    "non_hkust_share_by_year": {str(k): round(v, 3) for k, v in shares.items()},
                    "non_hkust_last_year_peak_month": pk_m,
                    "non_hkust_last_year_peak_month_volume_pct": round(pk_frac, 1),
                    "non_hkust_last_year_share_excl_peak_month": round(share_ex, 3),
                    "non_hkust_last_year_q4_share": round(share_q4, 3),
                },
            }
        )

    return insights


# ============================================================================
# Presentation layer (v2: English only, tabbed story layout, 2026-11-13).
# Everything above this line is the reviewed data/model logic and is unchanged;
# everything below only arranges and draws it.
# ============================================================================
import html as _html_mod  # noqa: E402


def predict_grid(model, meta, floor: str, group: str, dows, months) -> np.ndarray:
    """Vectorised `predict_user_count` for many (dow, month) cells of one Floor × User-Group.

    Builds exactly the same feature rows as `predict_user_count`, so each value equals the
    single-cell prediction.
    """
    rows = pd.DataFrame(
        {
            "Floor": floor,
            "User-Group": group,
            "dow": [int(d) for d in dows],
            "month": [int(m) for m in months],
        }
    )
    rows["is_weekend"] = rows["dow"].isin((5, 6))
    X = _encode_features(rows, meta["floor_cats"], meta["group_cats"])
    return model.predict(X).astype(float)


@st.cache_data
def seasonal_2019_means(df: pd.DataFrame, cut: str) -> pd.Series:
    """Observed 2019 Floor × Group × month × dow mean, train window only (= the seasonal baseline)."""
    tr = df[(df["Date"] < pd.Timestamp(cut)) & (df["Date"].dt.year == 2019)]
    return tr.groupby(["Floor", "User-Group", "month", "dow"])["User-Count"].mean()


@st.cache_data
def raw_insights(df: pd.DataFrame) -> list[dict]:
    """compute_insights() run with rounding switched off, so cards format the same unrounded
    numbers as v1's text (v1 formats raw values; `values` holds 3-decimal rounded copies)."""
    import types
    fn = types.FunctionType(compute_insights.__code__, {**globals(), "round": lambda x, n=None: x})
    return fn(df)


def _md(body: str) -> None:
    st.markdown(body, unsafe_allow_html=True)


def _esc(s: str) -> str:
    return _html_mod.escape(str(s), quote=False)


def _kpi_strip(items: list[tuple[str, str, str]]) -> None:
    cells = "".join(
        f"<div class='kpi'><div class='l'>{_esc(l)}</div><div class='v'>{_esc(v)}</div>"
        f"<div class='s'>{_esc(s)}</div></div>"
        for l, v, s in items
    )
    _md(f"<div class='kpis'>{cells}</div>")


def _plot(fig: go.Figure, key: str) -> None:
    st.plotly_chart(fig, width="stretch", theme=None, config=T.PLOTLY_CONFIG, key=key)


# ---------------------------------------------------------------- story charts
# These redraw numbers the insight code already computes (same formulas, same data),
# so a chart and its insight card always agree.
@st.cache_data
def term_summer_daily(df: pd.DataFrame) -> dict:
    """Mean weekday DAILY total (all floors & groups), teaching months vs Jul–Aug, per year."""
    wd = df[~df["is_weekend"]]
    out = {}
    for y in (2018, 2019):
        s = wd[wd["year"] == y]
        term = s[s["month"].isin([2, 3, 4, 5, 9, 10])].groupby("Date")["User-Count"].sum().mean()
        summer = s[s["month"].isin([7, 8])].groupby("Date")["User-Count"].sum().mean()
        out[y] = (float(term), float(summer))
    return out


@st.cache_data
def monthly_drop_2020(df: pd.DataFrame) -> pd.Series:
    """Same calculation as insight 2: 2020 monthly mean vs 2018–19 same-month mean, in %."""
    U = "User-Count"
    base = df[df["year"].isin([2018, 2019])].groupby(df.loc[df["year"].isin([2018, 2019]), "month"])[U].mean()
    y20 = df[df["year"] == 2020].groupby("month")[U].mean()
    common = sorted(set(base.index) & set(y20.index))
    return (y20.loc[common] - base.loc[common]) / base.loc[common] * 100


def fig_term_summer(vals: dict, r18: float, r19: float) -> go.Figure:
    years = ["2018", "2019"]
    term = [vals[2018][0], vals[2019][0]]
    summer = [vals[2018][1], vals[2019][1]]
    fig = go.Figure()
    fig.add_bar(x=years, y=term, name="Teaching-month weekdays (Feb–May, Sep–Oct)", marker_color=T.HUE,
                text=[f"{v:,.0f}" for v in term], textposition="outside", cliponaxis=False,
                hovertemplate="Teaching months, %{x}: %{y:,.0f} per day<extra></extra>")
    fig.add_bar(x=years, y=summer, name="Summer weekdays (Jul–Aug)", marker_color=T.NEUTRAL_BAR,
                text=[f"{v:,.0f}" for v in summer], textposition="outside", cliponaxis=False,
                hovertemplate="Summer, %{x}: %{y:,.0f} per day<extra></extra>")
    for x, r in zip(years, (r18, r19)):
        fig.add_annotation(x=x, y=max(term) * 1.22, text=f"<b>{r:.2f}×</b>", showarrow=False,
                           font=dict(size=15, color=T.INK))
    fig.update_layout(barmode="group", barcornerradius=3, showlegend=True)
    fig.update_yaxes(title_text="Wi-Fi users per weekday", rangemode="tozero", tickformat=",",
                     range=[0, max(term) * 1.35])
    return T.finish(fig, "Average weekday, whole library",
                    "Each year compared with itself (the counting method changed in Nov 2018)", height=340)


def fig_drop_2020(drop: pd.Series) -> go.Figure:
    months = [T.MONTH_EN[m - 1] for m in drop.index]
    fig = go.Figure()
    fig.add_bar(x=months, y=drop.values, marker_color=T.NEGATIVE,
                text=[f"{v:.0f}%" for v in drop.values], textposition="outside", cliponaxis=False,
                hovertemplate="%{x} 2020: %{y:.1f}% vs 2018–19<extra></extra>")
    fig.add_hline(y=float(drop.mean()), line=dict(color=T.INK, width=1, dash="dash"))
    fig.update_yaxes(title_text="Change vs 2018–19 (%)", range=[-110, 5], ticksuffix="%")
    fig.update_layout(barcornerradius=3, showlegend=False)
    return T.finish(fig, "2020, month by month",
                    f"Each 2020 month vs the same month in 2018–19 · dashed line = average ({drop.mean():.1f}%)",
                    height=340)


def fig_weekend(ug_y: dict, rpg_y: dict) -> go.Figure:
    years = sorted(ug_y)
    fig = go.Figure()
    fig.add_bar(x=years, y=[ug_y[y] * 100 for y in years], name=T.group_label("UG"),
                marker_color=T.GROUP_COLORS["UG"], text=[f"{ug_y[y]:.0%}" for y in years],
                textposition="outside", cliponaxis=False,
                hovertemplate="UG %{x}: %{y:.1f}%<extra></extra>")
    fig.add_bar(x=years, y=[rpg_y[y] * 100 for y in years], name=T.group_label("RPG"),
                marker_color=T.GROUP_COLORS["RPG"], text=[f"{rpg_y[y]:.0%}" for y in years],
                textposition="outside", cliponaxis=False,
                hovertemplate="RPG %{x}: %{y:.1f}%<extra></extra>")
    fig.update_layout(barmode="group", barcornerradius=3)
    fig.update_yaxes(title_text="Weekend as % of weekday", range=[0, 80], ticksuffix="%")
    fig.update_xaxes(type="category")
    return T.finish(fig, "Weekend use as a share of weekday use", "By year, all floors", height=340)


def fig_guests(s18: list, s19: list) -> go.Figure:
    years = ["2018", "2019"]
    term = [s18[0], s19[0]]
    summer = [s18[1], s19[1]]
    fig = go.Figure()
    fig.add_bar(x=years, y=term, name="Teaching months", marker_color=T.NEUTRAL_BAR,
                text=[f"{v:.2f}%" for v in term], textposition="outside", cliponaxis=False,
                hovertemplate="Teaching months %{x}: %{y:.2f}%<extra></extra>")
    fig.add_bar(x=years, y=summer, name="Summer (Jul–Aug)", marker_color=T.GROUP_COLORS["Non-HKUST"],
                text=[f"{v:.2f}%" for v in summer], textposition="outside", cliponaxis=False,
                hovertemplate="Summer %{x}: %{y:.2f}%<extra></extra>")
    fig.update_layout(barmode="group", barcornerradius=3)
    fig.update_yaxes(title_text="Visitors' share of weekday users", range=[0, 6], ticksuffix="%")
    return T.finish(fig, "Visitors' share of weekday users", "Non-HKUST users, teaching months vs summer", height=340)


# ---------------------------------------------------------------- explore charts
def fig_heatmap(fdf: pd.DataFrame) -> go.Figure:
    heat = fdf.groupby(["Floor", "dow"], as_index=False)["User-Count"].mean()
    pivot = heat.pivot(index="Floor", columns="dow", values="User-Count").reindex(columns=range(7))
    pivot = pivot.dropna(axis=1, how="all")
    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=[T.dow_label(int(d)) for d in pivot.columns],
            y=[f"{f}" for f in pivot.index],
            colorscale=[[i / (len(T.SEQ_BLUE) - 1), c] for i, c in enumerate(T.SEQ_BLUE)],
            texttemplate="%{z:.0f}",
            textfont=dict(size=13),
            xgap=2,
            ygap=2,
            colorbar=dict(title=dict(text="Average", font=dict(size=11, color=T.MUTED)), thickness=10),
            hovertemplate="Floor %{y} · %{x}<br>Average count: %{z:.1f}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed", showline=False, title_text="Floor", showgrid=False)
    fig.update_xaxes(showline=False)
    return T.finish(fig, "Which floors are busy on which days?",
                    "Average count per floor, per user group, per day · darker = busier", height=380)


def fig_buckets(fdf: pd.DataFrame) -> go.Figure | None:
    focus = {"UG", "RPG", "TPG", "Staff"}
    bar_df = fdf[fdf["User-Group"].isin(focus)].copy()
    if bar_df.empty:
        return None
    bar_df["bucket"] = bar_df["User-Group"].map(
        lambda g: "UG" if g == "UG" else ("Staff" if g == "Staff" else "PG (RPG+TPG)")
    )
    bar = bar_df.groupby(["Floor", "bucket"], as_index=False)["User-Count"].mean()
    fig = go.Figure()
    for b in ["UG", "PG (RPG+TPG)", "Staff"]:
        sub = bar[bar["bucket"] == b]
        if sub.empty:
            continue
        fig.add_bar(
            x=sub["Floor"], y=sub["User-Count"], name=T.group_label(b), marker_color=T.BUCKET_COLORS[b],
            hovertemplate=f"{T.group_label(b)} · Floor %{{x}}<br>Average per day: %{{y:.1f}}<extra></extra>",
        )
    fig.update_layout(barmode="group", barcornerradius=3)
    fig.update_xaxes(title_text="Floor")
    fig.update_yaxes(title_text="Average count per day", rangemode="tozero")
    return T.finish(fig, "Who uses which floor?",
                    "Postgrads bar = average of RPG and TPG, not their total", height=380)


def fig_series(fdf: pd.DataFrame, smooth: bool) -> go.Figure:
    ts = fdf.groupby(["Date", "User-Group"], as_index=False)["User-Count"].sum()
    if smooth:
        ts = ts.sort_values("Date")
        ts["User-Count"] = ts.groupby("User-Group")["User-Count"].transform(
            lambda s: s.rolling(7, min_periods=1).mean()
        )
    fig = go.Figure()
    order = [g for g in T.GROUP_COLORS if g in set(ts["User-Group"])] + sorted(
        set(ts["User-Group"]) - set(T.GROUP_COLORS)
    )
    for g in order:
        sub = ts[ts["User-Group"] == g]
        fig.add_scatter(
            x=sub["Date"], y=sub["User-Count"], mode="lines", name=T.group_label(g),
            line=dict(width=1.4, color=T.GROUP_COLORS.get(g, T.FAINT)),
            hovertemplate=f"{T.group_label(g)}: %{{y:,.0f}}<extra></extra>",
        )
    change = pd.Timestamp("2018-11-05")
    if len(ts) and ts["Date"].min() <= change <= ts["Date"].max():
        fig.add_shape(type="line", x0=change, x1=change, y0=0, y1=1, yref="paper",
                      line=dict(color=T.FAINT, width=1, dash="dot"))
        fig.add_annotation(x=change, y=1, yref="paper", text="counting method changed",
                           showarrow=False, xanchor="left", yanchor="top", xshift=4,
                           font=dict(size=11, color=T.MUTED))
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(title_text="Wi-Fi users per day", rangemode="tozero", tickformat=",")
    fig.update_xaxes(title_text="")
    title = "How use changes over time" + (" (7-day average)" if smooth else "")
    return T.finish(fig, title, "Daily total for each user group", height=420)


# ---------------------------------------------------------------- forecast charts
def fig_pred_months(yhat_m, obs_m, base_pred, p_month, scope: str) -> go.Figure:
    months = list(range(1, 13))
    fig = go.Figure()
    fig.add_shape(type="rect", x0=p_month - 0.5, x1=p_month + 0.5, y0=0, y1=1, yref="paper",
                  fillcolor=T.HUE_TINT, line=dict(width=0), layer="below")
    fig.add_scatter(
        x=months, y=[base_pred] * 12, mode="lines", name="Weekday average (all months)",
        line=dict(color=T.FAINT, width=1.5, dash="dot"),
        hovertemplate="Weekday average (all months): %{y:.1f}<extra></extra>",
    )
    fig.add_scatter(
        x=months, y=obs_m, mode="markers", name="Actual 2019 average",
        marker=dict(size=8, color=T.SURFACE, line=dict(color=T.MUTED, width=1.5)),
        hovertemplate="Actual 2019 average: %{y:.1f}<extra></extra>",
    )
    fig.add_scatter(
        x=months, y=yhat_m, mode="lines+markers", name="Model forecast",
        line=dict(color=T.HUE, width=2.5), marker=dict(size=7, color=T.HUE, line=dict(color=T.SURFACE, width=1.5)),
        hovertemplate="Model forecast: %{y:.1f}<extra></extra>",
    )
    sel = float(yhat_m[p_month - 1])
    fig.add_scatter(
        x=[p_month], y=[sel], mode="markers+text", showlegend=False, hoverinfo="skip",
        marker=dict(size=15, color=T.HUE, line=dict(color=T.SURFACE, width=2.5)),
        text=[f"<b>{sel:.0f}</b>"], textposition="top center", textfont=dict(size=13, color=T.INK),
    )
    fig.update_xaxes(tickvals=months, ticktext=T.MONTH_EN, range=[0.5, 12.5], title_text="")
    ymax = np.nanmax([np.nanmax(yhat_m), np.nanmax(obs_m) if np.isfinite(np.nanmax(obs_m)) else 0, base_pred])
    fig.update_yaxes(title_text="Wi-Fi users", range=[0, ymax * 1.18 + 1])
    fig.update_layout(hovermode="x unified")
    return T.finish(fig, "Forecast for every month", f"{scope} · shaded = your month", height=360)


def fig_pred_weekdays(yhat_w, base_w, p_dow, p_month) -> go.Figure:
    labels = [T.dow_label(d) for d in range(7)]
    sel = [y if d == p_dow else None for d, y in enumerate(yhat_w)]
    oth = [None if d == p_dow else y for d, y in enumerate(yhat_w)]
    fig = go.Figure()
    fig.add_bar(x=labels, y=oth, name="Other days", marker_color=T.NEUTRAL_BAR,
                hovertemplate="Model forecast: %{y:.1f}<extra></extra>")
    fig.add_bar(x=labels, y=sel, name="Your day", marker_color=T.HUE,
                text=[f"{v:.0f}" if v is not None else "" for v in sel], textposition="outside",
                textfont=dict(size=13, color=T.INK), cliponaxis=False,
                hovertemplate="Model forecast: %{y:.1f}<extra></extra>")
    fig.add_scatter(x=labels, y=base_w, mode="markers", name="Weekday average (all months)",
                    marker=dict(symbol="line-ew-open", size=30, color=T.INK, line=dict(color=T.INK, width=2)),
                    hovertemplate="Weekday average (all months): %{y:.1f}<extra></extra>")
    fig.update_layout(barmode="overlay", barcornerradius=3, bargap=0.3)
    ymax = np.nanmax([np.nanmax(yhat_w), np.nanmax(base_w)])
    fig.update_yaxes(title_text="Wi-Fi users", range=[0, ymax * 1.2 + 1])
    fig.update_xaxes(title_text="")
    return T.finish(fig, "Forecast for every weekday", f"{T.month_label(p_month)} · blue = your day", height=360)


def fig_mae(meta: dict) -> go.Figure:
    rows = [
        ("Our model", meta["mae_model"], T.HUE),
        ("Long-run weekday average", meta["mae_baseline"], T.NEUTRAL_BAR),
        ("Last-28-days weekday average", meta["mae_baseline_recent28d_same_dow"], T.NEUTRAL_BAR),
        ("2019 same month & weekday", meta["mae_baseline_2019_same_month_dow"], T.NEUTRAL_BAR),
    ]
    fig = go.Figure(go.Bar(
        y=[r[0] for r in rows][::-1], x=[r[1] for r in rows][::-1], orientation="h",
        marker_color=[r[2] for r in rows][::-1], text=[f"{r[1]:.2f}" for r in rows][::-1],
        textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: average error %{x:.2f}<extra></extra>",
    ))
    fig.update_xaxes(title_text="Average error on the test period (lower is better)", rangemode="tozero",
                     range=[0, max(r[1] for r in rows) * 1.18])
    fig.update_yaxes(showgrid=False)
    fig.update_layout(barcornerradius=3, showlegend=False, margin=dict(l=10, r=40))
    return T.finish(fig, "How good is the model?", "Average miss per floor × group × day, after 15 Mar 2021",
                    height=300)


# ---------------------------------------------------------------- cards
def _insight_card(key: str, eyebrow: str, big: str, big_cap: str, title: str, body_html: str,
                  so_what: str, fig: go.Figure | None, fig_key: str, details: str) -> None:
    with st.container(border=True, key=key):
        left, right = st.columns([5, 6], gap="medium")
        with left:
            _md(
                f"<p class='eyebrow'>{_esc(eyebrow)}</p>"
                f"<div class='ins-fig'>{_esc(big)}</div><div class='ins-figcap'>{_esc(big_cap)}</div>"
                f"<div class='ins-title'>{_esc(title)}</div>"
                f"<p class='ins-body'>{body_html}</p>"
                f"<div class='sowhat'><b>Why it matters:</b> {so_what}</div>"
            )
            with st.expander("How we calculated this"):
                st.write(details)
        with right:
            if fig is not None:
                _plot(fig, fig_key)


# ---------------------------------------------------------------- page
def main() -> None:
    st.set_page_config(page_title="HKUST Library Wi-Fi Dashboard", layout="wide",
                       initial_sidebar_state="collapsed")
    st.html(T.CSS)

    df = load_data()
    floors_all = sorted(df["Floor"].unique().tolist())
    groups_all = sorted(df["User-Group"].unique().tolist())
    dmin, dmax = df["Date"].min().date(), df["Date"].max().date()
    n_months = df["Date"].dt.to_period("M").nunique()
    n_dates = df["Date"].nunique()

    insights = raw_insights(df)  # same calculations as compute_insights(), unrounded for display
    ivals: dict = {}
    for ins in insights:
        ivals.update(ins["values"])
    ins_by_key = {}
    for ins in insights:
        v = ins["values"]
        if "term_vs_summer_weekday_2018" in v:
            ins_by_key["term"] = ins
        elif "y2020_vs_1819_mean_drop_pct" in v:
            ins_by_key["y2020"] = ins
        elif "ug_weekend_retention" in v:
            ins_by_key["weekend"] = ins
        elif "non_hkust_weekday_share_term_vs_summer_2019" in v:
            ins_by_key["guests"] = ins

    model, meta, _means = train_models(df)

    # ---- header
    st.title("HKUST Library Wi-Fi Occupancy Explorer")
    _md(
        "<p class='lede'>When do library floors get busy, and can we predict it without tracking anyone? "
        "We used the library's public Wi-Fi counts (2018–2021) to find out.</p>"
        "<p class='team'>GenZ glasses · AISC1000B · Area 4 Predictive · Group 4B</p>"
    )
    _md("<div class='section-gap'></div>")

    tab_story, tab_explore, tab_predict, tab_about = st.tabs(
        ["What we found", "Explore the data", "Forecast", "About the data"]
    )

    # =================================================================== TAB 1: story
    with tab_story:
        c1, c2 = st.columns(2, gap="small")
        with c1.container(border=True, key="card_why_data"):
            _md(
                "<p class='eyebrow'>Why this data?</p>"
                "<p class='qhead'>A privacy-safe signal of how busy each floor is</p>"
                "<ul class='plain'>"
                "<li>HKUST Library publishes daily Wi-Fi counts for every floor and user group, free to use.</li>"
                f"<li>We used <b>{len(df):,}</b> records over <b>{n_months}</b> months "
                f"({dmin:%b %Y} – {dmax:%b %Y}): 5 floors × 7 user groups, every day.</li>"
                "<li>Only totals: <b>no names, no device IDs</b>. Nobody can be identified.</li>"
                "</ul>"
            )
        with c2.container(border=True, key="card_why_matters"):
            _md(
                "<p class='eyebrow'>Why does it matter?</p>"
                "<p class='qhead'>Libraries plan staff, hours and space around demand</p>"
                "<ul class='plain'>"
                "<li>When to staff desks and how long to stay open.</li>"
                "<li>Which floors to keep open in quiet periods like summer.</li>"
                "<li>Who uses the library when, so services fit real users, without watching anyone.</li>"
                "</ul>"
            )

        _md("<div class='section-gap'></div>")
        st.subheader("Trends you might not expect")
        _md("<p class='note'>All findings below are calculated automatically from the full 2018–2021 data. "
            "<b>They don't change with the filters</b> in the Explore tab.</p>")

        if "term" in ins_by_key:
            v = ins_by_key["term"]["values"]
            r18, r19 = v["term_vs_summer_weekday_2018"], v["term_vs_summer_weekday_2019"]
            bf = v["term_vs_summer_weekday_by_floor_2019"]
            lo, top = v["term_vs_summer_low_floor"], v["term_vs_summer_top_floor"]
            may = v["exam_proxy_may_vs_marapr_weekday_2019"]
            _insight_card(
                "card_ins1", "Trend 1 · Term vs summer",
                f"{r18:.2f}× / {r19:.2f}×", "Teaching-month weekdays vs summer weekdays, 2018 / 2019",
                "Term weekdays are 4–5× busier than summer, but exam month is not the peak",
                f"A teaching-month weekday had <b>{r18:.2f}×</b> (2018) and <b>{r19:.2f}×</b> (2019) the users of a "
                f"July–August weekday. The gap differs by floor: from <b>{bf[lo]:.2f}×</b> on {lo} to "
                f"<b>{bf[top]:.2f}×</b> on {top} (2019). Surprisingly, May 2019, exam season, was only "
                f"<b>{may:.2f}×</b> March–April. There is no exam-month rush.",
                "the busiest time is ordinary teaching weeks, not exams, so staffing should follow the "
                "teaching calendar. Summer is quiet enough to reduce hours or close floors.",
                fig_term_summer(term_summer_daily(df), r18, r19), "fig_ins1",
                ins_by_key["term"]["body_en"],
            )

        if "y2020" in ins_by_key:
            v = ins_by_key["y2020"]["values"]
            _insight_card(
                "card_ins2", "Trend 2 · The 2020 shock",
                f"{v['y2020_vs_1819_mean_drop_pct']:.1f}%", "2020 vs the same months of 2018–19 (average)",
                "In 2020 the library lost about two-thirds of its users, for most of the year",
                f"Compared with the same months in 2018–19, use fell by <b>{abs(v['y2020_vs_1819_mean_drop_pct']):.1f}%</b> "
                f"on average (median {abs(v['y2020_vs_1819_median_drop_pct']):.1f}%, total volume "
                f"{abs(v['y2020_vs_1819_total_drop_pct']):.1f}%). The worst month was "
                f"{T.MONTH_FULL[v['y2020_worst_month'] - 1]} (<b>{v['y2020_worst_month_drop_pct']:.1f}%</b>), and "
                f"<b>{v['y2020_months_below_minus50']} of 12</b> months were more than 50% down. "
                "The timing matches the campus closure and remote teaching.",
                "demand can collapse quickly and stay low for months. Patterns from normal years "
                "don't hold in a disrupted year, so any plan needs to allow for shocks.",
                fig_drop_2020(monthly_drop_2020(df)), "fig_ins2",
                ins_by_key["y2020"]["body_en"],
            )

        _md("<div class='section-gap'></div>")
        st.subheader("Hidden patterns")

        if "weekend" in ins_by_key:
            v = ins_by_key["weekend"]["values"]
            ug_y = {int(k): x for k, x in v["ug_weekend_retention_by_year"].items()}
            rpg_y = {int(k): x for k, x in v["rpg_weekend_retention_by_year"].items()}
            _insight_card(
                "card_ins3", "Pattern 1 · Who comes on weekends",
                f"{v['ug_weekend_retention']:.0%} vs {v['rpg_weekend_retention']:.0%}",
                "Weekend use as a share of weekday use, UG vs RPG (2018–21)",
                "Research postgrads keep coming at weekends; undergrads mostly don't",
                f"On weekends, undergraduates keep only <b>{v['ug_weekend_retention']:.1%}</b> of their weekday numbers, "
                f"while research postgrads keep <b>{v['rpg_weekend_retention']:.1%}</b>, and RPG is higher "
                "<b>in every year</b>. The two groups also favour different floors over the whole period "
                f"(UG: {v['ug_peak_floor']}, RPG: {v['rpg_peak_floor']}), although in 2018–19 both peaked on "
                f"{v['ug_peak_floor_1819']}.",
                "weekend opening mostly serves research students. That's useful when deciding weekend "
                "hours and which floors stay open.",
                fig_weekend(ug_y, rpg_y), "fig_ins3",
                ins_by_key["weekend"]["body_en"],
            )

        if "guests" in ins_by_key:
            v = ins_by_key["guests"]["values"]
            s18, s19 = v["non_hkust_weekday_share_term_vs_summer_2018"], v["non_hkust_weekday_share_term_vs_summer_2019"]
            sh = v["non_hkust_share_by_year"]
            _insight_card(
                "card_ins4", "Pattern 2 · Visitors in summer",
                f"{s19[0]:.2f}% → {s19[1]:.2f}%", "Visitors' share of weekday users, term → summer 2019",
                "When students leave, visitors become a bigger part of the library",
                f"Visitors (Non-HKUST) made up <b>{s18[0]:.2f}% → {s18[1]:.2f}%</b> of weekday users from term to "
                f"summer in 2018, and <b>{s19[0]:.2f}% → {s19[1]:.2f}%</b> in 2019. Visitor numbers fall in summer "
                "too, just much less than students'. The 2021 annual share "
                f"({sh.get('2021', float('nan')):.2f}%) is <b>not</b> a trend: about "
                f"{v['non_hkust_last_year_peak_month_volume_pct']:.0f}% of that year's visitors came in one month "
                f"({T.MONTH_FULL[v['non_hkust_last_year_peak_month'] - 1]}).",
                "in summer, services for outside visitors (access, signage, help desks) matter "
                "relatively more than during term.",
                fig_guests(s18, s19), "fig_ins4",
                ins_by_key["guests"]["body_en"],
            )

    # =================================================================== TAB 2: explore
    with tab_explore:
        st.subheader("Explore the data yourself")
        _md("<p class='note'>Choose dates, floors and user groups. The numbers and the three charts below update. "
            "Leaving a box empty shows nothing for it.</p>")
        with st.container(border=True, key="card_filters"):
            f1c, f2c, f3c = st.columns([2, 2, 3], gap="medium")
            with f1c:
                date_range = st.date_input("Dates", value=(dmin, dmax), min_value=dmin, max_value=dmax)
            with f2c:
                floors = st.multiselect("Floors", floors_all, default=floors_all)
            with f3c:
                groups = st.multiselect("User groups", groups_all, default=groups_all, format_func=T.group_label)

        missing = [name for name, sel in (("a floor", floors), ("a user group", groups)) if not sel]
        if missing:
            fdf = df.iloc[0:0]
        else:
            fdf = filter_df(df, date_range, floors, groups)

        if len(fdf):
            daily = fdf.groupby("Date")["User-Count"].sum()
            peak_day, peak_v = daily.idxmax(), daily.max()
            f0, f1 = fdf["Date"].min().date(), fdf["Date"].max().date()
            kpis = [
                ("Days shown", f"{fdf['Date'].nunique():,}", f"{f0:%d %b %Y} → {f1:%d %b %Y}"),
                ("Records shown", f"{len(fdf):,}", f"of {len(df):,} · 1 record = 1 floor, 1 group, 1 day"),
                ("Average per record", f"{fdf['User-Count'].mean():.1f}", "Wi-Fi users per floor, group and day"),
                ("Busiest day", f"{peak_v:,.0f}", f"{peak_day:%a %d %b %Y} · all selected floors & groups"),
            ]
        else:
            kpis = [("Days shown", "0", "—"), ("Records shown", "0", f"of {len(df):,}"),
                    ("Average per record", "—", "no data selected"), ("Busiest day", "—", "no data selected")]
        _kpi_strip(kpis)

        if missing:
            st.info("Nothing to show. Pick at least " + " and ".join(missing) + " above.")
        elif not len(fdf):
            st.info("No data for these dates. Try a wider date range.")
        else:
            c_left, c_right = st.columns([3, 2], gap="small")
            with c_left.container(border=True, key="card_heat"):
                _plot(fig_heatmap(fdf), "fig_heat")
                st.caption("Weekends are clearly quieter on every floor. Because each cell is an average per user "
                           "group, adding small groups (e.g. Alumni) lowers all numbers.")
            with c_right.container(border=True, key="card_bucket"):
                fb = fig_buckets(fdf)
                if fb is None:
                    _md("<p class='note'>This chart needs UG, RPG, TPG or Staff in the user-group filter.</p>")
                else:
                    _plot(fb, "fig_bucket")
                    st.caption("Shows UG, postgrads and staff only. Alumni, visitors and other HKUST are left out.")
            with st.container(border=True, key="card_series"):
                smooth = st.radio(
                    "View", ["Daily", "7-day average"], index=0,
                    horizontal=True, label_visibility="collapsed", key="series_res",
                ).startswith("7")
                _plot(fig_series(fdf, smooth), "fig_series")
                st.caption("Look for the summer dips, the weekly saw-tooth (weekends) and the 2020 collapse. "
                           "The dotted line marks a change in how the library counted users.")

    # =================================================================== TAB 3: forecast
    with tab_predict:
        st.subheader("Forecast: how many people should we expect?")
        _md("<p class='note'>Pick a floor, user group, day and month. The model gives a <b>typical</b> count "
            "for that combination, not an exact count for a specific date.</p>")
        with st.container(border=True, key="card_pcontrols"):
            p1, p2, p3, p4 = st.columns(4, gap="medium")
            with p1:
                p_floor = st.selectbox("Floor", floors_all, index=0)
            with p2:
                p_group = st.selectbox(
                    "User group", groups_all,
                    index=groups_all.index("UG") if "UG" in groups_all else 0,
                    format_func=T.group_label,
                )
            with p3:
                p_dow = st.selectbox("Day of week", options=list(range(7)), format_func=lambda i: T.DOW_FULL[i], index=0)
            with p4:
                p_month = st.selectbox("Month", options=list(range(1, 13)), format_func=T.month_label, index=9)

        pred = predict_user_count(model, meta, p_floor, p_group, int(p_dow), int(p_month))
        base_key = (p_floor, p_group, int(p_dow))
        train_cut = pd.Timestamp(meta["cut"])
        train_part = df[df["Date"] < train_cut]
        base_means = train_part.groupby(["Floor", "User-Group", "dow"])["User-Count"].mean()
        base_pred = float(base_means.get(base_key, train_part["User-Count"].mean()))

        _md(
            f"<p class='pred-sentence'>On a typical <b>{T.DOW_FULL[int(p_dow)]}</b> in <b>{T.month_label(int(p_month))}</b>, "
            f"the model expects about <b>{pred:.0f}</b> {_esc(T.GROUP_PHRASE.get(p_group, p_group))} "
            f"Wi-Fi users on <b>Floor {_esc(p_floor)}</b>.</p>"
        )

        with st.container(border=True, key="card_pred"):
            m1, m2, m3, m4 = st.columns(4)
            with m1.container(key="kpi_pred"):
                st.metric(
                    "Model forecast", f"{pred:.1f}",
                    delta=f"{pred - base_pred:+.1f} vs weekday average", delta_color="off",
                )
            m2.metric("Weekday average (all months)", f"{base_pred:.1f}",
                      help="Average for this floor, group and weekday over the whole training period, "
                           "ignoring the month. It's a simple comparison, not a forecast for your month.")
            m3.metric("Model's average error", f"{meta['mae_model']:.2f}",
                      help="Mean absolute error on the test period (after the split date). Lower is better.")
            m4.metric(
                "Weekday-average error", f"{meta['mae_baseline']:.2f}",
                help=f"Error of the simple weekday average on the same test period. The model's error is "
                     f"{meta['mae_baseline'] - meta['mae_model']:.2f} lower.",
            )

            months = list(range(1, 13))
            yhat_m = predict_grid(model, meta, p_floor, p_group, [int(p_dow)] * 12, months)
            s19 = seasonal_2019_means(df, meta["cut"])
            obs_m = np.array([s19.get((p_floor, p_group, m, int(p_dow)), np.nan) for m in months], dtype=float)
            yhat_w = predict_grid(model, meta, p_floor, p_group, list(range(7)), [int(p_month)] * 7)
            base_w = np.array(
                [base_means.get((p_floor, p_group, d), train_part["User-Count"].mean()) for d in range(7)], dtype=float
            )
            scope = f"Floor {p_floor} · {T.group_label(p_group)} · {T.DOW_FULL[int(p_dow)]}"
            g1, g2 = st.columns([3, 2], gap="small")
            with g1:
                _plot(fig_pred_months(yhat_m, obs_m, base_pred, int(p_month), scope), "fig_pred_months")
            with g2:
                _plot(fig_pred_weekdays(yhat_w, base_w, int(p_dow), int(p_month)), "fig_pred_weekdays")

        _md("<div class='section-gap'></div>")
        e1, e2 = st.columns([3, 2], gap="small")
        with e1.container(border=True, key="card_mae"):
            _plot(fig_mae(meta), "fig_mae")
        with e2.container(border=True, key="card_honest"):
            _md(
                "<p class='eyebrow'>Honest limits</p>"
                f"<p class='honest'>The model beats the long-run weekday average by "
                f"<b>{meta['mae_rel_improvement_pct']:.1f}%</b>. But two simple baselines do slightly better: the "
                f"last-28-days weekday average (<b>{meta['mae_baseline_recent28d_same_dow']:.2f}</b>) and the 2019 "
                f"same-month-and-weekday average (<b>{meta['mae_baseline_2019_same_month_dow']:.2f}</b>). The model has "
                "no recent-level signal, so it under-predicts the 2021 rebound.</p>"
                f"<p class='honest'>Errors stay large (test-period average {meta['hold_mean']:.0f} users vs "
                f"{meta['train_mean']:.0f} in training), so treat the forecast as a <b>typical level</b> for "
                "planning, not an exact daily number. A librarian should review any use.</p>"
            )
        with st.expander("Model details"):
            st.write(
                f"Model: {meta['model_name']} (scikit-learn). Inputs: floor, user group, day of week, month, "
                f"weekend yes/no. Trained on data before {meta['cut']} ({meta['n_train']:,} records) and "
                f"tested on data from {meta['cut']} on ({meta['n_hold']:,} records). "
                "Error = mean absolute difference between forecast and actual count on the test records."
            )

    # =================================================================== TAB 4: about
    with tab_about:
        a1, a2 = st.columns(2, gap="small")
        with a1.container(border=True, key="card_source"):
            _md(
                "<p class='eyebrow'>Source</p>"
                "<p class='qhead'>Wi-Fi Connection in HKUST Library</p>"
                f"<p class='qbody'>HKUST Library, via HKUST DataSpace · <a href='{SOURCE_URL}' target='_blank'>{DOI}</a> · "
                "free to use (CC0).</p>"
                "<ul class='plain'>"
                f"<li><b>{len(df):,}</b> cleaned records, {dmin:%d %b %Y} → {dmax:%d %b %Y} ({n_months} months, "
                f"{n_dates:,} days with data; 1–2 Aug 2018 are missing in the source).</li>"
                "<li>One record = one floor × one user group × one day.</li>"
                "<li>5 floors (1, G, LG1, LG3, LG4) · 7 user groups. Library-wide total rows were removed to avoid "
                "double counting.</li>"
                "</ul>"
            )
        with a2.container(border=True, key="card_quality"):
            _md(
                "<p class='eyebrow'>Read the numbers carefully</p>"
                "<ul class='plain'>"
                "<li><b>It's a Wi-Fi count, not a headcount.</b> People without a device, or not on Wi-Fi, "
                "aren't counted.</li>"
                "<li><b>The counting method changed on 5 Nov 2018</b> (the daily total doubled overnight; "
                "connections per user went from 3.0 to 5.3) and again in 2021. That's why we compare each "
                "year with itself.</li>"
                "<li><b>29 days are all zero</b> (12 Feb–2 Mar 2021, 12–14 Jun 2021, 25–31 Dec 2021): "
                "closure or a logging gap.</li>"
                "<li>Month groups (e.g. Feb–May = teaching) are approximations, not the official calendar.</li>"
                "</ul>"
            )
        with st.container(border=True, key="card_ethics"):
            _md(
                "<p class='eyebrow'>Ethics</p>"
                "<ul class='plain'>"
                "<li><b>Privacy by aggregation:</b> only public daily totals; no device IDs, names or sessions; "
                "not joined with any other data.</li>"
                "<li><b>Planning, not surveillance:</b> for staffing, hours and space only. Never to monitor, rank "
                "or restrict people, or to single out any user group.</li>"
                "<li><b>No over-claiming:</b> the 2020 drop matches the campus closure, but we don't claim one "
                "single cause.</li>"
                "<li><b>Honest model limits:</b> simple baselines beat our model, and we show that openly.</li>"
                "</ul>"
            )

    _md(
        f"<p class='foot'>Data: Wi-Fi Connection in HKUST Library · <a href='{SOURCE_URL}' target='_blank'>{DOI}</a> · "
        f"{len(df):,} cleaned records · {dmin} → {dmax} · GenZ glasses, AISC1000B Group 4B</p>"
    )


if __name__ == "__main__":
    main()
