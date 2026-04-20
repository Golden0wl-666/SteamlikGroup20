import json
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import onnxruntime as ort

from pandas.tseries.holiday import USFederalHolidayCalendar


# =========================
# Paths / constants
# =========================
APP_DIR = Path(__file__).resolve().parent
ART_DIR = APP_DIR / "artifacts"
DATA_DIR = ART_DIR / "data_v2"
MODEL_DIR = APP_DIR / "models"
FIG_DIR = APP_DIR / "figures"
OUTPUT_DIR = APP_DIR / "outputs"

LAT_COL, LON_COL = "Latitude", "Longitude"
CRIME_TYPES_DEFAULT = ["THEFT", "BATTERY", "CRIMINAL DAMAGE", "ASSAULT", "DECEPTIVE PRACTICE"]

# 未来预测天数
FORECAST_DAYS = 30

st.set_page_config(page_title="Chicago Crime Analytics + STGCN Forecast", layout="wide")
st.title("Chicago Crime Analytics and STGCN Prediction Dashboard")
st.caption("EDA + ONNX spatiotemporal forecasting + calendar-based EVA")


# =========================
# Generic helpers
# =========================
def first_existing(*paths: Path):
    for p in paths:
        if p.exists():
            return p
    return None


def safe_read_csv(path: Path):
    return pd.read_csv(path) if path and path.exists() else None


def safe_read_json(path: Path):
    if path and path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# =========================
# Artifact loading
# =========================
@st.cache_data
def load_artifacts():
    art = {}

    # EDA tables
    art["yearly"] = safe_read_csv(ART_DIR / "agg_yearly.csv")
    art["monthly"] = safe_read_csv(ART_DIR / "agg_monthly.csv")
    art["weekly"] = safe_read_csv(ART_DIR / "agg_weekly.csv")
    art["daily"] = safe_read_csv(ART_DIR / "agg_daily.csv")
    art["top_types"] = safe_read_csv(ART_DIR / "top_types.csv")
    art["hourly_topN"] = safe_read_csv(ART_DIR / "hourly_by_type_topN.csv")
    art["yearly_topN"] = safe_read_csv(ART_DIR / "yearly_by_type_topN.csv")
    art["arrest_yearly"] = safe_read_csv(ART_DIR / "arrest_rate_yearly.csv")
    art["arrest_yearly_topN"] = safe_read_csv(ART_DIR / "arrest_rate_yearly_topN.csv")
    art["grid"] = safe_read_csv(ART_DIR / "spatial_grid_precomputed.csv")
    art["points"] = safe_read_csv(ART_DIR / "sample_points.csv")

    # Metrics / metadata
    art["metrics_overall"] = safe_read_json(
        first_existing(APP_DIR / "metrics_overall.json", ART_DIR / "metrics_overall.json")
    )
    art["metrics_compare"] = safe_read_json(
        first_existing(APP_DIR / "metrics_compare_vs_xgboost.json", ART_DIR / "metrics_compare_vs_xgboost.json")
    )
    art["split_info"] = safe_read_json(
        first_existing(OUTPUT_DIR / "split_info.json", APP_DIR / "split_info.json", ART_DIR / "split_info.json")
    )

    # Data_v2 meta
    art["meta"] = safe_read_json(DATA_DIR / "meta.json")

    # Images
    art["images"] = {}
    image_names = [
        "metrics_by_crime_type.png",
        "accuracy_by_crime_type.png",
        "compare_stgcn_vs_xgboost.png",
        "loss_curve.png",
        "test_pred_vs_true_type0.png",
    ]
    for name in image_names:
        p = first_existing(FIG_DIR / name, ART_DIR / name)
        if p:
            art["images"][name] = p

    # Cleanup
    if art["daily"] is not None and "Date" in art["daily"].columns:
        art["daily"]["Date"] = pd.to_datetime(art["daily"]["Date"], errors="coerce")
        art["daily"] = art["daily"].dropna(subset=["Date"]).sort_values("Date")

    if art["points"] is not None:
        for c in [LAT_COL, LON_COL, "Year", "Month", "Hour"]:
            if c in art["points"].columns:
                art["points"][c] = pd.to_numeric(art["points"][c], errors="coerce")
        art["points"] = art["points"].dropna(subset=[LAT_COL, LON_COL])

    return art


# =========================
# EDA helpers
# =========================
def filter_year(df: pd.DataFrame, year_range):
    if df is None or "Year" not in df.columns:
        return df
    return df[(df["Year"] >= year_range[0]) & (df["Year"] <= year_range[1])]


def mark_holidays_daily(daily_df: pd.DataFrame):
    out = daily_df.copy()
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=out["Date"].min(), end=out["Date"].max())
    out["Is_Holiday"] = out["Date"].dt.normalize().isin(holidays)

    out["Is_Holiday_Window"] = False
    for h in holidays:
        window = pd.date_range(start=h - pd.Timedelta(days=2), end=h + pd.Timedelta(days=7))
        out.loc[out["Date"].dt.normalize().isin(window), "Is_Holiday_Window"] = True

    out["Period_Type"] = "Normal Day"
    out.loc[out["Is_Holiday_Window"], "Period_Type"] = "Holiday Window"
    out.loc[out["Is_Holiday"], "Period_Type"] = "Holiday Day"
    return out


def plot_year_trend(df_):
    fig = px.line(df_, x="Year", y="Total_Crimes", markers=True, title="Annual Crime Trend")
    st.plotly_chart(fig, use_container_width=True)


def plot_monthly(df_):
    fig = px.bar(df_.sort_values("Month"), x="Month", y="Total_Crimes", color="Month", title="Monthly Seasonality")
    st.plotly_chart(fig, use_container_width=True)


def plot_weekly(df_):
    fig = px.bar(df_.sort_values("DayNum"), x="DayOfWeek", y="Total_Crimes", color="DayOfWeek", title="Weekly Cycle")
    st.plotly_chart(fig, use_container_width=True)


def plot_top_types(df_):
    fig = px.bar(df_, x="Primary Type", y="Total_Crimes", color="Primary Type", title="Top Crime Types")
    st.plotly_chart(fig, use_container_width=True)


def plot_hourly_by_type(df_):
    fig = px.line(
        df_,
        x="Hour",
        y="Total_Crimes",
        color="Primary Type",
        markers=True,
        title="Hourly Crime Pattern by Type",
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_structure_over_time(df_):
    pivot = df_.pivot_table(index="Year", columns="Primary Type", values="Total_Crimes", fill_value=0)
    ratio = pivot.div(pivot.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ratio.plot(kind="area", stacked=True, ax=ax, alpha=0.8)
    ax.set_title("Crime Type Structure Over Time")
    ax.set_ylabel("Proportion")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    st.pyplot(fig, use_container_width=True)


def plot_arrest_rate(df_):
    tmp = df_.copy()
    tmp["Arrest_Rate_%"] = tmp["Arrest_Rate"] * 100
    fig = px.line(tmp, x="Year", y="Arrest_Rate_%", markers=True, title="Arrest Rate by Year (%)")
    st.plotly_chart(fig, use_container_width=True)


def plot_arrest_rate_by_type(df_):
    tmp = df_.copy()
    tmp["Arrest_Rate_%"] = tmp["Arrest_Rate"] * 100
    fig = px.line(
        tmp,
        x="Year",
        y="Arrest_Rate_%",
        color="Primary Type",
        markers=True,
        title="Arrest Rate by Crime Type",
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_holiday(daily_df):
    dfh = mark_holidays_daily(daily_df)
    comp = dfh.groupby("Period_Type")["Total_Crimes"].mean().reset_index()
    fig = px.bar(comp, x="Period_Type", y="Total_Crimes", title="Mean Daily Crimes: Holiday vs Normal")
    st.plotly_chart(fig, use_container_width=True)


def plot_moran(grid_df):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    hb = ax.hexbin(grid_df["z_standardized"], grid_df["lag"], gridsize=70, bins="log", mincnt=1, linewidths=0)
    fig.colorbar(hb, ax=ax, shrink=0.9).set_label("log10(count)")
    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)
    I = float(grid_df["Moran_I_overall"].iloc[0]) if "Moran_I_overall" in grid_df.columns else None
    ax.set_title("Moran Scatter" + (f" (I={I:.3f})" if I is not None else ""))
    ax.set_xlabel("Standardized cell count (z)")
    ax.set_ylabel("Spatial lag (rook mean)")
    st.pyplot(fig, use_container_width=True)


def plot_gistar(grid_df):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(grid_df["lon"], grid_df["lat"], c=grid_df["Gi_cat"], s=6)
    ax.set_title("Gi* Hotspot/Coldspot Classes")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    st.pyplot(fig, use_container_width=True)

    fig2, ax2 = plt.subplots(figsize=(7, 6))
    ax2.scatter(grid_df["lon"], grid_df["lat"], c=grid_df["Gi_z"], s=6)
    ax2.set_title("Gi* z-scores")
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    st.pyplot(fig2, use_container_width=True)


def plot_location_map(points_df, year_range, crime_filter):
    if points_df is None:
        st.info("No sample_points.csv found.")
        return
    tmp = points_df.copy()
    if "Year" in tmp.columns:
        tmp = tmp[(tmp["Year"] >= year_range[0]) & (tmp["Year"] <= year_range[1])]
    if crime_filter and "Primary Type" in tmp.columns:
        tmp = tmp[tmp["Primary Type"].isin(crime_filter)]
    if tmp.empty:
        st.warning("No points left after filters.")
        return
    fig = px.density_mapbox(
        tmp,
        lat=LAT_COL,
        lon=LON_COL,
        radius=10,
        center=dict(lat=41.8781, lon=-87.6298),
        zoom=10,
        mapbox_style="carto-positron",
        hover_data=[c for c in ["Primary Type", "Location Description"] if c in tmp.columns],
    )
    st.plotly_chart(fig, use_container_width=True)


# =========================
# ONNX loading
# =========================
@st.cache_resource
def load_onnx_session():
    onnx_path = MODEL_DIR / "stgcn_best.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(f"Missing ONNX model: {onnx_path}")

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    return session


# =========================
# Tensor / meta loading
# =========================
@st.cache_data
def load_tensor_and_meta(mode="Demo"):
    meta = safe_read_json(DATA_DIR / "meta.json")

    if mode == "Demo":
        tensor_path = DATA_DIR / "demo_tensor.npy"
    else:
        tensor_path = DATA_DIR / "tensor.npy"

    if not tensor_path.exists():
        raise FileNotFoundError(f"Missing tensor file: {tensor_path}")

    tensor = np.load(tensor_path, mmap_mode="r")
    return tensor, meta


def get_slots_per_day(meta: dict) -> int:
    return int(meta.get("slots_per_day", 6)) if meta else 6


def get_model_lookback_steps(meta: dict) -> int:
    # 优先取预处理里保存的 lookback_steps
    if meta and "lookback_steps" in meta:
        return int(meta["lookback_steps"])

    # 兼容旧 meta
    if meta and "lookback" in meta:
        return int(meta["lookback"])

    # 兜底：你当前 ONNX 模型固定输入要求
    return 180


def get_crime_types(meta: dict):
    if meta and "crime_types" in meta and isinstance(meta["crime_types"], list):
        return meta["crime_types"]
    return CRIME_TYPES_DEFAULT


# =========================
# Prediction helpers
# =========================
def run_inference(session, x_array: np.ndarray):
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    pred = session.run([output_name], {input_name: x_array.astype(np.float32)})[0]
    pred = np.asarray(pred)

    # 兼容常见导出形状
    if pred.ndim == 4 and pred.shape[2] == 1:
        pred = np.squeeze(pred, axis=2)
    if pred.ndim == 3:
        pred = pred[0]

    # 预期: (C, N)
    if pred.ndim != 2:
        raise ValueError(f"Unexpected ONNX output ndim: {pred.ndim}, shape={pred.shape}")

    pred_count = np.expm1(pred)
    pred_count = np.clip(pred_count, 0, None)
    return pred_count.astype(np.float32)


def prepare_model_input(window_lnc: np.ndarray) -> np.ndarray:
    """
    window_lnc: (L, N, C)
    return: (1, C, L, N)
    """
    x = np.asarray(window_lnc, dtype=np.float32)
    x = np.transpose(x, (2, 0, 1))   # (C, L, N)
    x = np.log1p(x)
    x = np.expand_dims(x, 0)         # (1, C, L, N)
    return x.astype(np.float32)


def recursive_forecast_daily(session, tensor, meta, forecast_days=FORECAST_DAYS):
    """
    tensor: (T, N, C)
    输出:
      daily_preds: (forecast_days, C, N)
      info: dict
    """
    slots_per_day = get_slots_per_day(meta)
    lookback_steps = get_model_lookback_steps(meta)

    if tensor.shape[0] < lookback_steps:
        raise ValueError(
            f"Tensor has only {tensor.shape[0]} steps, but model needs {lookback_steps} lookback steps."
        )

    horizon_steps = forecast_days * slots_per_day

    # 用最后 lookback_steps 作为输入窗口
    window = np.asarray(tensor[-lookback_steps:], dtype=np.float32)  # (L, N, C)
    step_preds = []

    for _ in range(horizon_steps):
        x_input = prepare_model_input(window)        # (1, C, L, N)
        y_pred = run_inference(session, x_input)     # (C, N)
        step_preds.append(y_pred)

        next_step = np.transpose(y_pred, (1, 0))     # (N, C)
        window = np.concatenate([window[1:], next_step[None, :, :]], axis=0)

    step_preds = np.asarray(step_preds, dtype=np.float32)  # (H, C, N)
    daily_preds = step_preds.reshape(forecast_days, slots_per_day, step_preds.shape[1], step_preds.shape[2]).sum(axis=1)

    info = {
        "lookback_steps": lookback_steps,
        "slots_per_day": slots_per_day,
        "forecast_days": forecast_days,
        "horizon_steps": horizon_steps,
        "tensor_steps": int(tensor.shape[0]),
    }
    return daily_preds, info


@st.cache_data
def precompute_daily_predictions(mode, forecast_days=FORECAST_DAYS):
    tensor, meta = load_tensor_and_meta(mode)
    session = load_onnx_session()
    daily_preds, info = recursive_forecast_daily(session, tensor, meta, forecast_days=forecast_days)
    return daily_preds, meta, info


def build_forecast_dates(meta: dict, forecast_days=FORECAST_DAYS):
    """
    从 meta 里的 end_date 往后推。
    如果没有 end_date，就从今天开始。
    """
    if meta and "end_date" in meta:
        last_real_day = pd.Timestamp(meta["end_date"])
        start_day = last_real_day + pd.Timedelta(days=1)
    else:
        start_day = pd.Timestamp(datetime.today().date())

    return [start_day + pd.Timedelta(days=i) for i in range(forecast_days)]


# =========================
# Visualization
# =========================
def get_grid_shape(meta: dict):
    n_rows = int(meta.get("n_rows", 43))
    n_cols = int(meta.get("n_cols", 35))
    if meta and "n_grids" in meta and n_rows * n_cols != int(meta["n_grids"]):
        return 43, 35
    return n_rows, n_cols


def plot_prediction_summary(y_pred, crime_types):
    total_by_type = y_pred.sum(axis=1)
    pred_df = pd.DataFrame({"Crime Type": crime_types, "Predicted Count": total_by_type})
    fig = px.bar(pred_df, x="Crime Type", y="Predicted Count", color="Crime Type",
                 title="Predicted Daily Crime Count by Type")
    st.plotly_chart(fig, use_container_width=True)


def plot_hotspot_heatmap(vec, meta, title):
    n_rows, n_cols = get_grid_shape(meta)
    grid_map = vec.reshape(n_rows, n_cols)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(grid_map)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.85)
    st.pyplot(fig, use_container_width=True)


def plot_top_grids(y_pred, crime_types, top_k=20):
    rows = []
    for i, ctype in enumerate(crime_types):
        vals = y_pred[i]
        top_idx = np.argsort(vals)[::-1][:top_k]
        for rank, idx in enumerate(top_idx, start=1):
            rows.append({
                "Crime Type": ctype,
                "Rank": rank,
                "Grid Index": int(idx),
                "Predicted Value": float(vals[idx]),
            })
    out = pd.DataFrame(rows)
    st.dataframe(out, use_container_width=True)


def render_metrics_panel(art):
    st.subheader("Model Evaluation")

    metrics_overall = art.get("metrics_overall")
    metrics_compare = art.get("metrics_compare")
    split_info = art.get("split_info")
    meta = art.get("meta")

    c1, c2, c3 = st.columns(3)
    if metrics_overall:
        c1.metric("Avg Test MAE", f"{metrics_overall.get('avg_test_mae', 0):.4f}")
        c2.metric("Avg Test RMSE", f"{metrics_overall.get('avg_test_rmse', 0):.4f}")
        c3.metric("Avg Test Accuracy", f"{metrics_overall.get('avg_test_acc', 0):.4f}")

    if split_info and isinstance(split_info, dict) and "fields_used" in split_info:
        st.info(
            f"Lookback = {split_info['fields_used'].get('lookback', 'NA')} steps, "
            f"Grids = {split_info['fields_used'].get('n_grids', 'NA')}, "
            f"Crime types = {split_info['fields_used'].get('n_types', 'NA')}."
        )
    elif meta:
        st.info(
            f"Tensor steps = {meta.get('n_steps', 'NA')}, "
            f"Grids = {meta.get('n_grids', 'NA')}, "
            f"Crime types = {meta.get('n_types', 'NA')}."
        )

    if metrics_compare:
        with st.expander("STGCN vs XGBoost comparison"):
            st.json(metrics_compare)

    imgs = art.get("images", {})
    ordered = [
        "metrics_by_crime_type.png",
        "accuracy_by_crime_type.png",
        "compare_stgcn_vs_xgboost.png",
        "loss_curve.png",
        "test_pred_vs_true_type0.png",
    ]
    for name in ordered:
        if name in imgs:
            st.image(str(imgs[name]), caption=name)


def render_prediction_results(y_pred, art, source_name, crime_types):
    st.subheader(source_name)

    plot_prediction_summary(y_pred, crime_types)

    meta = art["meta"] if art.get("meta") else {}
    crime_choice = st.selectbox("Select crime type for hotspot map", crime_types, index=0)
    type_idx = crime_types.index(crime_choice)

    plot_hotspot_heatmap(y_pred[type_idx], meta, f"Predicted Daily Hotspot: {crime_choice}")

    with st.expander("Top predicted grids"):
        plot_top_grids(y_pred, crime_types, top_k=20)

    st.divider()
    render_metrics_panel(art)


# =========================
# Page renderers
# =========================
def render_eda_page(art):
    st.header("EDA Dashboard")

    needed = [
        "yearly", "monthly", "weekly", "daily",
        "top_types", "hourly_topN", "yearly_topN",
        "arrest_yearly", "arrest_yearly_topN", "grid"
    ]
    missing = [k for k in needed if art.get(k) is None]
    if missing:
        st.warning(f"Missing EDA artifacts: {missing}")
        return

    options = ["Time", "Category", "Location", "Arrest"]
    selection = st.pills("Which aspect do you intend to know about?", options, selection_mode="multi")
    selected = set(selection)

    if not selected:
        st.info("Select at least one pill to start.")
        return

    year_min = int(art["yearly"]["Year"].min())
    year_max = int(art["yearly"]["Year"].max())
    year_range = (year_min, year_max)
    if "Time" in selected:
        year_range = st.slider("Select Year Range", year_min, year_max, (year_min, year_max))

    yearly = filter_year(art["yearly"], year_range)
    monthly = art["monthly"]
    weekly = art["weekly"]
    daily = art["daily"]
    hourly_topN = art["hourly_topN"]
    yearly_topN = filter_year(art["yearly_topN"], year_range)
    arrest_yearly = filter_year(art["arrest_yearly"], year_range)
    arrest_yearly_topN = filter_year(art["arrest_yearly_topN"], year_range)
    grid = art["grid"]
    points = art["points"]

    crime_options = sorted(art["top_types"]["Primary Type"].unique().tolist())
    crime_filter = st.multiselect("Crime types", crime_options, default=crime_options[:5])

    if crime_filter:
        hourly_topN_f = hourly_topN[hourly_topN["Primary Type"].isin(crime_filter)]
        yearly_topN_f = yearly_topN[yearly_topN["Primary Type"].isin(crime_filter)]
        arrest_yearly_topN_f = arrest_yearly_topN[arrest_yearly_topN["Primary Type"].isin(crime_filter)]
    else:
        hourly_topN_f, yearly_topN_f, arrest_yearly_topN_f = hourly_topN, yearly_topN, arrest_yearly_topN

    if len(selected) == 1:
        only = next(iter(selected))

        if only == "Time":
            t1, t2, t3 = st.tabs(["Interactive basics", "Professional add-ons", "Holiday"])
            with t1:
                plot_year_trend(yearly)
                plot_monthly(monthly)
                plot_weekly(weekly)
            with t2:
                plot_hourly_by_type(hourly_topN_f)
                plot_structure_over_time(yearly_topN_f)
            with t3:
                plot_holiday(daily)

        elif only == "Category":
            c1, c2 = st.tabs(["Top types", "Structure & hourly"])
            with c1:
                plot_top_types(art["top_types"])
            with c2:
                plot_hourly_by_type(hourly_topN_f)
                plot_structure_over_time(yearly_topN_f)

        elif only == "Arrest":
            a1, a2 = st.tabs(["Overall arrest rate", "By type"])
            with a1:
                plot_arrest_rate(arrest_yearly)
            with a2:
                plot_arrest_rate_by_type(arrest_yearly_topN_f)

        elif only == "Location":
            l1, l2 = st.tabs(["Interactive map", "Spatial statistics"])
            with l1:
                plot_location_map(points, year_range, crime_filter)
            with l2:
                plot_moran(grid)
                plot_gistar(grid)

    else:
        t1, t2 = st.tabs(["Interactive overview", "Professional overview"])
        with t1:
            plot_year_trend(yearly)
            plot_top_types(art["top_types"])
            plot_location_map(points, year_range, crime_filter)
            plot_arrest_rate(arrest_yearly)
        with t2:
            plot_structure_over_time(yearly_topN_f)
            plot_moran(grid)
            plot_gistar(grid)


def render_prediction_page(art):
    st.header("Forecast Calendar")

    try:
        daily_preds, meta, info = precompute_daily_predictions(FORECAST_DAYS)
    except Exception as e:
        st.error(f"Failed to precompute future predictions: {e}")
        return

    crime_types = get_crime_types(meta)
    forecast_dates = build_forecast_dates(meta, FORECAST_DAYS)

    st.info(
        f"Using the latest {info['lookback_steps']} time steps as model input, "
        f"rolling forward {info['forecast_days']} days "
        f"({info['horizon_steps']} slot-level forecasts aggregated to daily results)."
    )

    # 日期选择
    date_options = list(range(len(forecast_dates)))
    selected_idx = st.selectbox(
        "Select forecast date",
        options=date_options,
        format_func=lambda i: pd.Timestamp(forecast_dates[i]).strftime("%Y-%m-%d"),
    )

    selected_date = pd.Timestamp(forecast_dates[selected_idx])
    selected_pred = daily_preds[selected_idx]  # (C, N)

    # 日期条
    with st.expander("Available forecast window"):
        df_dates = pd.DataFrame({
            "Index": np.arange(len(forecast_dates)),
            "Date": [pd.Timestamp(d).strftime("%Y-%m-%d") for d in forecast_dates]
        })
        st.dataframe(df_dates, use_container_width=True, hide_index=True)

    render_prediction_results(
        selected_pred,
        art,
        source_name=f"Predicted crime distribution for {selected_date.strftime('%Y-%m-%d')}",
        crime_types=crime_types
    )


def render_about_page(art):
    st.header("About This App")
    st.write(
        """
        This dashboard combines:
        - EDA for Chicago crime data
        - ONNX-based STGCN forecasting
        - Calendar-based future daily prediction
        - Static model evaluation artifacts
        """
    )
    if art.get("meta"):
        with st.expander("meta.json"):
            st.json(art["meta"])
    if art.get("split_info"):
        with st.expander("split_info.json"):
            st.json(art["split_info"])
    if art.get("metrics_overall"):
        with st.expander("metrics_overall.json"):
            st.json(art["metrics_overall"])


# =========================
# Main
# =========================
def main():
    art = load_artifacts()

    with st.sidebar:
        st.header("Navigation")
        page = st.radio("Go to", ["EDA", "Forecast", "About"])

    if page == "EDA":
        render_eda_page(art)
    elif page == "Forecast":
        render_prediction_page(art)
    else:
        render_about_page(art)


if __name__ == "__main__":
    main()
