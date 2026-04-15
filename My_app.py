import json
from pathlib import Path

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
DEFAULT_HORIZON_SLOTS = 6
CRIME_TYPES = ["THEFT", "BATTERY", "CRIMINAL DAMAGE", "ASSAULT", "DECEPTIVE PRACTICE"]

st.set_page_config(page_title="Chicago Crime Analytics + STGCN (ONNX)", layout="wide")
st.title("Chicago Crime Analytics and STGCN Prediction Dashboard")
st.caption("EDA + ONNX-based spatiotemporal forecasting app")


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
    fig = px.line(df_, x="Hour", y="Total_Crimes", color="Primary Type", markers=True,
                  title="Hourly Crime Pattern by Type")
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
    fig = px.line(tmp, x="Year", y="Arrest_Rate_%", color="Primary Type", markers=True,
                  title="Arrest Rate by Crime Type")
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

    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"]
    )
    return session


# =========================
# Prediction preprocess
# =========================
@st.cache_data
def load_tensor_and_meta():
    meta = safe_read_json(DATA_DIR / "meta.json")
    tensor_path = first_existing(DATA_DIR / "demo_tensor.npy", DATA_DIR / "tensor.npy")
    if tensor_path is None:
        raise FileNotFoundError("Neither demo_tensor.npy nor tensor.npy was found in artifacts/data_v2/")
    tensor = np.load(tensor_path, mmap_mode="r")
    return tensor, meta


def make_demo_input(horizon_slots: int = DEFAULT_HORIZON_SLOTS):
    tensor, meta = load_tensor_and_meta()

    lookback = int(meta["lookback"])
    val_end = int(meta.get("val_end", lookback))

    anchor_t = max(val_end, lookback)

    target_t = anchor_t + horizon_slots
    if target_t >= tensor.shape[0]:
        target_t = tensor.shape[0] - 1

    x = np.asarray(tensor[anchor_t - lookback: anchor_t], dtype=np.float32)
    y_true = np.asarray(tensor[target_t], dtype=np.float32)

    x = np.transpose(x, (2, 0, 1))
    x = np.log1p(x)
    x = np.expand_dims(x, 0)

    y_true = np.transpose(y_true, (1, 0))

    return x, y_true, {"anchor_t": anchor_t}


def preprocess_uploaded_csv(df: pd.DataFrame):
    """
    Format A:
      columns = [time_idx, grid_id, THEFT, BATTERY, CRIMINAL DAMAGE, ASSAULT, DECEPTIVE PRACTICE]

    Format B:
      a flattened numeric CSV containing exactly 5 * 180 * 1505 values
    """
    required_cols = {"time_idx", "grid_id", *CRIME_TYPES}

    if required_cols.issubset(df.columns):
        tmp = df.copy()

        tmp["time_idx"] = pd.to_numeric(tmp["time_idx"], errors="raise").astype(int)
        tmp["grid_id"] = pd.to_numeric(tmp["grid_id"], errors="raise").astype(int)

        if tmp["time_idx"].min() != 0 or tmp["time_idx"].max() != 179:
            raise ValueError("time_idx must span exactly 0..179.")
        if tmp["grid_id"].min() != 0 or tmp["grid_id"].max() != 1504:
            raise ValueError("grid_id must span exactly 0..1504.")

        expected_rows = 180 * 1505
        if len(tmp) != expected_rows:
            raise ValueError(f"Expected {expected_rows} rows, got {len(tmp)}.")

        tmp = tmp.sort_values(["time_idx", "grid_id"]).reset_index(drop=True)

        for c in CRIME_TYPES:
            tmp[c] = pd.to_numeric(tmp[c], errors="raise").astype(np.float32)

        values = tmp[CRIME_TYPES].values.reshape(180, 1505, 5)  # (L, N, C)
        x = np.transpose(values, (2, 0, 1))                     # (C, L, N)
        x = np.log1p(x)
        x = np.expand_dims(x, 0)                                # (1, C, L, N)

        return x.astype(np.float32)

    numeric = df.select_dtypes(include=[np.number])
    arr = numeric.values.flatten()

    expected = 5 * 180 * 1505
    if arr.size != expected:
        raise ValueError(
            "Unsupported CSV format. Use either:\n"
            "1) columns: time_idx, grid_id, 5 crime columns\n"
            "2) flattened numeric CSV with exactly 5*180*1505 values."
        )

    x = arr.astype(np.float32).reshape(1, 5, 180, 1505)
    x = np.log1p(x)
    return x.astype(np.float32)


# =========================
# Inference / visualization
# =========================
def run_inference(session, x_array: np.ndarray):
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    pred = session.run([output_name], {input_name: x_array})[0]
    pred = np.asarray(pred)

    # 兼容导出后常见形状
    if pred.ndim == 4:
        pred = np.squeeze(pred, axis=2) if pred.shape[2] == 1 else pred
    if pred.ndim == 3:
        pred = pred[0]

    if pred.shape != (5, 1505):
        raise ValueError(f"Unexpected ONNX output shape: {pred.shape}")

    pred_count = np.expm1(pred)
    pred_count = np.clip(pred_count, 0, None)
    return pred_count.astype(np.float32)


def get_grid_shape(meta: dict):
    n_rows = int(meta.get("n_rows", 43))
    n_cols = int(meta.get("n_cols", 35))
    if n_rows * n_cols != int(meta["n_grids"]):
        return 43, 35
    return n_rows, n_cols


def plot_prediction_summary(y_pred):
    total_by_type = y_pred.sum(axis=1)
    pred_df = pd.DataFrame({"Crime Type": CRIME_TYPES, "Predicted Count": total_by_type})
    fig = px.bar(pred_df, x="Crime Type", y="Predicted Count", color="Crime Type",
                 title="Predicted Crime Count by Type")
    st.plotly_chart(fig, use_container_width=True)


def plot_true_vs_pred_type_bar(y_pred, y_true):
    pred_total = y_pred.sum(axis=1)
    true_total = y_true.sum(axis=1)

    df = pd.DataFrame({
        "Crime Type": CRIME_TYPES * 2,
        "Count": np.concatenate([true_total, pred_total]),
        "Series": ["True"] * len(CRIME_TYPES) + ["Predicted"] * len(CRIME_TYPES),
    })
    fig = px.bar(df, x="Crime Type", y="Count", color="Series", barmode="group",
                 title="True vs Predicted Total Count by Type")
    st.plotly_chart(fig, use_container_width=True)


def plot_hotspot_heatmap(vec, meta, title):
    n_rows, n_cols = get_grid_shape(meta)
    grid_map = vec.reshape(n_rows, n_cols)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(grid_map)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.85)
    st.pyplot(fig, use_container_width=True)


def plot_top_grids(y_pred, top_k=20):
    rows = []
    for i, ctype in enumerate(CRIME_TYPES):
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
            f"Lookback = {meta.get('lookback', 'NA')} steps, "
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


def render_prediction_results(y_pred, art, source_name, y_true=None, demo_info=None):
    st.subheader(f"Prediction Results ({source_name})")
    plot_prediction_summary(y_pred)

    if demo_info is not None:
        st.caption(
            f"Demo anchor_t = {demo_info['anchor_t']}, "
            f"target_t = {demo_info['target_t']}, "
            f"lookback = {demo_info['lookback']}, "
            f"horizon_slots = {demo_info['horizon_slots']}"
        )

    if y_true is not None:
        plot_true_vs_pred_type_bar(y_pred, y_true)

    meta = art["meta"]
    crime_choice = st.selectbox("Select crime type for hotspot map", CRIME_TYPES, index=0)
    type_idx = CRIME_TYPES.index(crime_choice)

    plot_hotspot_heatmap(y_pred[type_idx], meta, f"Predicted Hotspot Heatmap: {crime_choice}")
    if y_true is not None:
        plot_hotspot_heatmap(y_true[type_idx], meta, f"True Hotspot Heatmap: {crime_choice}")

    with st.expander("Top predicted grids"):
        plot_top_grids(y_pred, top_k=20)

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
    st.header("STGCN Prediction (ONNX)")

    try:
        session = load_onnx_session()
    except Exception as e:
        st.error(f"Failed to load ONNX model: {e}")
        return

    mode = st.radio("Choose input mode", ["Demo", "CSV Upload"], horizontal=True)

    if mode == "Demo":
        st.write("Run one prepared sample on the exported ONNX STGCN model.")
        if st.button("Run Demo Prediction"):
            try:
                x_array, y_true, info = make_demo_input(DEFAULT_HORIZON_SLOTS)
                y_pred = run_inference(session, x_array)
                render_prediction_results(y_pred, art, source_name="Demo", y_true=y_true, demo_info=info)
            except Exception as e:
                st.error(f"Demo prediction failed: {e}")

        st.divider()
        st.subheader("Evaluation Preview")
        render_metrics_panel(art)

    else:
        st.write(
            "Upload a prepared sequence CSV. Recommended schema:\n"
            "`time_idx, grid_id, THEFT, BATTERY, CRIMINAL DAMAGE, ASSAULT, DECEPTIVE PRACTICE`"
        )
        uploaded_file = st.file_uploader("Upload CSV for STGCN input", type=["csv"])

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.subheader("Uploaded CSV Preview")
                st.dataframe(df.head(), use_container_width=True)

                if st.button("Run CSV Prediction"):
                    x_array = preprocess_uploaded_csv(df)
                    y_pred = run_inference(session, x_array)
                    render_prediction_results(y_pred, art, source_name="CSV Upload")
            except Exception as e:
                st.error(f"CSV prediction failed: {e}")


def render_about_page(art):
    st.header("About This App")
    st.write(
        """
        This dashboard combines:
        - EDA for Chicago crime data
        - ONNX-based STGCN forecasting
        - Demo mode and CSV upload mode
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
        page = st.radio("Go to", ["EDA", "Prediction", "About"])

    if page == "EDA":
        render_eda_page(art)
    elif page == "Prediction":
        render_prediction_page(art)
    else:
        render_about_page(art)


if __name__ == "__main__":
    main()
