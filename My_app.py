import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import torch

from pandas.tseries.holiday import USFederalHolidayCalendar

# =========================
# Optional model import
# =========================
# TODO:
# 1) 把你们自己的模型定义文件放到 model/model_def.py
# 2) 确认类名是否是 STGCNGraphConv 或 STGCNChebGraphConv
# 3) 确认 layers.py 已存在于 model/ 下
try:
    from model.model_def import STGCNGraphConv
except Exception:
    STGCNGraphConv = None


# =========================
# Basic config
# =========================
LAT_COL, LON_COL = "Latitude", "Longitude"
ART_DIR = "artifacts"
MODEL_DIR = "models"

st.set_page_config(page_title="Chicago Crime EDA + STGCN Demo", layout="wide")
st.title("Chicago Crime Analytics and STGCN Prediction Dashboard")
st.caption("EDA + proof-of-concept deployment for crime forecasting")


# =========================
# Artifact loaders
# =========================
@st.cache_data
def load_artifacts(base: str = ART_DIR):
    art = {}

    def safe_read_csv(path):
        return pd.read_csv(path) if os.path.exists(path) else None

    def safe_read_json(path):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    # EDA artifacts
    art["yearly"] = safe_read_csv(os.path.join(base, "agg_yearly.csv"))
    art["monthly"] = safe_read_csv(os.path.join(base, "agg_monthly.csv"))
    art["weekly"] = safe_read_csv(os.path.join(base, "agg_weekly.csv"))
    art["daily"] = safe_read_csv(os.path.join(base, "agg_daily.csv"))
    art["top_types"] = safe_read_csv(os.path.join(base, "top_types.csv"))
    art["hourly_topN"] = safe_read_csv(os.path.join(base, "hourly_by_type_topN.csv"))
    art["yearly_topN"] = safe_read_csv(os.path.join(base, "yearly_by_type_topN.csv"))
    art["arrest_yearly"] = safe_read_csv(os.path.join(base, "arrest_rate_yearly.csv"))
    art["arrest_yearly_topN"] = safe_read_csv(os.path.join(base, "arrest_rate_yearly_topN.csv"))
    art["grid"] = safe_read_csv(os.path.join(base, "spatial_grid_precomputed.csv"))
    art["points"] = safe_read_csv(os.path.join(base, "sample_points.csv"))

    # Metrics / metadata
    art["split_info"] = safe_read_json(os.path.join(base, "split_info.json"))
    art["metrics_compare"] = safe_read_json(os.path.join(base, "metrics_compare_vs_xgboost.json"))
    art["metrics_overall"] = safe_read_json(os.path.join(base, "metrics_overall.json"))

    # Images
    image_names = [
        "metrics_by_crime_type.png",
        "accuracy_by_crime_type.png",
        "test_pred_vs_true_type0.png",
        "compare_stgcn_vs_xgboost.png",
        "loss_curve.png",
    ]
    art["images"] = {}
    for name in image_names:
        p = os.path.join(base, name)
        if os.path.exists(p):
            art["images"][name] = p

    # daily datetime cleanup
    if art["daily"] is not None and "Date" in art["daily"].columns:
        art["daily"]["Date"] = pd.to_datetime(art["daily"]["Date"], errors="coerce")
        art["daily"] = art["daily"].dropna(subset=["Date"]).sort_values("Date")

    # point cleanup
    if art["points"] is not None:
        for c in [LAT_COL, LON_COL, "Year", "Month", "Hour"]:
            if c in art["points"].columns:
                art["points"][c] = pd.to_numeric(art["points"][c], errors="coerce")
        art["points"] = art["points"].dropna(subset=[LAT_COL, LON_COL])

    return art


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


# =========================
# Plot helpers (EDA)
# =========================
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
        lat=LAT_COL, lon=LON_COL,
        radius=10,
        center=dict(lat=41.8781, lon=-87.6298),
        zoom=10,
        mapbox_style="carto-positron",
        hover_data=[c for c in ["Primary Type", "Location Description"] if c in tmp.columns]
    )
    st.plotly_chart(fig, use_container_width=True)


# =========================
# STGCN config
# =========================
def build_model_config():
    """
    TODO:
    用你们训练时真实的 args / blocks / n_vertex 替换这里。
    这里先给一个占位写法。
    """
    class Args:
        Kt = 3
        Ks = 3
        n_his = 180
        act_func = "glu"
        graph_conv_type = "graph_conv"
        gso = None
        enable_bias = True
        droprate = 0.3

    args = Args()

    # TODO: 用你们训练时真实 blocks 替换
    blocks = [
        [5],
        [64, 16, 64],
        [64, 16, 64],
        [128],
        [5]
    ]

    n_vertex = 1505
    return args, blocks, n_vertex


@st.cache_resource
def load_stgcn_model():
    if STGCNGraphConv is None:
        return None, "Model definition import failed. Please check model/model_def.py and model/layers.py."

    model_path = os.path.join(MODEL_DIR, "stgcn_best.pt")
    if not os.path.exists(model_path):
        return None, f"Model file not found: {model_path}"

    try:
        args, blocks, n_vertex = build_model_config()
        model = STGCNGraphConv(args, blocks, n_vertex)

        state = torch.load(model_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            model.load_state_dict(state["state_dict"], strict=False)
        else:
            model.load_state_dict(state, strict=False)

        model.eval()
        return model, None
    except Exception as e:
        return None, f"Model loading failed: {e}"


# =========================
# Input preprocess
# =========================
def preprocess_demo_input():
    """
    Demo 输入推荐保存成 models/demo_input.npy
    shape 推荐: [1, 5, 180, 1505]
    """
    demo_path = os.path.join(MODEL_DIR, "demo_input.npy")
    if not os.path.exists(demo_path):
        raise FileNotFoundError("models/demo_input.npy not found")

    x = np.load(demo_path)
    if x.ndim != 4:
        raise ValueError(f"demo_input.npy must be 4D, got shape={x.shape}")

    return torch.tensor(x, dtype=torch.float32)


def preprocess_uploaded_csv(df: pd.DataFrame):
    """
    TODO:
    这里必须替换成你们真实的 CSV -> tensor 逻辑。

    当前假设：
    - CSV 内只包含数值
    - 总元素数量 = 5 * 180 * 1505
    - reshape 到 [1, 5, 180, 1505]
    """
    arr = df.select_dtypes(include=[np.number]).values.flatten()

    expected = 5 * 180 * 1505
    if arr.size != expected:
        raise ValueError(
            f"CSV numeric size mismatch. Expected {expected} values for shape [1,5,180,1505], got {arr.size}."
        )

    x = arr.reshape(1, 5, 180, 1505)
    return torch.tensor(x, dtype=torch.float32)


# =========================
# Inference + postprocess
# =========================
def run_inference(model, x_tensor):
    with torch.no_grad():
        y_pred = model(x_tensor)

    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    return y_pred


def normalize_output_shape(y_pred):
    """
    统一处理输出维度，尽量变成 [5, N]
    这里只做宽松兼容。
    """
    y = np.array(y_pred)

    if y.ndim == 4:
        # [B, C, T, N]
        y = y[0]
        if y.shape[1] == 1:
            y = y[:, 0, :]
        else:
            y = y[:, -1, :]
    elif y.ndim == 3:
        # [B, C, N]
        y = y[0]
    else:
        raise ValueError(f"Unexpected output shape: {y.shape}")

    return y  # [C, N]


def plot_prediction_summary(y_2d, crime_labels):
    total_by_type = y_2d.sum(axis=1)

    pred_df = pd.DataFrame({
        "Crime Type": crime_labels,
        "Predicted Count": total_by_type
    })

    fig = px.bar(pred_df, x="Crime Type", y="Predicted Count", color="Crime Type",
                 title="Predicted Crime Count by Type")
    st.plotly_chart(fig, use_container_width=True)


def plot_prediction_heatmap(y_2d, selected_type_idx, crime_labels):
    vec = y_2d[selected_type_idx]

    # 尝试自动拼成近似方图
    n = vec.shape[0]
    side = int(np.sqrt(n))
    if side * side == n:
        grid_map = vec.reshape(side, side)
    else:
        # 如果不是完美平方，做 padding
        side = int(np.ceil(np.sqrt(n)))
        padded = np.zeros(side * side)
        padded[:n] = vec
        grid_map = padded.reshape(side, side)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(grid_map)
    ax.set_title(f"Predicted Hotspot Heatmap: {crime_labels[selected_type_idx]}")
    fig.colorbar(im, ax=ax, shrink=0.85)
    st.pyplot(fig, use_container_width=True)


def plot_prediction_table(y_2d, crime_labels, top_k=20):
    rows = []
    for i, ctype in enumerate(crime_labels):
        vals = y_2d[i]
        top_idx = np.argsort(vals)[::-1][:top_k]
        for rank, idx in enumerate(top_idx, start=1):
            rows.append({
                "Crime Type": ctype,
                "Rank": rank,
                "Grid Index": int(idx),
                "Predicted Value": float(vals[idx])
            })
    out = pd.DataFrame(rows)
    st.dataframe(out, use_container_width=True)


# =========================
# Results / evaluation page
# =========================
def render_metrics_panel(art):
    st.subheader("Model Evaluation")

    metrics_overall = art.get("metrics_overall")
    metrics_compare = art.get("metrics_compare")
    split_info = art.get("split_info")

    c1, c2, c3 = st.columns(3)
    if metrics_overall:
        c1.metric("Avg Test MAE", f"{metrics_overall.get('avg_test_mae', 0):.4f}")
        c2.metric("Avg Test RMSE", f"{metrics_overall.get('avg_test_rmse', 0):.4f}")
        c3.metric("Avg Test Accuracy", f"{metrics_overall.get('avg_test_acc', 0):.4f}")

    if split_info:
        st.info(
            f"Lookback = {split_info['fields_used']['lookback']} steps, "
            f"Grids = {split_info['fields_used']['n_grids']}, "
            f"Crime types = {split_info['fields_used']['n_types']}, "
            f"Slots/day = {split_info['notes']['slots_per_day']}."
        )

    if metrics_compare:
        st.write("Comparison against XGBoost reference:")
        st.json(metrics_compare)

    imgs = art.get("images", {})
    show_list = [
        "metrics_by_crime_type.png",
        "accuracy_by_crime_type.png",
        "compare_stgcn_vs_xgboost.png",
        "loss_curve.png",
        "test_pred_vs_true_type0.png",
    ]
    for name in show_list:
        if name in imgs:
            st.image(imgs[name], caption=name)


def render_prediction_results(y_pred, art, source_name="Demo"):
    st.subheader(f"Prediction Results ({source_name})")

    crime_labels = ["THEFT", "BATTERY", "CRIMINAL DAMAGE", "ASSAULT", "DECEPTIVE PRACTICE"]
    y_2d = normalize_output_shape(y_pred)

    if y_2d.shape[0] != len(crime_labels):
        st.warning(f"Expected 5 crime channels, got shape={y_2d.shape}. Please check output mapping.")

    plot_prediction_summary(y_2d, crime_labels)

    type_choice = st.selectbox("Select crime type for hotspot view", crime_labels, index=0)
    type_idx = crime_labels.index(type_choice)
    plot_prediction_heatmap(y_2d, type_idx, crime_labels)

    with st.expander("Top predicted grids"):
        plot_prediction_table(y_2d, crime_labels, top_k=20)

    st.divider()
    render_metrics_panel(art)


# =========================
# EDA page
# =========================
def render_eda_page(art):
    st.header("EDA Dashboard")

    needed = ["yearly", "monthly", "weekly", "daily", "top_types", "hourly_topN", "yearly_topN", "arrest_yearly", "arrest_yearly_topN", "grid"]
    missing_files = [k for k in needed if art.get(k) is None]
    if missing_files:
        st.error(f"Missing required EDA artifacts: {missing_files}")
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
        st.caption("Combined view")
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


# =========================
# Prediction page
# =========================
def render_prediction_page(art):
    st.header("STGCN Prediction Demo")

    model, err = load_stgcn_model()
    if err:
        st.error(err)
        st.stop()

    mode = st.radio("Choose prediction input mode", ["Demo", "CSV Upload"], horizontal=True)

    if mode == "Demo":
        st.write("Use a pre-saved demo tensor to run the model and display the result page.")
        if st.button("Run Demo Prediction"):
            try:
                x_tensor = preprocess_demo_input()
                y_pred = run_inference(model, x_tensor)
                render_prediction_results(y_pred, art, source_name="Demo")
            except Exception as e:
                st.error(f"Demo prediction failed: {e}")

        st.divider()
        st.subheader("Evaluation Page")
        st.write("Even without re-running, you can directly review the model evaluation figures below.")
        render_metrics_panel(art)

    else:
        st.write("Upload a CSV and reuse the same result page and evaluation layout.")
        uploaded_file = st.file_uploader("Upload CSV for STGCN input", type=["csv"])

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.subheader("Uploaded CSV Preview")
                st.dataframe(df.head(), use_container_width=True)

                if st.button("Run CSV Prediction"):
                    x_tensor = preprocess_uploaded_csv(df)
                    y_pred = run_inference(model, x_tensor)
                    render_prediction_results(y_pred, art, source_name="CSV Upload")

            except Exception as e:
                st.error(f"CSV prediction failed: {e}")


# =========================
# Main app
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
        st.header("About This App")
        st.write("""
        This dashboard combines:
        - EDA for Chicago crime data
        - STGCN-based spatio-temporal prediction
        - Static evaluation figures and metrics
        - Demo mode and CSV upload mode
        """)

        if art.get("split_info"):
            st.json(art["split_info"])

        if art.get("metrics_overall"):
            st.json(art["metrics_overall"])


if __name__ == "__main__":
    main()
