import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats as scipy_stats

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Visual Recognition Task",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Visual Recognition Task")
st.markdown("Upload a CSV with columns: `subject`, `day`, `trainingtype`, `accuracy`, `RT`")

# ── Color palette ─────────────────────────────────────────────────────────────
DAY_COLORS   = {1: "#4C72B0", 2: "#DD8452", 3: "#55A868", 4: "#C44E52"}
TRAIN_COLORS = px.colors.qualitative.Set2

RT_MIN, RT_MAX = 100, 2000   # outlier window (ms)

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    rt_min = st.number_input("RT outlier lower bound (ms)", value=RT_MIN, step=10)
    rt_max = st.number_input("RT outlier upper bound (ms)", value=RT_MAX, step=10)
    rolling_win_rt  = st.slider("Rolling window — RT (trials)", 3, 30, 7)
    rolling_win_acc = st.slider("Rolling window — Accuracy (trials)", 3, 30, 15)
    st.markdown("---")
    st.caption("Built for the lab 💡")

# ── File upload ───────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload CSV file(s)",
    type="csv",
    accept_multiple_files=True,
    help="Columns required: subject, day, trainingtype, accuracy, RT"
)

SAMPLE_PATH = "data/visual_recognition_clean.csv"  
df_loaded = pd.read_csv(SAMPLE_PATH)

# if not uploaded:
#     st.info("📂 Showing sample data. Upload your own CSV to analyse it.")
    
# else:
#     df_loaded = pd.concat([pd.read_csv(f) for f in uploaded], ignore_index=True)

# ── Load & validate ───────────────────────────────────────────────────────────
REQUIRED = {"subject", "day", "trainingtype", "accuracy", "rt"}

@st.cache_data
def load_files(files):
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df.columns = df.columns.str.strip().str.lower()
        missing = REQUIRED - set(df.columns)
        if missing:
            st.error(f"**{f.name}** is missing columns: {missing}")
            return None
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    combined["rt"]       = pd.to_numeric(combined["rt"], errors="coerce")
    combined["accuracy"] = pd.to_numeric(combined["accuracy"], errors="coerce")
    combined["day"]      = combined["day"].astype(int)
    return combined

raw = load_files(uploaded)
if raw is None:
    st.stop()

# Apply RT filter
df_clean = raw[(raw["rt"] >= rt_min) & (raw["rt"] <= rt_max)].copy()

# ── Subject selector ──────────────────────────────────────────────────────────
all_subjects = sorted(raw["subject"].unique())
selected_subjects = st.multiselect(
    "Select subject(s) to display",
    options=all_subjects,
    default=all_subjects[:1],
    help="Select multiple subjects to enable cross-subject comparisons"
)

if not selected_subjects:
    st.warning("Please select at least one subject.")
    st.stop()

df = df_clean[df_clean["subject"].isin(selected_subjects)].copy()
df_raw = raw[raw["subject"].isin(selected_subjects)].copy()

trainings = sorted(df["trainingtype"].unique())
days      = sorted(df["day"].unique())

train_color_map = {t: TRAIN_COLORS[i % len(TRAIN_COLORS)] for i, t in enumerate(trainings)}
day_color_map   = {d: DAY_COLORS.get(d, "#888888") for d in days}

# ── Helper ─────────────────────────────────────────────────────────────────────
def section(title, icon=""):
    st.markdown(f"---\n## {icon} {title}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Summary stats
# ══════════════════════════════════════════════════════════════════════════════
section("Summary Statistics", "📊")

summary = df.groupby(["subject", "day", "trainingtype"]).agg(
    n_trials  =("rt",       "count"),
    mean_acc  =("accuracy", "mean"),
    sd_acc    =("accuracy", "std"),
    mean_rt   =("rt",       "mean"),
    sd_rt     =("rt",       "std"),
    median_rt =("rt",       "median"),
).round(3).reset_index()

st.dataframe(
    summary.style.format({
        "mean_acc": "{:.3f}", "sd_acc": "{:.3f}",
        "mean_rt": "{:.1f}",  "sd_rt": "{:.1f}", "median_rt": "{:.1f}"
    }),
    use_container_width=True,
    hide_index=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Distributions (RT & Accuracy)
# ══════════════════════════════════════════════════════════════════════════════
section("Distributions", "📈")

col1, col2 = st.columns(2)

# RT distribution

st.subheader("RT distribution by training type")
fig = go.Figure()
for t in trainings:
    sub = df[df["trainingtype"] == t]["rt"].dropna()
    fig.add_trace(go.Histogram(
        x=sub, name=t, opacity=0.6,
        marker_color=train_color_map[t],
        nbinsx=40, histnorm="probability density"
    ))
fig.update_layout(barmode="overlay", xaxis_title="RT (ms)", yaxis_title="Density", legend_title="Training")
st.plotly_chart(fig, use_container_width=True)

# Accuracy distribution (violin)
# with col2:
#     st.subheader("Accuracy distribution by training type")
#     fig = go.Figure()
#     for t in trainings:
#         sub = df[df["trainingtype"] == t]["accuracy"].dropna()
#         fig.add_trace(go.Violin(
#             y=sub, name=t, box_visible=True, meanline_visible=True,
#             fillcolor=train_color_map[t], opacity=0.7, line_color="black"
#         ))
#     fig.update_layout(yaxis_title="Accuracy (proportion correct)", showlegend=True)
#     st.plotly_chart(fig, use_container_width=True)

# RT histogram split by Day
st.subheader("RT distribution by training type — split by day")
if len(days) > 0:
    fig = make_subplots(rows=1, cols=len(days), shared_yaxes=True,
                        subplot_titles=[f"Day {d}" for d in days])
    for i, day in enumerate(days):
        day_df = df[df["day"] == day]
        for t in trainings:
            sub = day_df[day_df["trainingtype"] == t]["rt"].dropna()
            fig.add_trace(go.Histogram(
                x=sub, name=t, opacity=0.55,
                marker_color=train_color_map[t],
                nbinsx=30, histnorm="probability density",
                showlegend=(i == 0)
            ), row=1, col=i+1)
    fig.update_layout(barmode="overlay", xaxis_title="RT (ms)")
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Mean ACC & RT by Training × Day (grouped bars)
# ══════════════════════════════════════════════════════════════════════════════
section("Mean Accuracy & RT by Training × Day", "📉")

agg = df.groupby(["day", "trainingtype"]).agg(
    mean_acc=("accuracy", "mean"),
    mean_rt =("rt",       "mean"),
    sem_acc =("accuracy", lambda x: x.sem()),
    sem_rt  =("rt",       lambda x: x.sem()),
).reset_index()

def delta_annotations(pivot, fmt):
    """Annotate Δ (Day2 − Day1) above each training group."""
    anns = []
    if len(pivot.columns) < 2:
        return anns
    d1_col, d2_col = pivot.columns[0], pivot.columns[1]
    for tr in pivot.index:
        d1, d2 = pivot.loc[tr, d1_col], pivot.loc[tr, d2_col]
        if pd.isna(d1) or pd.isna(d2):
            continue
        delta = d2 - d1
        anns.append(dict(
            x=tr, y=max(d1, d2),
            text=fmt.format(delta),
            showarrow=False, yshift=18,
            font=dict(size=11, color="green" if delta >= 0 else "red"),
        ))
    return anns

col1, col2 = st.columns(2)

pivot_acc    = agg.pivot(index="trainingtype", columns="day", values="mean_acc")
pivot_sem    = agg.pivot(index="trainingtype", columns="day", values="sem_acc")
pivot_rt     = agg.pivot(index="trainingtype", columns="day", values="mean_rt")
pivot_sem_rt = agg.pivot(index="trainingtype", columns="day", values="sem_rt")

with col1:
    st.subheader("Mean accuracy by training — per day")
    fig = go.Figure()
    for day in days:
        if day not in pivot_acc.columns:
            continue
        fig.add_trace(go.Bar(
            x=pivot_acc.index, y=pivot_acc[day],
            name=f"Day {day}",
            marker_color=day_color_map[day],
            error_y=dict(type="data", array=pivot_sem[day].tolist(), visible=True),
            opacity=0.85
        ))
    fig.update_layout(barmode="group", yaxis_title="Mean Accuracy", legend_title="Day",
                      yaxis_range=[0, 1.25],
                      annotations=delta_annotations(pivot_acc, "Δ={:+.2f}"))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Mean RT by training — per day")
    fig = go.Figure()
    for day in days:
        if day not in pivot_rt.columns:
            continue
        fig.add_trace(go.Bar(
            x=pivot_rt.index, y=pivot_rt[day],
            name=f"Day {day}",
            marker_color=day_color_map[day],
            error_y=dict(type="data", array=pivot_sem_rt[day].tolist(), visible=True),
            opacity=0.85
        ))
    fig.update_layout(barmode="group", yaxis_title="Mean RT (ms)", legend_title="Day",
                      annotations=delta_annotations(pivot_rt, "Δ={:+.0f} ms"))
    st.plotly_chart(fig, use_container_width=True)

# Delta change bar charts
if len(days) >= 2:
    st.subheader("Change from Day 1 to Day 2 (Δ = Day2 − Day1)")
    d1_col, d2_col = days[0], days[1]
    delta_df = pd.DataFrame(index=trainings)
    if d1_col in pivot_acc.columns and d2_col in pivot_acc.columns:
        delta_df["delta_acc"] = (pivot_acc[d2_col] - pivot_acc[d1_col]).round(3)
    if d1_col in pivot_rt.columns and d2_col in pivot_rt.columns:
        delta_df["delta_rt"]  = (pivot_rt[d2_col]  - pivot_rt[d1_col]).round(1)
    delta_df = delta_df.reset_index().rename(columns={"index": "trainingtype"})

    col1, col2 = st.columns(2)
    with col1:
        if "delta_acc" in delta_df.columns:
            fig = px.bar(delta_df, x="trainingtype", y="delta_acc",
                         color="delta_acc",
                         color_continuous_scale=["#d62728", "#aaa", "#2ca02c"],
                         color_continuous_midpoint=0,
                         labels={"delta_acc": "Δ Accuracy", "trainingtype": "Training"})
            fig.add_hline(y=0, line_color="black", line_width=1)
            fig.update_layout(yaxis_title="Δ Accuracy (Day2 − Day1)", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        if "delta_rt" in delta_df.columns:
            fig = px.bar(delta_df, x="trainingtype", y="delta_rt",
                         color="delta_rt",
                         color_continuous_scale=["#2ca02c", "#aaa", "#d62728"],
                         color_continuous_midpoint=0,
                         labels={"delta_rt": "Δ RT (ms)", "trainingtype": "Training"})
            fig.add_hline(y=0, line_color="black", line_width=1)
            fig.update_layout(yaxis_title="Δ RT ms (Day2 − Day1) — green = faster", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

# Boxplots
st.subheader("RT boxplot by training × day")
fig = px.box(
    df, x="trainingtype", y="rt", color="day",
    color_discrete_map={str(d): day_color_map[d] for d in days},
    points="outliers", category_orders={"trainingtype": trainings,"day": [1, 2]}
)
fig.update_layout(yaxis_title="RT (ms)", xaxis_title="Training type", legend_title="Day")
st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — RT & Accuracy over Trials (rolling mean)
# ══════════════════════════════════════════════════════════════════════════════
section("RT & Accuracy over Trials (Rolling Mean)", "🔄")

if "trial" in df.columns:
    # RT over trials
    st.subheader("RT over trials — rolling mean by training × day")
    fig = make_subplots(rows=1, cols=len(trainings), shared_yaxes=True,
                        subplot_titles=trainings)
    for col_i, training in enumerate(trainings):
        for day in days:
            sub = df[(df["trainingtype"] == training) & (df["day"] == day)]
            if sub.empty:
                continue
            sub = sub.sort_values("trial")
            rm = sub.groupby("trial")["rt"].mean().rolling(rolling_win_rt, min_periods=1, center=True).mean().reset_index()
            fig.add_trace(go.Scatter(
                x=rm["trial"], y=rm["rt"],
                mode="lines", name=f"Day {day}",
                line=dict(color=day_color_map[day], width=2.2),
                showlegend=(col_i == 0)
            ), row=1, col=col_i+1)
    fig.update_yaxes(title_text="RT (ms)", col=1)
    fig.update_layout(height=350, legend_title="Day")
    st.plotly_chart(fig, use_container_width=True)

    # Accuracy over trials
    st.subheader("Accuracy over trials — rolling mean by training × day")
    fig = make_subplots(rows=1, cols=len(trainings), shared_yaxes=True,
                        subplot_titles=trainings)
    for col_i, training in enumerate(trainings):
        for day in days:
            sub = df_raw[(df_raw["trainingtype"] == training) & (df_raw["day"] == day) & df_raw["subject"].isin(selected_subjects)]
            if sub.empty:
                continue
            sub = sub.sort_values("trial")
            rm = sub.groupby("trial")["accuracy"].mean().rolling(rolling_win_acc, min_periods=1, center=True).mean().reset_index()
            fig.add_trace(go.Scatter(
                x=rm["trial"], y=rm["accuracy"],
                mode="lines", name=f"Day {day}",
                line=dict(color=day_color_map[day], width=2.2),
                showlegend=(col_i == 0)
            ), row=1, col=col_i+1)
    fig.update_yaxes(title_text="Accuracy", col=1, range=[0, 1.05])
    fig.update_layout(height=350, legend_title="Day")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No `trial` column found — skipping trial-level plots. Add a `trial` column to your CSV to enable these.")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Block-level analysis
# ══════════════════════════════════════════════════════════════════════════════
# if "block" in df.columns:
#     section("Block-level Accuracy & RT", "🧱")
#     block_stats = df.groupby(["day", "block", "trainingtype"]).agg(
#         mean_acc=("accuracy", "mean"),
#         mean_rt =("rt",       "mean"),
#     ).reset_index()

#     col1, col2 = st.columns(2)
#     with col1:
#         st.subheader("Accuracy per block by training × day")
#         fig = px.line(block_stats, x="block", y="mean_acc",
#                       color="day", facet_col="trainingtype",
#                       color_discrete_map={str(d): day_color_map[d] for d in days},
#                       markers=True, labels={"mean_acc": "Mean Accuracy"})
#         fig.update_yaxes(range=[0, 1.05])
#         st.plotly_chart(fig, use_container_width=True)
#     with col2:
#         st.subheader("RT per block by training × day")
#         fig = px.line(block_stats, x="block", y="mean_rt",
#                       color="day", facet_col="trainingtype",
#                       color_discrete_map={str(d): day_color_map[d] for d in days},
#                       markers=True, labels={"mean_rt": "Mean RT (ms)"})
#         st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Error Rate vs RT scatter (speed-accuracy tradeoff)
# ══════════════════════════════════════════════════════════════════════════════
# section("Speed–Accuracy Tradeoff", "⚡")

# st.subheader("Mean RT vs error rate per subject × training × day")
# sat = df.groupby(["subject", "day", "trainingtype"]).agg(
#     mean_rt    =("rt",       "mean"),
#     error_rate =("accuracy", lambda x: 1 - x.mean()),
#     n_trials   =("rt",       "count"),
# ).reset_index()

# fig = px.scatter(
#     sat, x="mean_rt", y="error_rate",
#     color="trainingtype", facet_col="day",
#     symbol="subject" if len(selected_subjects) > 1 else None,
#     size="n_trials", size_max=18,
#     color_discrete_map=train_color_map,
#     hover_data=["subject", "n_trials"],
#     labels={"mean_rt": "Mean RT (ms)", "error_rate": "Error rate", "trainingtype": "Training"},
#     trendline="ols" if len(sat) > 3 else None,
# )
# fig.update_yaxes(range=[-0.02, 0.55])
# st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Multi-subject comparison (only when >1 subject selected)
# ══════════════════════════════════════════════════════════════════════════════
if len(selected_subjects) > 1:
    section("Cross-subject Comparison", "👥")

    subj_agg = df.groupby(["subject", "day", "trainingtype"]).agg(
        mean_acc=("accuracy", "mean"),
        mean_rt =("rt",       "mean"),
    ).reset_index()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Mean accuracy per subject")
        fig = px.bar(subj_agg, x="subject", y="mean_acc",
                     color="trainingtype", barmode="group",
                     facet_col="day",
                     color_discrete_map=train_color_map,
                     labels={"mean_acc": "Mean Accuracy"})
        fig.update_yaxes(range=[0, 1.05])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Mean RT per subject")
        fig = px.bar(subj_agg, x="subject", y="mean_rt",
                     color="trainingtype", barmode="group",
                     facet_col="day",
                     color_discrete_map=train_color_map,
                     labels={"mean_rt": "Mean RT (ms)"})
        st.plotly_chart(fig, use_container_width=True)

    # Line plot: each subject's trajectory across days, per training
    st.subheader("Accuracy trajectory across days — per subject")
    fig = px.line(subj_agg, x="day", y="mean_acc",
                  color="subject", facet_col="trainingtype",
                  markers=True,
                  labels={"mean_acc": "Mean Accuracy", "day": "Day"})
    fig.update_yaxes(range=[0, 1.05])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("RT trajectory across days — per subject")
    fig = px.line(subj_agg, x="day", y="mean_rt",
                  color="subject", facet_col="trainingtype",
                  markers=True,
                  labels={"mean_rt": "Mean RT (ms)", "day": "Day"})
    st.plotly_chart(fig, use_container_width=True)

    # Heatmap: mean RT per subject × day
    st.subheader("Heatmap — mean RT per subject × day")
    hm = subj_agg.groupby(["subject", "day"])["mean_rt"].mean().reset_index()
    hm_pivot = hm.pivot(index="subject", columns="day", values="mean_rt")
    fig = px.imshow(
        hm_pivot, text_auto=".0f",
        color_continuous_scale="YlOrRd",
        labels=dict(color="Mean RT (ms)"),
        aspect="auto"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Outlier check (z-score within day × training)
    st.subheader("Outlier check — z-score within day × training")
    outlier_df = subj_agg.copy()
    for metric in ["mean_acc", "mean_rt"]:
        outlier_df[f"z_{metric}"] = outlier_df.groupby(["day", "trainingtype"])[metric].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )
    flagged = outlier_df[(outlier_df["z_mean_acc"].abs() > 2) | (outlier_df["z_mean_rt"].abs() > 2)]
    if len(flagged):
        st.warning(f"⚠️ {len(flagged)} subject–condition(s) flagged as outliers (|z| > 2):")
        st.dataframe(flagged.round(3), use_container_width=True, hide_index=True)
    else:
        st.success("✅ No subjects flagged as outliers (|z| > 2).")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — Basic stats tests
# ══════════════════════════════════════════════════════════════════════════════
section("Statistical Tests", "🔬")
st.markdown("Mann–Whitney U tests comparing Day 1 vs Day 2, per training type.")

if len(days) >= 2:
    rows = []
    for t in trainings:
        for metric, label in [("rt", "RT"), ("accuracy", "Accuracy")]:
            d1 = df[(df["trainingtype"] == t) & (df["day"] == days[0])][metric].dropna()
            d2 = df[(df["trainingtype"] == t) & (df["day"] == days[1])][metric].dropna()
            if len(d1) < 3 or len(d2) < 3:
                continue
            u_stat, p_val = scipy_stats.mannwhitneyu(d1, d2, alternative="two-sided")
            rows.append({
                "Training": t, "Metric": label,
                f"Day {days[0]} mean": round(d1.mean(), 3),
                f"Day {days[1]} mean": round(d2.mean(), 3),
                "U statistic": round(u_stat, 1),
                "p-value": round(p_val, 4),
                "Significant (p<.05)": "✅" if p_val < 0.05 else "❌"
            })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("Not enough data for statistical tests.")
else:
    st.info("Statistical tests require data from at least 2 days.")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — Raw data preview
# ══════════════════════════════════════════════════════════════════════════════
with st.expander("🔍 View raw data"):
    st.dataframe(df_raw[df_raw["subject"].isin(selected_subjects)], use_container_width=True)
