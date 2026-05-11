import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats as scipy_stats

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Association Task Dashboard",
    page_icon="🔗",
    layout="wide",
)

st.title("🔗 Association Task Dashboard")
st.markdown(
    "Upload a CSV with columns: `subject`, `day`, `trainingtype`, `accuracy`, `RT`  \n"
    "Optional extra columns for richer plots: `trial`, `block`, `matching`, `typemismatch`"
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    use_iqr = st.toggle("Use IQR outlier filter", value=True,
                        help="If ON: removes RTs outside 1.5×IQR per training×day. If OFF: use fixed bounds below.")
    rt_min_fixed = st.number_input("Fixed RT lower bound (ms)", value=100, step=10,
                                    help="Only used when IQR filter is OFF")
    rt_max_fixed = st.number_input("Fixed RT upper bound (ms)", value=2000, step=10,
                                    help="Only used when IQR filter is OFF")
    rolling_win_rt  = st.slider("Rolling window — RT (trials)", 3, 30, 5)
    rolling_win_acc = st.slider("Rolling window — Accuracy (trials)", 3, 30, 10)
    st.markdown("---")
    st.caption("Association Task · Lab Dashboard 💡")

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload CSV file(s)",
    type="csv",
    accept_multiple_files=True,
)

SAMPLE_PATH = "data/association_clean.csv"  # or association_sample.csv
df_loaded = pd.read_csv(SAMPLE_PATH)

# if not uploaded:
#     st.info("📂 Showing association task data. Upload your own CSV to analyse it.")
    
# else:
#     df_loaded = pd.concat([pd.read_csv(f) for f in uploaded], ignore_index=True)

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
    combined["rt"]       = pd.to_numeric(combined["rt"],       errors="coerce")
    combined["accuracy"] = pd.to_numeric(combined["accuracy"], errors="coerce")
    combined["day"]      = combined["day"].astype(int)
    combined.dropna(subset=["accuracy", "rt"], inplace=True)
    return combined

raw = load_files(uploaded)
if raw is None:
    st.stop()

# ── IQR outlier filter ────────────────────────────────────────────────────────
def apply_iqr_filter(df):
    keep = pd.Series(False, index=df.index)
    for (tr, day), grp in df.groupby(["trainingtype", "day"]):
        q1, q3 = grp["rt"].quantile(0.25), grp["rt"].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = grp["rt"].between(lo, hi) & grp["rt"].gt(0)
        keep.loc[mask.index[mask]] = True
    return df.loc[keep].copy()

if use_iqr:
    df_clean = apply_iqr_filter(raw)
else:
    df_clean = raw[(raw["rt"] >= rt_min_fixed) & (raw["rt"] <= rt_max_fixed)].copy()

# ── Subject selector ──────────────────────────────────────────────────────────
all_subjects = sorted(raw["subject"].unique())
selected_subjects = st.multiselect(
    "Select subject(s)",
    options=all_subjects,
    default=all_subjects[:1],
)

if not selected_subjects:
    st.warning("Please select at least one subject.")
    st.stop()

df     = df_clean[df_clean["subject"].isin(selected_subjects)].copy()
df_raw = raw[raw["subject"].isin(selected_subjects)].copy()

trainings = sorted(df["trainingtype"].unique())
days      = sorted(df["day"].unique())

TRAIN_COLORS = px.colors.qualitative.Set2
DAY_COLORS   = {1: "#4C72B0", 2: "#DD8452", 3: "#55A868", 4: "#C44E52"}

train_color_map = {t: TRAIN_COLORS[i % len(TRAIN_COLORS)] for i, t in enumerate(trainings)}
day_color_map   = {d: DAY_COLORS.get(d, "#888") for d in days}

has_trial   = "trial"   in df.columns
has_block   = "block"   in df.columns
has_match   = "matching" in df.columns
has_mismatch= "typemismatch" in df.columns

def section(title, icon=""):
    st.markdown(f"---\n## {icon} {title}")

# ══════════════════════════════════════════════════════════════════════════════
# 1 · Summary stats
# ══════════════════════════════════════════════════════════════════════════════
section("Summary Statistics", "📊")

summary = df.groupby(["subject", "day", "trainingtype"]).agg(
    n_trials  =("rt",       "count"),
    mean_acc  =("accuracy", "mean"),
    sem_acc   =("accuracy", lambda x: x.sem()),
    mean_rt   =("rt",       "mean"),
    sem_rt    =("rt",       lambda x: x.sem()),
    median_rt =("rt",       "median"),
).round(3).reset_index()

st.dataframe(
    summary.style.format({
        "mean_acc": "{:.3f}", "sem_acc": "{:.3f}",
        "mean_rt": "{:.1f}",  "sem_rt": "{:.1f}", "median_rt": "{:.1f}"
    }),
    use_container_width=True, hide_index=True,
)

# Outlier removal report
n_before = len(df_raw)
n_after  = len(df)
n_removed = n_before - n_after
if n_removed > 0:
    st.caption(
        f"🔍 RT outlier filter removed **{n_removed}** trials "
        f"({n_removed/n_before*100:.1f}%) — {n_after} trials remaining."
    )

# ══════════════════════════════════════════════════════════════════════════════
# 2 · RT Distributions
# ══════════════════════════════════════════════════════════════════════════════
section("RT Distributions", "📈")

col1, col2 = st.columns(2)


st.subheader("RT histogram by training type")
fig = go.Figure()
for t in trainings:
    sub = df[df["trainingtype"] == t]["rt"].dropna()
    fig.add_trace(go.Histogram(
        x=sub, name=t, opacity=0.6,
        marker_color=train_color_map[t],
        nbinsx=40, histnorm="probability density"
    ))
fig.update_layout(barmode="overlay", xaxis_title="RT (ms)", yaxis_title="Density")
st.plotly_chart(fig, use_container_width=True)

# with col2:
#     st.subheader("RT violin + strip by training type")
#     fig = go.Figure()
#     for t in trainings:
#         sub = df[df["trainingtype"] == t]["rt"].dropna()
#         fig.add_trace(go.Violin(
#             y=sub, name=t, box_visible=True, meanline_visible=True,
#             points="all", pointpos=0, jitter=0.3,
#             fillcolor=train_color_map[t], opacity=0.65, line_color="black"
#         ))
#     fig.update_layout(yaxis_title="RT (ms)")
#     st.plotly_chart(fig, use_container_width=True)

# RT distributions split by day
st.subheader("RT distribution split by day")
if len(days) > 0:
    fig = make_subplots(rows=1, cols=len(days), shared_yaxes=True,
                        subplot_titles=[f"Day {d}" for d in days])
    for i, day in enumerate(days):
        day_df = df[df["day"] == day]
        for t in trainings:
            sub = day_df[day_df["trainingtype"] == t]["rt"].dropna()
            fig.add_trace(go.Histogram(
                x=sub, name=t, opacity=0.6,
                marker_color=train_color_map[t],
                nbinsx=30, histnorm="probability density",
                showlegend=(i == 0)
            ), row=1, col=i+1)
    fig.update_layout(barmode="overlay")
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# 3 · Mean Accuracy & RT by Training × Day  (bar charts + delta annotations)
# ══════════════════════════════════════════════════════════════════════════════
section("Mean Accuracy & RT by Training × Day", "📉")

agg = df.groupby(["day", "trainingtype"]).agg(
    mean_acc=("accuracy", "mean"),
    sem_acc =("accuracy", lambda x: x.sem()),
    mean_rt =("rt",       "mean"),
    sem_rt  =("rt",       lambda x: x.sem()),
).reset_index()

col1, col2 = st.columns(2)

def delta_annotations(pivot, fmt):
    """Return list of annotation dicts showing Day2−Day1 above each group."""
    anns = []
    if len(pivot.columns) < 2:
        return anns
    d1_col, d2_col = pivot.columns[0], pivot.columns[1]
    for i, tr in enumerate(pivot.index):
        d1 = pivot.loc[tr, d1_col]
        d2 = pivot.loc[tr, d2_col]
        if pd.isna(d1) or pd.isna(d2):
            continue
        delta = d2 - d1
        color = "green" if delta >= 0 else "red"
        anns.append(dict(
            x=tr, y=max(d1, d2),
            text=fmt.format(delta),
            showarrow=False,
            yshift=18,
            font=dict(size=11, color=color),
        ))
    return anns

with col1:
    st.subheader("Mean accuracy — with Δ (Day2−Day1)")
    pivot_acc = agg.pivot(index="trainingtype", columns="day", values="mean_acc")
    pivot_sem = agg.pivot(index="trainingtype", columns="day", values="sem_acc")
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
    fig.update_layout(
        barmode="group", yaxis_title="Mean Accuracy",
        yaxis_range=[0, 1.25],
        annotations=delta_annotations(pivot_acc, "Δ={:+.2f}")
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Mean RT — with Δ (Day2−Day1)")
    pivot_rt  = agg.pivot(index="trainingtype", columns="day", values="mean_rt")
    pivot_sem_rt = agg.pivot(index="trainingtype", columns="day", values="sem_rt")
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
    fig.update_layout(
        barmode="group", yaxis_title="Mean RT (ms)",
        annotations=delta_annotations(pivot_rt, "Δ={:+.0f} ms")
    )
    st.plotly_chart(fig, use_container_width=True)

# Delta change bar chart
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

# Boxplot
st.subheader("RT boxplot by training × day")
fig = px.box(
    df, x="trainingtype", y="rt", color="day",
    color_discrete_map={str(d): day_color_map[d] for d in days},
    points="outliers", category_orders={"trainingtype": trainings, "day": [1, 2]}
)
fig.update_layout(yaxis_title="RT (ms)", xaxis_title="Training type")
st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# 4 · Rolling mean over trials
# ══════════════════════════════════════════════════════════════════════════════
if has_trial:
    section("RT over Trials (Rolling Mean)", "🔄")

    # RT over trials — per training, colored by day
    st.subheader("RT over trials — rolling mean: one panel per training, colored by day")
    fig = make_subplots(rows=1, cols=len(trainings), shared_yaxes=True,
                        subplot_titles=trainings)
    for col_i, training in enumerate(trainings):
        for day in days:
            sub = df[(df["trainingtype"] == training) & (df["day"] == day)].sort_values("trial")
            if sub.empty:
                continue
            # correct trials rolling mean
            correct = sub[sub["accuracy"] == 1]
            tm = correct.groupby("trial")["rt"].mean()
            rm = tm.rolling(rolling_win_rt, min_periods=1, center=True)
            fig.add_trace(go.Scatter(
                x=tm.index, y=rm.mean().values,
                mode="lines", name=f"Day {day}",
                line=dict(color=day_color_map[day], width=2.2),
                showlegend=(col_i == 0)
            ), row=1, col=col_i+1)
            # error scatter
            errors = sub[sub["accuracy"] == 0]
            if not errors.empty:
                fig.add_trace(go.Scatter(
                    x=errors["trial"], y=errors["rt"],
                    mode="markers",
                    marker=dict(symbol="x", color=day_color_map[day], size=6, opacity=0.6),
                    name=f"Day {day} errors",
                    showlegend=False
                ), row=1, col=col_i+1)
    fig.update_yaxes(title_text="RT (ms)", col=1)
    fig.update_layout(height=380, legend_title="Day",
                      title="Correct RT rolling mean + error scatter (×)")
    st.plotly_chart(fig, use_container_width=True)

    # Accuracy over trials — per training, colored by day
    # st.subheader("Accuracy over trials — rolling mean: one panel per training, colored by day")
    # fig = make_subplots(rows=1, cols=len(trainings), shared_yaxes=True,
    #                     subplot_titles=trainings)
    # for col_i, training in enumerate(trainings):
    #     for day in days:
    #         sub = df_raw[(df_raw["trainingtype"] == training) & (df_raw["day"] == day)
    #                      & df_raw["subject"].isin(selected_subjects)].sort_values("trial")
    #         if sub.empty:
    #             continue
    #         rm = sub["accuracy"].rolling(rolling_win_acc, min_periods=1, center=True)
    #         fig.add_trace(go.Scatter(
    #             x=sub["trial"].values, y=rm.mean().values,
    #             mode="lines", name=f"Day {day}",
    #             line=dict(color=day_color_map[day], width=2.2),
    #             showlegend=(col_i == 0)
    #         ), row=1, col=col_i+1)
    # fig.update_yaxes(title_text="Accuracy", col=1, range=[0, 1.05])
    # fig.update_layout(height=380, legend_title="Day")
    # st.plotly_chart(fig, use_container_width=True)

else:
    st.info("No `trial` column found — trial-level plots skipped. Add a `trial` column to enable them.")

# ══════════════════════════════════════════════════════════════════════════════
# 5 · Block-level analysis
# ══════════════════════════════════════════════════════════════════════════════
# if has_block:
#     section("Accuracy & RT across Blocks", "🧱")
#     block_stats = df.groupby(["day", "block", "trainingtype"]).agg(
#         mean_acc=("accuracy", "mean"),
#         sem_acc =("accuracy", lambda x: x.sem()),
#         mean_rt =("rt",       "mean"),
#         sem_rt  =("rt",       lambda x: x.sem()),
#     ).reset_index()

#     col1, col2 = st.columns(2)
#     with col1:
#         st.subheader("Accuracy per block — Day 1 vs Day 2 overlay")
#         fig = make_subplots(rows=1, cols=len(trainings), shared_yaxes=True,
#                             subplot_titles=trainings)
#         for col_i, training in enumerate(trainings):
#             for day in days:
#                 d = block_stats[(block_stats["trainingtype"] == training) & (block_stats["day"] == day)]
#                 if d.empty:
#                     continue
#                 fig.add_trace(go.Scatter(
#                     x=d["block"], y=d["mean_acc"],
#                     mode="lines+markers", name=f"Day {day}",
#                     line=dict(color=day_color_map[day], width=2),
#                     error_y=dict(type="data", array=d["sem_acc"].tolist(), visible=True),
#                     showlegend=(col_i == 0)
#                 ), row=1, col=col_i+1)
#         fig.update_yaxes(range=[0, 1.1], title_text="Mean Accuracy", col=1)
#         fig.update_layout(height=350, legend_title="Day")
#         st.plotly_chart(fig, use_container_width=True)

#     with col2:
#         st.subheader("RT per block — Day 1 vs Day 2 overlay")
#         fig = make_subplots(rows=1, cols=len(trainings), shared_yaxes=True,
#                             subplot_titles=trainings)
#         for col_i, training in enumerate(trainings):
#             for day in days:
#                 d = block_stats[(block_stats["trainingtype"] == training) & (block_stats["day"] == day)]
#                 if d.empty:
#                     continue
#                 fig.add_trace(go.Scatter(
#                     x=d["block"], y=d["mean_rt"],
#                     mode="lines+markers", name=f"Day {day}",
#                     line=dict(color=day_color_map[day], width=2),
#                     error_y=dict(type="data", array=d["sem_rt"].tolist(), visible=True),
#                     showlegend=(col_i == 0)
#                 ), row=1, col=col_i+1)
#         fig.update_yaxes(title_text="Mean RT (ms)", col=1)
#         fig.update_layout(height=350, legend_title="Day")
#         st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# 6 · Matching / Mismatch breakdown  (if column present)
# ══════════════════════════════════════════════════════════════════════════════
if has_match or has_mismatch:
    section("Matching / Mismatch Breakdown", "🔀")
    cond_col = "matching" if has_match else "typemismatch"
    cond_vals = sorted(df[cond_col].dropna().unique())

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Mean accuracy by {cond_col} × training × day")
        cond_agg = df.groupby(["day", "trainingtype", cond_col]).agg(
            mean_acc=("accuracy", "mean")
        ).reset_index()
        fig = px.bar(cond_agg, x="trainingtype", y="mean_acc",
                     color=cond_col, barmode="group",
                     facet_col="day",
                     labels={"mean_acc": "Mean Accuracy", "trainingtype": "Training"})
        fig.update_yaxes(range=[0, 1.1])
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader(f"Mean RT by {cond_col} × training × day")
        fig = px.bar(cond_agg.merge(
            df.groupby(["day", "trainingtype", cond_col])["rt"].mean().reset_index(),
            on=["day", "trainingtype", cond_col]
        ), x="trainingtype", y="rt",
                     color=cond_col, barmode="group", facet_col="day",
                     labels={"rt": "Mean RT (ms)", "trainingtype": "Training"})
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# 7 · Heatmaps
# ══════════════════════════════════════════════════════════════════════════════
section("Heatmaps", "🟥")

hm_agg = df.groupby(["trainingtype", "day"]).agg(
    mean_rt =("rt",       "mean"),
    mean_acc=("accuracy", "mean"),
).reset_index()

col1, col2 = st.columns(2)
with col1:
    st.subheader("Mean RT — training × day")
    hm_rt = hm_agg.pivot(index="trainingtype", columns="day", values="mean_rt")
    fig = px.imshow(hm_rt, text_auto=".0f", color_continuous_scale="YlOrRd",
                    labels=dict(color="Mean RT (ms)"), aspect="auto")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Mean accuracy — training × day")
    hm_acc = hm_agg.pivot(index="trainingtype", columns="day", values="mean_acc")
    fig = px.imshow(hm_acc, text_auto=".3f", color_continuous_scale="Blues",
                    zmin=0, zmax=1, labels=dict(color="Mean Accuracy"), aspect="auto")
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# 8 · Multi-subject comparison
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
                     color="trainingtype", barmode="group", facet_col="day",
                     color_discrete_map=train_color_map,
                     labels={"mean_acc": "Mean Accuracy"})
        fig.update_yaxes(range=[0, 1.1])
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Mean RT per subject")
        fig = px.bar(subj_agg, x="subject", y="mean_rt",
                     color="trainingtype", barmode="group", facet_col="day",
                     color_discrete_map=train_color_map,
                     labels={"mean_rt": "Mean RT (ms)"})
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Accuracy trajectory across days — per subject")
    fig = px.line(subj_agg, x="day", y="mean_acc", color="subject",
                  facet_col="trainingtype", markers=True,
                  labels={"mean_acc": "Mean Accuracy", "day": "Day"})
    fig.update_yaxes(range=[0, 1.05])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("RT trajectory across days — per subject")
    fig = px.line(subj_agg, x="day", y="mean_rt", color="subject",
                  facet_col="trainingtype", markers=True,
                  labels={"mean_rt": "Mean RT (ms)", "day": "Day"})
    st.plotly_chart(fig, use_container_width=True)

    # heatmap per subject × day
    st.subheader("Heatmap — mean RT per subject × day")
    hm = subj_agg.groupby(["subject", "day"])["mean_rt"].mean().reset_index()
    hm_piv = hm.pivot(index="subject", columns="day", values="mean_rt")
    fig = px.imshow(hm_piv, text_auto=".0f", color_continuous_scale="YlOrRd",
                    labels=dict(color="Mean RT (ms)"), aspect="auto")
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# 8b · Speed–Accuracy Tradeoff
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
# 8c · Outlier check (multi-subject only)
# ══════════════════════════════════════════════════════════════════════════════
if len(selected_subjects) > 1:
    subj_agg_out = df.groupby(["subject", "day", "trainingtype"]).agg(
        mean_acc=("accuracy", "mean"),
        mean_rt =("rt",       "mean"),
        n_trials=("rt",       "count"),
    ).reset_index()
    for metric in ["mean_acc", "mean_rt"]:
        subj_agg_out[f"z_{metric}"] = subj_agg_out.groupby(["day", "trainingtype"])[metric].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )
    flagged = subj_agg_out[
        (subj_agg_out["z_mean_acc"].abs() > 2) | (subj_agg_out["z_mean_rt"].abs() > 2)
    ]
    if len(flagged):
        st.warning(f"⚠️ {len(flagged)} subject–condition(s) flagged as outliers (|z| > 2):")
        st.dataframe(flagged.round(3), use_container_width=True, hide_index=True)
    else:
        st.success("✅ No subjects flagged as outliers (|z| > 2).")

# ══════════════════════════════════════════════════════════════════════════════
# 9 · Statistical tests  (paired t-test Day1 vs Day2 per subject × training)
# ══════════════════════════════════════════════════════════════════════════════
section("Statistical Tests", "🔬")

st.markdown(
    "**Paired t-test** (Day 1 vs Day 2) on per-subject means, per training condition.  \n"
    "Requires ≥ 2 subjects with data on both days."
)

if len(days) >= 2:
    rows = []
    subj_by_cond = df.groupby(["subject", "day", "trainingtype"]).agg(
        mean_acc=("accuracy", "mean"),
        mean_rt =("rt",       "mean"),
    ).reset_index()

    for training in trainings:
        for metric, label in [("mean_acc", "Accuracy"), ("mean_rt", "RT")]:
            per_subj = (subj_by_cond[subj_by_cond["trainingtype"] == training]
                        .pivot(index="subject", columns="day", values=metric))
            d1_col, d2_col = days[0], days[1]
            if d1_col not in per_subj.columns or d2_col not in per_subj.columns:
                continue
            both = per_subj[[d1_col, d2_col]].dropna()
            if len(both) < 2:
                rows.append({"Training": training, "Metric": label,
                              f"Day {d1_col} mean": "—", f"Day {d2_col} mean": "—",
                              "Δ": "—", "t": "—", "p-value": "—",
                              "Significant (p<.05)": "⚠️ too few subjects"})
                continue
            t, p = scipy_stats.ttest_rel(both[d1_col], both[d2_col])
            rows.append({
                "Training": training, "Metric": label,
                f"Day {d1_col} mean": round(both[d1_col].mean(), 3),
                f"Day {d2_col} mean": round(both[d2_col].mean(), 3),
                "Δ": round(both[d2_col].mean() - both[d1_col].mean(), 3),
                "t": round(t, 3), "p-value": round(p, 4),
                "Significant (p<.05)": "✅" if p < 0.05 else "❌"
            })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("Not enough data for statistical tests.")
else:
    st.info("Statistical tests require data from at least 2 days.")

# ══════════════════════════════════════════════════════════════════════════════
# 10 · Raw data
# ══════════════════════════════════════════════════════════════════════════════
with st.expander("🔍 View raw data"):
    st.dataframe(df_raw, use_container_width=True)
