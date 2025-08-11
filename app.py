import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import re

st.set_page_config(page_title="Canvas Gradebook Dashboard", page_icon="🎓", layout="wide")

# ------------------ Helpers ------------------
@st.cache_data
def load_csv(file):
    return pd.read_csv(file, dtype=str)

def detect_points_row(df):
    cand = df[df.iloc[:,0].astype(str).str.contains("Points Possible", case=False, na=False)]
    if not cand.empty:
        return cand.index[0]
    return None

def parse_numeric_series(s):
    raw = s.astype(str)
    blanks = raw.str.strip().isin(["", "nan", "NaN", "-", "—", "–"])
    excused = raw.str.strip().str.upper().isin(["EX","EXCUSED"])
    def to_num(x):
        x = str(x).strip().replace("%","")
        if x == "" or x.upper() in ["EX","EXCUSED","MI","MISSING","INC","INCOMPLETE","-","—","–"]:
            return np.nan
        try:
            return float(x)
        except:
            return np.nan
    numeric = raw.map(to_num)
    return numeric, blanks, excused

def infer_columns(df):
    cols = list(df.columns)
    meta_names = {"student","id","sis user id","sis login id","section"}
    grade_names = {
        "current score","unposted current score",
        "final score","unposted final score",
        "current grade","unposted current grade",
        "final grade","unposted final grade"
    }
    lower = {c: c.lower().strip() for c in cols}
    meta_cols = [c for c in cols if lower[c] in meta_names]
    grade_cols = [c for c in cols if lower[c] in grade_names]
    assign_cols = [c for c in cols if c not in meta_cols + grade_cols]
    return meta_cols, grade_cols, assign_cols

def compute_percent_scores(df_numeric, points_row, assign_cols):
    pct = pd.DataFrame(index=df_numeric.index, columns=assign_cols, dtype=float)
    pts = {}
    for c in assign_cols:
        try:
            max_pts = float(points_row[c])
        except Exception:
            max_pts = np.nan
        pts[c] = max_pts
        if np.isfinite(max_pts) and max_pts > 0:
            pct[c] = (df_numeric[c] / max_pts) * 100.0
        else:
            pct[c] = np.nan
    return pct, pd.Series(pts)

def shorten_label(s, limit=36):
    s = re.sub(r"\(\d+\)$", "", s).strip()
    return (s[:limit] + "…") if len(s) > limit else s

def kpi(title, value, help=None):
    with st.container():
        st.metric(title, value, help=help)

def fig_layout(fig, h=420, margin=dict(l=40,r=20,t=50,b=40)):
    fig.update_layout(height=h, margin=margin, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

# ------------------ Sidebar ------------------
with st.sidebar:
    st.title("🎓 Canvas Gradebook")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    st.caption("Export from Canvas with **Points Possible** enabled (first row includes max points).")

    st.markdown("**Options**")
    treat_excused_as_zero = st.toggle("Excused counts as zero (incl. missing)", value=False)
    st.caption("Affects 'including missing' averages.")
    st.divider()
    st.markdown("**About**")
    st.caption("Tabs organize views: Overview • Assignments • Students • Patterns • Export")

# ------------------ Data ingest ------------------
if uploaded:
    raw = load_csv(uploaded)
else:
    st.info("No file uploaded — showing demo with `sample_canvas_gradebook.csv`.")
    raw = load_csv("sample_canvas_gradebook.csv")

points_idx = detect_points_row(raw)
if points_idx is None:
    points_row = pd.Series(dtype=str)
    data = raw.copy()
    has_pts = False
else:
    points_row = raw.loc[points_idx]
    data = raw.drop(index=points_idx).reset_index(drop=True)
    has_pts = True

meta_cols, grade_cols, assign_cols = infer_columns(data)

num_df = pd.DataFrame(index=data.index, columns=assign_cols, dtype=float)
is_missing = pd.DataFrame(False, index=data.index, columns=assign_cols)
is_excused = pd.DataFrame(False, index=data.index, columns=assign_cols)

for c in assign_cols:
    num, miss_blank, exc = parse_numeric_series(data[c])
    num_df[c] = num
    # Missing: blank OR parsed NaN; Canvas zeros are real zeros (submitted)
    is_missing[c] = miss_blank | num.isna()
    is_excused[c] = exc

pct_df, points_possible = compute_percent_scores(num_df, points_row, assign_cols) if has_pts else (pd.DataFrame(), pd.Series(dtype=float))

# Prefer Canvas numeric final; else compute
final_score_col = None
for name in ["Final Score","Unposted Final Score","Current Score","Unposted Current Score"]:
    if name in data.columns:
        s, _, _ = parse_numeric_series(data[name])
        if s.notna().any():
            data[name+"_num"] = s
            final_score_col = name+"_num"
            break
if final_score_col is None and not pct_df.empty:
    data["Computed Final Score"] = pct_df.mean(axis=1)
    final_score_col = "Computed Final Score"

# ------------------ Top header ------------------
st.title("Canvas Gradebook Dashboard")
st.caption("Cleaner layout • Larger charts • Logical navigation via tabs")

c1, c2, c3, c4 = st.columns(4)
kpi("Students", data.shape[0])
kpi("Assignments", len(assign_cols))
if final_score_col:
    fs = data[final_score_col].astype(float)
    kpi("Avg Final", f"{fs.mean():.1f}%")
    kpi("Fs (Final<60)", int((fs < 60).sum()))
else:
    kpi("Avg Final", "—")
    kpi("Fs (Final<60)", "—")

st.divider()

# ------------------ Tabs ------------------
tabs = st.tabs(["🏠 Overview", "🧪 Assignments", "👤 Students", "🧩 Patterns", "📤 Export"])

# ===== Overview Tab =====
with tabs[0]:
    col1, col2 = st.columns([1.1, 1])
    with col1:
        st.subheader("Final Score Distribution")
        if final_score_col:
            fig = px.histogram(data, x=final_score_col, nbins=12, labels={final_score_col:"Final Score (%)"})
            fig.update_traces(opacity=0.85)
            fig.update_xaxes(title="Final Score (%)")
            fig = fig_layout(fig, h=420)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric final score detected.")
    with col2:
        st.subheader("Grade Bands")
        grade_col = next((nm for nm in ["Final Grade","Unposted Final Grade","Current Grade","Unposted Current Grade"] if nm in data.columns), None)
        if grade_col:
            counts = data[grade_col].fillna("N/A").value_counts().reset_index()
            counts.columns = ["Grade","Count"]
            fig = px.bar(counts, x="Grade", y="Count")
            fig = fig_layout(fig, h=420)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No letter-grade columns found.")
    st.markdown(" ")
    if has_pts and not pct_df.empty:
        st.subheader("Assignment Difficulty (Percent Averages)")
        incl = pct_df.fillna(0.0).copy()  # including missing (and excused as 0 if toggle on)
        if treat_excused_as_zero:
            pass  # excused were NaN -> now 0 due to fillna
        excl = pct_df.copy()              # excluding missing
        avg_incl = incl.mean().sort_values()
        avg_excl = excl.mean().sort_values()
        df_plot = pd.DataFrame({
            "Assignment": [shorten_label(c) for c in avg_incl.index],
            "Avg Including Missing": avg_incl.values,
            "Avg Excluding Missing": avg_excl.reindex(avg_incl.index).values
        })
        fig = go.Figure()
        fig.add_bar(x=df_plot["Assignment"], y=df_plot["Avg Including Missing"], name="Including Missing")
        fig.add_bar(x=df_plot["Assignment"], y=df_plot["Avg Excluding Missing"], name="Excluding Missing")
        fig.update_xaxes(tickangle=45)
        fig = fig_layout(fig, h=480)
        st.plotly_chart(fig, use_container_width=True)

# ===== Assignments Tab =====
with tabs[1]:
    st.subheader("Missing vs Excused Heatmap")
    if not is_missing.empty:
        # 1 = missing, 0.5 = excused, 0 = submitted
        mat = is_missing.astype(int).values.astype(float) - 0.5 * is_excused.astype(int).values
        fig = px.imshow(
            mat,
            labels=dict(x="Assignments", y="Students", color="Status"),
            x=[shorten_label(a, 20) for a in assign_cols],
            y=data["Student"].tolist() if "Student" in data.columns else None,
            aspect="auto"
        )
        fig = fig_layout(fig, h=min(800, 40 + 22*data.shape[0]))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No assignment columns detected.")

    st.markdown(" ")
    st.subheader("Per-Assignment Summary")
    if has_pts and not pct_df.empty:
        summary = pd.DataFrame({
            "Avg Including Missing": pct_df.fillna(0.0).mean().round(1),
            "Avg Excluding Missing": pct_df.mean().round(1),
            "Missing Rate": is_missing.mean().round(3),
            "Excused Rate": is_excused.mean().round(3),
            "Points Possible": points_possible
        })
        st.dataframe(summary.sort_values("Avg Excluding Missing"), use_container_width=True)
    else:
        st.info("Percent scores unavailable (no Points Possible row).")

# ===== Students Tab =====
with tabs[2]:
    st.subheader("Student Trajectories")
    if has_pts and not pct_df.empty and "Student" in data.columns:
        picks = st.multiselect("Pick up to 5 students", options=data["Student"].tolist(), default=data["Student"].tolist()[:5], max_selections=5)
        if picks:
            fig = go.Figure()
            for s in picks:
                row = pct_df.loc[data["Student"]==s]
                if not row.empty:
                    fig.add_scatter(x=[shorten_label(a) for a in assign_cols], y=row.iloc[0].values, mode="lines+markers", name=s)
            fig.update_xaxes(tickangle=45, title_text="Assignments")
            fig.update_yaxes(title_text="Score (%)", range=[0,100])
            fig = fig_layout(fig, h=460)
            st.plotly_chart(fig, use_container_width=True)
        with st.expander("Show student table"):
            if not pct_df.empty:
                st.dataframe(pd.concat([data[["Student"]] if "Student" in data.columns else pd.DataFrame(), pct_df.add_suffix(" [%]")], axis=1), use_container_width=True)
            else:
                st.dataframe(data, use_container_width=True)
    else:
        st.info("Need percent scores and a Student column.")

# ===== Patterns Tab =====
with tabs[3]:
    st.subheader("Assignment Correlations (Completed Work)")
    if has_pts and not pct_df.empty and len(assign_cols) >= 2:
        corr = pct_df.replace(0, np.nan).corr()
        fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="Blues")
        fig = fig_layout(fig, h=540)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Upload a CSV with at least two assignments to compute correlations.")
    st.markdown(" ")
    st.subheader("Early-Warning (First K Assignments)")
    if has_pts and not pct_df.empty:
        K = st.slider("Use first K assignments", 1, len(assign_cols), min(3, len(assign_cols)))
        early = pct_df[assign_cols[:K]].replace(0, np.nan).mean(axis=1)
        cutoff = np.nanpercentile(early, 20)
        flagged = early <= cutoff
        out = pd.DataFrame({
            "Student": data["Student"] if "Student" in data.columns else np.arange(len(early)),
            "Early Avg (%)": early.round(1),
            "Final (%)": data[final_score_col].round(1) if final_score_col else np.nan,
            "Flagged": flagged
        }).sort_values("Early Avg (%)")
        st.dataframe(out, use_container_width=True)
    else:
        st.info("Need percent scores to run early-warning.")

# ===== Export Tab =====
with tabs[4]:
    st.subheader("Export Cleaned Percent Dataset")
    if not pct_df.empty:
        cleaned = data.copy()
        if "Student" in cleaned.columns:
            cleaned.set_index("Student", inplace=True)
        out = pd.concat([cleaned, pct_df.add_suffix(" [%]")], axis=1)
        csv = out.to_csv(index=True).encode("utf-8")
        st.download_button("Download cleaned CSV", data=csv, file_name="canvas_gradebook_cleaned.csv", mime="text/csv")
    else:
        st.caption("Upload a Canvas CSV with the 'Points Possible' row to enable export.")
