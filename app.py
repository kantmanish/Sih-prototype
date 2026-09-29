from __future__ import annotations

import hashlib
import html
import io
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_option_menu import option_menu

from engine import QA, Corpus, build_report, ingest_paths, kpis, topics, validate, word_freq

st.set_page_config(
    page_title="CIL AI Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_EXT = {".pdf", ".docx", ".xlsx", ".xls", ".csv", ".txt", ".png", ".jpg", ".jpeg"}
UPLOAD_TYPES = ["pdf", "docx", "xlsx", "xls", "csv", "txt", "png", "jpg"]

# ---------------------------------------------------------------- Theme
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

:root {
    --bg: #090e18; --panel: #151d2d; --panel-2: #111827; --line: #263247;
    --muted: #7790af; --text: #f3f7ff; --cyan: #16c5e8; --blue: #3982f5;
    --purple: #8b5cf6; --green: #12c98c; --amber: #f4b740; --red: #f0566a;
}
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: var(--bg); color: var(--text); }
.block-container { padding: 0 38px 48px; max-width: 1600px; }
[data-testid="stHeader"] { background: var(--bg); }
/* ---- static (always-open, non-collapsible) left sidebar ---- */
[data-testid="stSidebar"] { background: #0d1522; border-right: 1px solid #202b3d; }
[data-testid="stSidebar"] > div:first-child { padding: 0 20px 20px; }
[data-testid="stSidebarUserContent"] { padding: 0 !important; }
[data-testid="stSidebarHeader"], [data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"],
[data-testid="stSidebarResizeHandle"], [data-testid="stExpandSidebarButton"] { display: none !important; }
@media (min-width: 769px) {
  section[data-testid="stSidebar"] {
    transform: none !important; margin-left: 0 !important; visibility: visible !important;
    width: 300px !important; min-width: 300px !important; max-width: 300px !important;
  }
}
@media (min-width: 769px) {
  section[data-testid="stSidebar"][aria-expanded="false"] {
    transform: none !important; margin-left: 0 !important; display: block !important;
  }
}
[data-testid="stSidebar"] iframe { border: 0; }

/* sidebar brand + user */
.brand { display:flex; align-items:center; gap:14px; padding:25px 10px 23px 30px; border-bottom:1px solid #202b3d; margin:0 -20px 16px; }
.brand-mark { width:51px; height:51px; border-radius:15px; display:flex; align-items:center; justify-content:center; font-size:27px; color:#fff; background:linear-gradient(145deg,#19c9e6,#2872f1); box-shadow:0 7px 22px rgba(24,159,239,.22); }
.brand-title { font-size:18px; font-weight:700; color:#f7faff; line-height:1.1; }
.brand-subtitle { color:#7289a9; font-size:13px; margin-top:4px; }
.section-label { color:#597293; font-size:12px; letter-spacing:.5px; text-transform:uppercase; margin:22px 0 10px; }
.source-card { border:1px solid rgba(22,197,232,.4); background:rgba(22,197,232,.07); border-radius:10px; padding:12px 15px; color:var(--cyan); display:flex; justify-content:space-between; font-size:14px; }
.source-card span { color:#7188a7; font-size:12px; }
.sidebar-user { border-top:1px solid #202b3d; margin:25px -20px 0; padding:20px 30px 0; display:flex; gap:12px; align-items:center; }
.avatar { width:40px; height:40px; border-radius:50%; background:linear-gradient(135deg,#11b7db,#2b83fa); display:flex; align-items:center; justify-content:center; font-weight:700; color:#fff; }
.user-name { font-weight:600; color:#f5f7fb; font-size:14px; }
.user-role { color:#6f86a5; font-size:12px; margin-top:3px; }

/* page header */
.page-head { display:flex; align-items:center; justify-content:space-between; padding:27px 0 25px; border-bottom:1px solid #1a2536; margin-bottom:30px; }
.page-title { font-size:26px; font-weight:700; letter-spacing:-.5px; margin:0; }
.page-subtitle { color:#687d9b; font-size:16px; margin-top:5px; }
.status-pill { border:1px solid rgba(18,201,140,.35); color:var(--green); border-radius:24px; padding:9px 16px; font-size:13px; font-weight:600; background:rgba(18,201,140,.06); }
.status-pill.demo { border-color:rgba(244,183,64,.4); color:var(--amber); background:rgba(244,183,64,.06); }

/* KPI cards */
[data-testid="stMetric"] { background:var(--panel); border:1px solid var(--line); border-radius:15px; padding:24px 26px; min-height:137px; box-shadow:0 12px 28px rgba(0,0,0,.10); }
[data-testid="stMetricLabel"] { color:#8da4c4 !important; font-size:16px !important; }
[data-testid="stMetricValue"] { color:#fbfdff !important; font-size:38px !important; font-weight:700; margin-top:12px; }
[data-testid="stMetricDelta"] { color:var(--green) !important; }

/* content cards (st.container(border=True)) */
[data-testid="stVerticalBlockBorderWrapper"] { background:var(--panel); border:1px solid var(--line) !important; border-radius:15px; }
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] { background:transparent; border:0 !important; }
.card-title { color:#f7f9ff; font-size:20px; font-weight:700; margin-bottom:4px; }
.card-subtitle { color:#7e96b5; font-size:14px; margin-bottom:14px; }

/* validation finding cards */
.finding { background:#111a29; border:1px solid #263247; border-left:4px solid var(--amber); border-radius:12px; padding:14px 18px; margin-bottom:12px; }
.f-top { display:flex; align-items:center; gap:12px; flex-wrap:wrap; }
.badge { font-size:11px; font-weight:700; letter-spacing:.4px; text-transform:uppercase; padding:3px 10px; border-radius:20px; }
.f-check { color:#f3f7ff; font-weight:700; font-size:15px; }
.f-where { color:#8da4c4; font-size:13px; margin-left:auto; }
.f-detail { color:#c9d8ee; font-size:14px; margin-top:8px; line-height:1.5; }
.f-src { color:#5f7796; font-size:12px; margin-top:6px; }
.ok-card { background:rgba(18,201,140,.07); border:1px solid rgba(18,201,140,.35); color:var(--green); border-radius:12px; padding:16px 20px; font-weight:600; }

/* chat */
[data-testid="stChatMessage"] { background:#111a29; border:1px solid #263247; border-radius:12px; padding:14px 18px; }
[data-testid="stChatInput"] textarea { color:#f5f8ff !important; }
.chip { display:inline-block; background:rgba(22,197,232,.09); border:1px solid rgba(22,197,232,.3); color:var(--cyan); border-radius:20px; padding:3px 11px; font-size:12px; margin:6px 6px 0 0; }
.mode-tag { color:#5f7796; font-size:12px; margin-top:8px; }

/* widgets */
.stButton > button { border-radius:9px; border:1px solid #2d3c53; background:#182235; color:#dce9fa; font-weight:600; transition:all .2s ease; }
.stButton > button:hover { border-color:var(--cyan); color:var(--cyan); transform:translateY(-1px); }
.stButton > button[kind="primary"] { background:var(--cyan); color:#06111b; border:0; font-weight:700; }
.stDownloadButton > button { border-radius:9px; background:var(--cyan); color:#06111b; border:0; font-weight:700; }
.stTextInput input, .stNumberInput input, .stTextArea textarea { background:#111a29 !important; border:1px solid #293750 !important; color:#f5f8ff !important; border-radius:8px !important; }
.stFileUploader { background:#111a29; border-radius:10px; }
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:10px; }
.stAlert { border-radius:10px; }

@media (max-width: 900px) {
  .block-container { padding: 0 18px 30px; }
  .page-head { align-items:flex-start; gap:14px; flex-direction:column; }
  .status-pill { align-self:flex-start; }
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- Helpers
@contextmanager
def card(title: str, subtitle: str = ""):
    with st.container(border=True):
        st.markdown(
            f'<div class="card-title">{html.escape(title)}</div>'
            f'<div class="card-subtitle">{html.escape(subtitle)}</div>',
            unsafe_allow_html=True,
        )
        yield


def plot_layout(fig: go.Figure, height: int = 315) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#7d94b4"),
        legend=dict(font=dict(color="#8ca3c2")),
        xaxis=dict(showgrid=False, zeroline=False, linecolor="#202c40"),
        yaxis=dict(showgrid=True, gridcolor="#202c40", zeroline=False, linecolor="#202c40"),
        hoverlabel=dict(bgcolor="#172235", font_color="#f4f7ff"),
    )
    return fig


def show_chart(fig: go.Figure):
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


PALETTE = ["#16c5e8", "#3982f5", "#8b5cf6", "#12c98c", "#f4b740", "#f0566a", "#12bfd7", "#a78bfa"]
SEV_COLOR = {"High": "#f0566a", "Medium": "#f4b740", "Low": "#3982f5"}


# ---------------------------------------------------------------- Demo corpus (used when no documents are found)
@st.cache_resource
def demo_corpus() -> Corpus:
    months = pd.date_range("2025-04-01", periods=6, freq="MS")
    base = {
        "BCCL": [5.5, 5.9, 6.1, 5.6, 5.9, 6.2],
        "CCL": [7.2, 7.4, 7.7, 7.5, 7.9, 7.6],
        "ECL": [6.0, 6.1, 6.4, 6.2, 6.6, 6.5],
        "MCL": [5.9, 6.0, 6.1, 5.8, 6.0, 6.1],
        "NCL": [3.8, 4.0, 3.9, 4.2, 4.1, 3.7],
        "SECL": [2.9, 3.1, 3.0, 2.8, 3.3, 3.4],
    }
    rows = []
    for sub, vals in base.items():
        for i, (mo, act) in enumerate(zip(months, vals)):
            tgt = round(act * 1.04 + 0.1, 2)
            stated = round(act / tgt * 100, 1)
            if sub == "NCL" and i == 5:      # planted issue so Validation has something to show
                stated = 99.0
            rows.append(dict(Subsidiary=sub, Month=mo, Target_MT=tgt, Actual_MT=act,
                             Despatch_MT=round(act * 0.97, 2), OB_Removal_Mm3=round(act * 2.1, 2),
                             Stated_Ach_Pct=stated, Source=f"{sub}_Monthly_Return.pdf", Page=1))
    res = pd.DataFrame([
        dict(Subsidiary="MCL", Block="Basundhara", Seam="Seam-IV", Grade="G10", Proved_MT=420.5, Indicated_MT=130.0, Avg_Depth_m=210, Source="Reserve_Statement.xlsx", Page=1),
        dict(Subsidiary="SECL", Block="Gevra", Seam="Seam-II", Grade="G9", Proved_MT=380.2, Indicated_MT=95.0, Avg_Depth_m=180, Source="Reserve_Statement.xlsx", Page=1),
        dict(Subsidiary="NCL", Block="Jayant", Seam="Seam-I", Grade="G11", Proved_MT=150.8, Indicated_MT=60.5, Avg_Depth_m=140, Source="Reserve_Statement.xlsx", Page=1),
        dict(Subsidiary="CCL", Block="Piparwar", Seam="Seam-III", Grade="G8", Proved_MT=210.4, Indicated_MT=70.0, Avg_Depth_m=160, Source="Reserve_Statement.xlsx", Page=1),
    ])
    texts = [
        "Coal production remained steady across subsidiaries during the review period. Dispatch and logistics improved with rake availability.",
        "Reserve classification was updated for several blocks. Geological exploration continued in deeper seams.",
        "Land acquisition delays affected overburden removal in some projects. Safety compliance audits were completed on schedule.",
        "Sep 2025 NCL produced 3.90 MT against a target of 4.10 MT. Output was affected by monsoon disruptions.",
        "Production performance improved for CCL and ECL. Despatch to power plants stayed above production.",
        "Safety compliance and environmental clearance remained key themes. Mining plans were reviewed for opencast projects.",
    ]
    chunks = [dict(source="Sample_Narrative.docx", page=i + 1, text=t) for i, t in enumerate(texts)]
    c = Corpus(chunks=chunks, production=pd.DataFrame(rows), reserves=res,
               files=sorted({r["Source"] for r in rows} | {"Reserve_Statement.xlsx", "Sample_Narrative.docx"}), seconds=0.4)
    return c


# ---------------------------------------------------------------- Ingestion (cached)
@st.cache_resource(show_spinner="Reading and extracting documents…")
def ingest_uploaded(payload: tuple) -> Corpus:
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for i, (name, data) in enumerate(payload):
            folder = Path(tmp) / str(i)
            folder.mkdir()
            p = folder / Path(name).name
            p.write_bytes(data)
            paths.append(str(p))
        return ingest_paths(paths)


@st.cache_resource(show_spinner="Reading and extracting documents…")
def ingest_sample(paths: tuple, stamp: tuple) -> Corpus:  # stamp busts the cache when files change
    return ingest_paths(list(paths))


@st.cache_data(show_spinner=False)
def cached_topics(sig: str, _corpus: Corpus):
    try:
        return topics(_corpus)
    except Exception:
        return []


@st.cache_resource(show_spinner=False)
def get_qa(sig: str, _corpus: Corpus) -> QA:
    return QA(_corpus)


def sample_paths() -> list[Path]:
    if not DATA_DIR.exists():
        return []
    return sorted(p for p in DATA_DIR.rglob("*")
                  if p.is_file() and p.suffix.lower() in SAMPLE_EXT and "truth" not in p.name.lower())


def find_truth_csv() -> str:
    if not DATA_DIR.exists():
        return ""
    hit = next(iter(sorted(DATA_DIR.rglob("*truth*.csv"))), None)
    return str(hit) if hit else ""


# ---------------------------------------------------------------- Sidebar
with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-mark"><svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M13 2 3 14h9l-1 8 10-12h-9l1-8z"/></svg></div><div><div class="brand-title">CIL AI Assistant</div>'
        '<div class="brand-subtitle">Geological &amp; Mining Analytics</div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-label" style="margin-top:4px">Navigation</div>', unsafe_allow_html=True)
    selected = option_menu(
        None,
        ["Extraction", "Validation", "Report", "Word Cloud & Topics", "Ask AI", "KPIs"],
        icons=["file-earmark-text", "shield-check", "file-earmark-richtext", "cloud", "chat-square", "bar-chart"],
        default_index=0,
        styles={
            "container": {"padding": "0", "background-color": "transparent"},
            "icon": {"color": "#91a6c3", "font-size": "18px"},
            "nav-link": {
                "font-size": "15px", "font-weight": "600", "color": "#9bb0cc", "padding": "13px 16px",
                "margin": "5px 0", "border-radius": "12px", "border": "1px solid transparent",
                "--hover-color": "#131c2c",
            },
            "nav-link-selected": {
                "background-color": "rgba(22,197,232,.10)", "color": "#16c5e8",
                "border": "1px solid rgba(22,197,232,.28)",
                # cyan dot on the right edge of the active item
                "background-image": "radial-gradient(circle at calc(100% - 18px) 50%, #16c5e8 0 4px, transparent 5px)",
            },
        },
    )
    st.markdown('<div class="section-label">Document Source</div>', unsafe_allow_html=True)
    uploads = st.file_uploader("Upload documents", type=UPLOAD_TYPES, accept_multiple_files=True, label_visibility="collapsed")

    # decide which corpus to use: uploads > data/ folder > built-in demo
    corpus: Corpus | None = None
    source_label = "Sample CIL Documents"
    if uploads:
        payload = tuple((f.name, f.getvalue()) for f in uploads)
        corpus = ingest_uploaded(payload)
        source_label = "Uploaded Documents"
        sig = hashlib.md5("|".join(f"{n}:{len(d)}" for n, d in payload).encode()).hexdigest()
    else:
        paths = sample_paths()
        if paths:
            stamp = tuple((p.name, p.stat().st_mtime_ns) for p in paths)
            corpus = ingest_sample(tuple(str(p) for p in paths), stamp)
            sig = hashlib.md5(repr(stamp).encode()).hexdigest()

    using_demo = corpus is None or corpus.production.empty
    if using_demo:
        corpus = demo_corpus()
        source_label = "Built-in Demo Data"
        sig = "demo"

    st.markdown(f'<div class="source-card">{source_label} <span>{len(corpus.files)} files</span></div>', unsafe_allow_html=True)
    if using_demo and uploads:
        st.warning("No production tables were detected in the uploaded files, so demo data is shown.")

    st.markdown('<div class="section-label">Configuration</div>', unsafe_allow_html=True)
    baseline = st.number_input("Manual effort per report (hours)", min_value=1.0, max_value=100.0, value=6.0, step=0.5)
    st.markdown('<div class="sidebar-user"><div class="avatar">CM</div><div><div class="user-name">CMPDI Analyst</div><div class="user-role">SIH 2026 Prototype</div></div></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- Derived data
prod = corpus.prod_clean()
reserves = corpus.reserves
findings = validate(corpus)
ach_by_sub = pd.DataFrame()
if not prod.empty:
    ach_by_sub = prod.groupby("Subsidiary")[["Target_MT", "Actual_MT"]].sum()
    ach_by_sub["Achievement_%"] = ach_by_sub.Actual_MT / ach_by_sub.Target_MT * 100

# reset chat when the document set changes
if st.session_state.get("corpus_sig") != sig:
    st.session_state["corpus_sig"] = sig
    st.session_state["chat"] = []
    st.session_state.pop("report_bytes", None)


# ---------------------------------------------------------------- Page header
pill_class = "status-pill demo" if using_demo else "status-pill"
pill_text = "Demo data loaded" if using_demo else "All systems operational"
st.markdown(
    f'<div class="page-head"><div><div class="page-title">{html.escape(selected)}</div>'
    f'<div class="page-subtitle">SIH 2026 · CMPDI / CIL subsidiaries · offline-ready</div></div>'
    f'<div class="{pill_class}">● &nbsp;{pill_text}</div></div>',
    unsafe_allow_html=True,
)


# ================================================================ EXTRACTION
if selected == "Extraction":
    c1, c2, c3 = st.columns(3)
    c1.metric("Files Read", len(corpus.files))
    c2.metric("Production Records", len(prod))
    c3.metric("Reserve Records", len(reserves))
    st.write("")

    left, right = st.columns([1.15, 1])
    with left:
        with card("Production Trends", "Monthly actual output by subsidiary (MT)"):
            fig = go.Figure()
            for i, (sub, g) in enumerate(prod.groupby("Subsidiary")):
                col = PALETTE[i % len(PALETTE)]
                fig.add_trace(go.Scatter(
                    x=g["Month"], y=g["Actual_MT"], name=sub, mode="lines+markers",
                    line=dict(color=col, width=2.5, shape="spline"),
                    marker=dict(size=6, color=col, line=dict(color="#dce8ff", width=1.5)),
                ))
            fig.update_xaxes(tickformat="%b %Y")
            show_chart(plot_layout(fig))
    with right:
        with card("Geological Reserves", "Proved vs indicated reserves by subsidiary (MT)"):
            if reserves.empty or "Proved_MT" not in reserves:
                st.info("No reserve tables were found in the loaded documents.")
            else:
                rs = reserves.groupby("Subsidiary")[[c for c in ["Proved_MT", "Indicated_MT"] if c in reserves]].sum().reset_index()
                fig = go.Figure()
                fig.add_bar(x=rs["Subsidiary"], y=rs["Proved_MT"], name="Proved", marker_color="#3982f5")
                if "Indicated_MT" in rs:
                    fig.add_bar(x=rs["Subsidiary"], y=rs["Indicated_MT"], name="Indicated", marker_color="#8b5cf6")
                fig.update_layout(barmode="stack")
                show_chart(plot_layout(fig))
    st.write("")

    with card("Extracted Production Data", "Structured values with source-ready records"):
        if prod.empty:
            st.info("No production tables were detected.")
        else:
            view = prod.copy()
            view["Achievement %"] = (view.Actual_MT / view.Target_MT * 100).round(1)
            view["Month"] = view.Month.dt.strftime("%b %Y")
            cols = [c for c in ["Subsidiary", "Month", "Target_MT", "Actual_MT", "Achievement %", "Despatch_MT", "OB_Removal_Mm3", "Source", "Page"] if c in view]
            st.dataframe(view[cols], use_container_width=True, hide_index=True)
    if not reserves.empty:
        st.write("")
        with card("Extracted Reserve Data", "Block-level reserve records"):
            st.dataframe(reserves, use_container_width=True, hide_index=True)
    with st.expander(f"Extracted text passages ({len(corpus.chunks)})"):
        st.dataframe(pd.DataFrame(corpus.chunks), use_container_width=True, hide_index=True)


# ================================================================ VALIDATION
elif selected == "Validation":
    n_high = int((findings.Severity == "High").sum()) if not findings.empty else 0
    n_med = int((findings.Severity == "Medium").sum()) if not findings.empty else 0
    flagged = findings[["Subsidiary", "Month"]].drop_duplicates().shape[0] if not findings.empty else 0
    clean = max(len(prod) - flagged, 0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records checked", len(prod))
    c2.metric("Clean records", clean)
    c3.metric("High severity", n_high)
    c4.metric("Medium severity", n_med)
    st.write("")

    with card("Automatic consistency checks", "Review flagged items before submission to the Ministry"):
        if findings.empty:
            st.markdown('<div class="ok-card">✓ All checks passed — no inconsistencies detected.</div>', unsafe_allow_html=True)
        else:
            f1, f2 = st.columns([2, 1])
            sev_sel = f1.multiselect("Severity", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
            checks = sorted(findings.Check.unique())
            chk_sel = f2.multiselect("Check type", checks, default=checks)
            shown = findings[findings.Severity.isin(sev_sel) & findings.Check.isin(chk_sel)]
            st.caption(f"Showing {len(shown)} of {len(findings)} findings")
            for r in shown.itertuples():
                col = SEV_COLOR.get(r.Severity, "#7790af")
                st.markdown(
                    f'<div class="finding" style="border-left-color:{col}">'
                    f'<div class="f-top"><span class="badge" style="background:{col}22;color:{col}">{html.escape(r.Severity)}</span>'
                    f'<span class="f-check">{html.escape(r.Check)}</span>'
                    f'<span class="f-where">{html.escape(str(r.Subsidiary))} · {html.escape(str(r.Month))}</span></div>'
                    f'<div class="f-detail">{html.escape(r.Detail)}</div>'
                    f'<div class="f-src">Source: {html.escape(str(r.Source))}</div></div>',
                    unsafe_allow_html=True,
                )
            with st.expander("View as table"):
                st.dataframe(shown, use_container_width=True, hide_index=True)


# ================================================================ REPORT
elif selected == "Report":
    tps = cached_topics(sig, corpus)
    with card("Auto-generated Word report", "Create a concise reporting package from the extracted records"):
        if prod.empty:
            st.info("Upload documents with production tables to generate a report.")
        else:
            tot_a, tot_t = prod.Actual_MT.sum(), prod.Target_MT.sum()
            best, worst = ach_by_sub["Achievement_%"].idxmax(), ach_by_sub["Achievement_%"].idxmin()
            st.markdown(
                f"**Executive summary preview** — Across **{prod.Subsidiary.nunique()}** subsidiaries, production for "
                f"{prod.Month.min():%b %Y}–{prod.Month.max():%b %Y} was **{tot_a:,.1f} MT** against a target of "
                f"{tot_t:,.1f} MT (**{tot_a / tot_t * 100:.1f}%** achievement). Best performer: **{best}** "
                f"({ach_by_sub.loc[best, 'Achievement_%']:.1f}%); most behind: **{worst}** ({ach_by_sub.loc[worst, 'Achievement_%']:.1f}%). "
                f"{len(findings)} data-quality flag(s) will be included."
            )
            st.write("")
            if st.button("Generate report", type="primary"):
                with st.spinner("Building Word report…"):
                    with tempfile.TemporaryDirectory() as tmp:
                        out = build_report(corpus, findings, tps, str(Path(tmp) / "CIL_Report.docx"))
                        st.session_state["report_bytes"] = Path(out).read_bytes()
                st.success("Report ready for download.")
            if "report_bytes" in st.session_state:
                st.download_button(
                    "Download report (.docx)", st.session_state["report_bytes"], "CIL_Report.docx",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
    if not prod.empty:
        st.write("")
        with card("Subsidiary-wise performance", "Target vs actual over the reporting period"):
            tbl = ach_by_sub.reset_index().rename(columns={"Target_MT": "Target (MT)", "Actual_MT": "Actual (MT)", "Achievement_%": "Achievement (%)"})
            st.dataframe(tbl.round(2), use_container_width=True, hide_index=True)


# ================================================================ WORD CLOUD & TOPICS
elif selected == "Word Cloud & Topics":
    freq = word_freq(corpus, top=60)
    left, right = st.columns([1.15, 1])
    with left:
        with card("Document themes", "Most frequent terms across the loaded documents"):
            if not freq:
                st.info("No narrative text was found in the loaded documents.")
            else:
                cloud = None
                try:
                    from wordcloud import WordCloud
                    cloud = WordCloud(width=900, height=420, background_color="#151d2d", colormap="cool").generate_from_frequencies(freq).to_image()
                except Exception:
                    cloud = None
                if cloud is not None:
                    st.image(cloud, use_container_width=True)
                else:
                    top = pd.DataFrame({"Term": list(freq)[:12], "Mentions": list(freq.values())[:12]}).sort_values("Mentions")
                    fig = go.Figure(go.Bar(x=top["Mentions"], y=top["Term"], orientation="h", marker_color="#16c5e8"))
                    fig.update_layout(yaxis=dict(showgrid=False))
                    show_chart(plot_layout(fig, 350))
    with right:
        with card("Topics identified", "Themes grouped from source material"):
            tps = cached_topics(sig, corpus)
            if tps:
                st.dataframe(pd.DataFrame(tps), use_container_width=True, hide_index=True)
            else:
                st.info("Not enough text to identify distinct topics.")
    if freq:
        st.write("")
        with card("Top terms", "Frequency of the most common words"):
            top = pd.DataFrame({"Term": list(freq)[:15], "Mentions": list(freq.values())[:15]})
            fig = go.Figure(go.Bar(x=top["Term"], y=top["Mentions"], marker_color="#3982f5"))
            show_chart(plot_layout(fig, 280))


# ================================================================ ASK AI
elif selected == "Ask AI":
    qa = get_qa(sig, corpus)
    history = st.session_state.setdefault("chat", [])

    with card("AI query & response", "Ask a parliamentary-question style question about the loaded documents"):
        examples = ["What was SECL production in August?", "Total shortfall for NCL", "What are the proved reserves for MCL?"]
        cols = st.columns(len(examples) + 1)
        clicked = None
        for col, ex in zip(cols, examples):
            if col.button(ex, use_container_width=True):
                clicked = ex
        if cols[-1].button("Clear chat", use_container_width=True):
            st.session_state["chat"] = []
            st.rerun()
        st.write("")

        for msg in history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant":
                    if msg.get("table") is not None:
                        with st.expander("Supporting data"):
                            st.dataframe(msg["table"], use_container_width=True, hide_index=True)
                    if msg.get("sources"):
                        st.markdown("".join(f'<span class="chip">{html.escape(s)}</span>' for s in msg["sources"]), unsafe_allow_html=True)
                    st.markdown(f'<div class="mode-tag">Mode: {msg.get("mode", "")}</div>', unsafe_allow_html=True)

    question = st.chat_input("Ask about production, targets, despatch, reserves…") or clicked
    if question:
        history.append({"role": "user", "content": question})
        try:
            res = qa.ask(question)
        except Exception as exc:
            res = {"answer": f"Sorry, I could not answer that ({exc}).", "sources": [], "mode": "error"}
        table = res.get("table")
        if isinstance(table, pd.DataFrame) and "Month" in table.columns and pd.api.types.is_datetime64_any_dtype(table["Month"]):
            table = table.assign(Month=table["Month"].dt.strftime("%b %Y"))
        history.append({"role": "assistant", "content": res["answer"], "sources": res.get("sources", []),
                        "mode": res.get("mode", ""), "table": table})
        st.rerun()


# ================================================================ KPIs
elif selected == "KPIs":
    try:
        k = kpis(corpus, find_truth_csv(), baseline_hours=baseline)
    except Exception:
        k = {}
    acc = k.get("Extraction accuracy (%)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Extraction accuracy", f"{acc:.1f}%" if acc is not None else "n/a")
    c2.metric("Report time reduction", f"{k.get('Report time reduction (%)', 0):.2f}%")
    c3.metric("Pipeline runtime", f"{k.get('Pipeline runtime (s)', 0):.1f} s", f"vs {baseline:g} h manual")
    c4.metric("Workflow automation", f"{k.get('Workflow automation (%)', 0):.1f}%")
    if acc is None:
        st.caption("Extraction accuracy needs a ground-truth CSV (a file with “truth” in its name inside the data/ folder).")
    st.write("")

    left, right = st.columns([1.15, 1])
    with left:
        with card("Achievement by subsidiary", "Actual production as % of target"):
            if ach_by_sub.empty:
                st.info("No production data available.")
            else:
                a = ach_by_sub.reset_index()
                fig = go.Figure(go.Bar(x=a["Subsidiary"], y=a["Achievement_%"], marker_color="#16c5e8",
                                       text=a["Achievement_%"].round(1), textposition="outside"))
                fig.add_hline(y=100, line_dash="dash", line_color="#f4b740", annotation_text="Target", annotation_font_color="#f4b740")
                fig.update_yaxes(range=[0, max(110, a["Achievement_%"].max() + 10)])
                show_chart(plot_layout(fig, 340))
    with right:
        with card("Pilot performance snapshot", "Automated vs human-in-the-loop workflow steps"):
            steps = pd.DataFrame({
                "Step": ["Ingest", "Extract", "Validate", "Chart", "Draft report", "Topics", "Q&A", "Final review"],
                "Handled by": ["Automated"] * 7 + ["Human"],
            })
            st.dataframe(steps, use_container_width=True, hide_index=True)
            manual_min = baseline * 60
            auto_min = k.get("Pipeline runtime (s)", 0) / 60
            st.caption(f"Manual effort: {manual_min:,.0f} min · automated pipeline: {auto_min:.2f} min. "
                       f"Total extracted production: {prod.Actual_MT.sum():,.1f} MT.")
