"""Core engine: ingest -> extract -> validate -> report -> topics -> Q&A -> KPIs.
Runs fully offline. If ANTHROPIC_API_KEY is set, Q&A / summary can use an LLM for nicer wording."""
import os, re, time, io
from dataclasses import dataclass, field
import numpy as np, pandas as pd
from collections import Counter

SUB_RE = re.compile(r"\b(SECL|NCL|WCL|MCL|BCCL|CCL|ECL|NEC|CMPDI)\b")
MONTH_RE = re.compile(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?,? (\d{4})\b")


def _num(x):
    try:
        return float(str(x).replace(",", "").replace("%", "").strip())
    except Exception:
        return np.nan


def _month(x):
    m = MONTH_RE.search(str(x))
    if not m:
        return pd.NaT
    return pd.to_datetime(f"{m.group(1)} {m.group(2)}", format="%b %Y")


@dataclass
class Corpus:
    chunks: list = field(default_factory=list)          # dict(source, page, text)
    production: pd.DataFrame = field(default_factory=pd.DataFrame)
    reserves: pd.DataFrame = field(default_factory=pd.DataFrame)
    files: list = field(default_factory=list)
    seconds: float = 0.0

    def prod_clean(self):
        if self.production.empty:
            return self.production
        return self.production.drop_duplicates(["Subsidiary", "Month"]).sort_values(["Subsidiary", "Month"]).reset_index(drop=True)


# ---------------------------------------------------------------- ingestion
def _classify_table(rows, sub_hint, source, page, prod, res):
    if not rows or len(rows) < 2:
        return
    head = [re.sub(r"\s+", " ", str(h or "")).strip().lower() for h in rows[0]]
    joined = " ".join(head)
    body = rows[1:]
    if "target" in joined and "actual" in joined:
        idx = {}
        for i, h in enumerate(head):
            if "month" in h: idx["Month"] = i
            elif "target" in h: idx["Target_MT"] = i
            elif "actual" in h or "production" in h: idx["Actual_MT"] = i
            elif "despatch" in h or "dispatch" in h: idx["Despatch_MT"] = i
            elif re.search(r"\bob\b|overburden", h): idx["OB_Removal_Mm3"] = i
            elif "achiev" in h: idx["Stated_Ach_Pct"] = i
            elif "subsidiary" in h or "company" in h: idx["Subsidiary"] = i
        for r in body:
            if "Month" not in idx:
                continue
            mo = _month(r[idx["Month"]])
            if pd.isna(mo):
                continue
            rec = {"Subsidiary": (r[idx["Subsidiary"]] if "Subsidiary" in idx else sub_hint) or "UNKNOWN", "Month": mo,
                   "Source": source, "Page": page}
            for k in ["Target_MT", "Actual_MT", "Despatch_MT", "OB_Removal_Mm3", "Stated_Ach_Pct"]:
                rec[k] = _num(r[idx[k]]) if k in idx else np.nan
            prod.append(rec)
    elif "proved" in joined or "reserve" in joined:
        for r in body:
            d = dict(zip(head, r))
            rec = {"Source": source, "Page": page}
            for h, v in d.items():
                key = ("Subsidiary" if "subsidiary" in h else "Block" if "block" in h else "Seam" if "seam" in h else
                       "Grade" if "grade" in h else "Proved_MT" if "proved" in h else "Indicated_MT" if "indicated" in h else
                       "Avg_Depth_m" if "depth" in h else None)
                if key: rec[key] = v
            rec.setdefault("Subsidiary", sub_hint)
            res.append(rec)


def _chunk_text(text, source, page):
    text = re.sub(r"\s+", " ", text or "").strip()
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    out = []
    for i in range(0, len(sents), 2):
        t = " ".join(sents[i:i + 2]).strip()
        if len(t) > 25:
            out.append({"source": source, "page": page, "text": t})
    return out


def _ocr(img):
    try:
        import pytesseract
        return pytesseract.image_to_string(img)
    except Exception:
        return ""


def ingest_paths(paths):
    t0 = time.time()
    c = Corpus(); prod, res = [], []
    for p in paths:
        name = os.path.basename(p); ext = name.lower().rsplit(".", 1)[-1]
        c.files.append(name)
        sub_hint = (SUB_RE.search(name) or [None])[0] if SUB_RE.search(name) else None
        try:
            if ext == "pdf":
                import pdfplumber
                with pdfplumber.open(p) as pdf:
                    if not sub_hint:
                        m = SUB_RE.search(pdf.pages[0].extract_text() or ""); sub_hint = m.group(1) if m else None
                    for i, pg in enumerate(pdf.pages, 1):
                        txt = pg.extract_text() or ""
                        if not txt.strip():                                 # scanned page -> OCR
                            try: txt = _ocr(pg.to_image(resolution=200).original)
                            except Exception: txt = ""
                        c.chunks += _chunk_text(txt, name, i)
                        for tb in pg.extract_tables():
                            _classify_table(tb, sub_hint, name, i, prod, res)
            elif ext == "docx":
                import docx
                d = docx.Document(p)
                c.chunks += _chunk_text(" ".join(x.text for x in d.paragraphs), name, 1)
                for tb in d.tables:
                    _classify_table([[cell.text for cell in r.cells] for r in tb.rows], sub_hint, name, 1, prod, res)
            elif ext in ("xlsx", "xls", "csv"):
                sheets = {"csv": pd.read_csv(p)} if ext == "csv" else pd.read_excel(p, sheet_name=None)
                for sn, df in sheets.items():
                    _classify_table([df.columns.tolist()] + df.astype(object).values.tolist(), sub_hint, name, sn, prod, res)
            elif ext in ("png", "jpg", "jpeg", "tif", "tiff"):
                from PIL import Image
                c.chunks += _chunk_text(_ocr(Image.open(p)), name, 1)
            elif ext == "txt":
                c.chunks += _chunk_text(open(p, encoding="utf-8", errors="ignore").read(), name, 1)
        except Exception as e:
            c.chunks.append({"source": name, "page": 0, "text": f"[could not fully read file: {e}]"})
    c.production = pd.DataFrame(prod)
    c.reserves = pd.DataFrame(res)
    for col in ["Proved_MT", "Indicated_MT", "Avg_Depth_m"]:
        if col in c.reserves: c.reserves[col] = c.reserves[col].map(_num)
    c.seconds = time.time() - t0
    return c


# ---------------------------------------------------------------- validation
NARR_RE = re.compile(r"(?P<m>[A-Z][a-z]{2,8} \d{4}),? (?P<s>[A-Z]{2,5}) produced (?P<a>[\d.,]+) MT against a target of (?P<t>[\d.,]+) MT")


def validate(c: Corpus) -> pd.DataFrame:
    F = []
    def add(sev, typ, sub, mo, detail, src):
        F.append({"Severity": sev, "Check": typ, "Subsidiary": sub, "Month": mo.strftime("%b %Y") if pd.notna(mo) else "", "Detail": detail, "Source": src})
    df = c.production
    if df.empty:
        return pd.DataFrame(F)
    for _, r in df.iterrows():
        src = f"{r.Source} p.{r.Page}"
        if pd.isna(r.Actual_MT) or pd.isna(r.Target_MT):
            add("High", "Missing value", r.Subsidiary, r.Month, "Target/Actual missing", src); continue
        calc = r.Actual_MT / r.Target_MT * 100
        if pd.notna(r.Stated_Ach_Pct) and abs(calc - r.Stated_Ach_Pct) > 0.5:
            add("High", "Achievement % mismatch", r.Subsidiary, r.Month, f"Stated {r.Stated_Ach_Pct:.1f}% but Actual/Target = {calc:.1f}%", src)
        if pd.notna(r.Despatch_MT) and r.Despatch_MT > 1.15 * r.Actual_MT:
            add("Medium", "Despatch exceeds production", r.Subsidiary, r.Month, f"Despatch {r.Despatch_MT:.2f} vs production {r.Actual_MT:.2f} MT", src)
    for sub, g in df.drop_duplicates(["Subsidiary", "Month"]).groupby("Subsidiary"):
        med = g.Actual_MT.median()
        for _, r in g.iterrows():
            if med > 0 and (r.Actual_MT < 0.8 * med or r.Actual_MT > 1.25 * med):
                add("Medium", "Unusual variation", sub, r.Month, f"Actual {r.Actual_MT:.2f} MT is {r.Actual_MT / med * 100 - 100:+.0f}% vs the subsidiary median ({med:.2f} MT)", f"{r.Source} p.{r.Page}")
    conf = df.groupby(["Subsidiary", "Month"]).Actual_MT.nunique()
    for (sub, mo), n in conf.items():
        if n > 1: add("High", "Conflict across documents", sub, mo, "Different Actual values in different sources", "multiple")
    tb = df.drop_duplicates(["Subsidiary", "Month"]).set_index(["Subsidiary", "Month"])
    for ch in c.chunks:
        for m in NARR_RE.finditer(ch["text"]):
            mo = _month(m["m"]); key = (m["s"], mo)
            if key in tb.index:
                a = _num(m["a"]); ta = tb.loc[key, "Actual_MT"]
                if abs(a - ta) > 0.005:
                    add("High", "Narrative vs table mismatch", m["s"], mo, f"Text says {a:.2f} MT, table says {ta:.2f} MT", f"{ch['source']} p.{ch['page']}")
    out = pd.DataFrame(F)
    if not out.empty:
        out = out.assign(_o=out.Severity.map({"High": 0, "Medium": 1, "Low": 2})).sort_values("_o").drop(columns="_o").reset_index(drop=True)
    return out


# ---------------------------------------------------------------- topics / word cloud
STOP = set("""a an the of and or to in on for at by with from as is are was were be been this that these those it its into during
against month year mt mm3 produced target achievement fy per two one also which has have had than over under more less synthetic demo prepared ministry monthly review annexure statistics""".split())


def _clean(t):
    return re.sub(r"[^a-z\s]", " ", t.lower())


def word_freq(c: Corpus, top=80):
    cnt = Counter()
    for ch in c.chunks:
        for w in _clean(ch["text"]).split():
            if len(w) > 3 and w not in STOP and not re.fullmatch(r"(april|may|june|july|august|september|secl|ncl|wcl|mcl)", w):
                cnt[w] += 1
    return dict(cnt.most_common(top))


def wordcloud_image(freq):
    try:
        from wordcloud import WordCloud
        return WordCloud(width=900, height=420, background_color="white", colormap="viridis").generate_from_frequencies(freq).to_image()
    except Exception:
        return None


def topics(c: Corpus, n_topics=4, n_words=6):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import NMF
    docs = [_clean(ch["text"]) for ch in c.chunks if not ch["text"].startswith("[could not")]
    if len(docs) < 4: return []
    vec = TfidfVectorizer(stop_words=list(STOP | set(["april", "may", "june", "july", "august", "september", "secl", "ncl", "wcl", "mcl"])),
                          min_df=2, max_df=0.9, token_pattern=r"[a-z]{4,}")
    X = vec.fit_transform(docs)
    k = min(n_topics, X.shape[0] - 1, X.shape[1])
    if k < 2: return []
    nmf = NMF(n_components=k, random_state=0, init="nndsvd", max_iter=400).fit(X)
    terms = vec.get_feature_names_out(); W = nmf.transform(X); share = np.bincount(W.argmax(1), minlength=k) / len(docs)
    return [{"Topic": i + 1, "Keywords": ", ".join(terms[j] for j in comp.argsort()[::-1][:n_words]), "Share of text": f"{share[i] * 100:.0f}%"}
            for i, comp in enumerate(nmf.components_)]


# ---------------------------------------------------------------- Q&A
MON = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def _llm(question, context):
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key: return None
    try:
        import anthropic
        r = anthropic.Anthropic(api_key=key).messages.create(
            model=os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5"), max_tokens=500,
            messages=[{"role": "user", "content": f"Answer ONLY from this context, cite sources in brackets, be concise.\n\nContext:\n{context}\n\nQuestion: {question}"}])
        return r.content[0].text
    except Exception:
        return None


class QA:
    def __init__(self, c: Corpus):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.c = c; self.texts = [x["text"] for x in c.chunks]
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)); self.M = self.vec.fit_transform(self.texts) if self.texts else None

    def ask(self, q):
        ql = q.lower(); df = self.c.prod_clean()
        subs = [s for s in SUB_RE.findall(q.upper()) if s != "CMPDI"]
        months = [MON[m[:3]] for m in re.findall(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*", ql)]
        if "reserve" in ql and not self.c.reserves.empty:
            r = self.c.reserves.copy()
            if subs: r = r[r.Subsidiary.isin(subs)]
            tot = r.Proved_MT.sum(); top = r.sort_values("Proved_MT", ascending=False).head(3)
            ans = f"Proved reserves{' for ' + ', '.join(subs) if subs else ''}: **{tot:,.1f} MT** across {len(r)} blocks. Largest: " + \
                  "; ".join(f"{b.Block} ({b.Proved_MT:,.1f} MT, {b.Grade})" for b in top.itertuples())
            return {"answer": ans, "sources": sorted(set(r.Source)), "mode": "structured", "table": r}
        if not df.empty and re.search(r"produc|target|despatch|dispatch|shortfall|achiev|overburden|ob removal|total", ql):
            d = df
            if subs: d = d[d.Subsidiary.isin(subs)]
            if months: d = d[d.Month.dt.month.isin(months)]
            if not d.empty:
                a, t, ds = d.Actual_MT.sum(), d.Target_MT.sum(), d.Despatch_MT.sum()
                scope = f"{', '.join(subs) or 'All subsidiaries'}{' / ' + ', '.join(sorted(set(d.Month.dt.strftime('%b %Y')))) if months else ''}"
                ans = (f"{scope}: production **{a:,.2f} MT** against target {t:,.2f} MT (**{a / t * 100:.1f}%** achievement, "
                       f"{'shortfall' if t >= a else 'surplus'} {abs(t - a):,.2f} MT); despatch {ds:,.2f} MT.")
                out = d[["Subsidiary", "Month", "Target_MT", "Actual_MT", "Despatch_MT", "Source", "Page"]].copy(); out["Month"] = out.Month.dt.strftime("%b %Y")
                return {"answer": ans, "sources": sorted({f"{s} p.{p}" for s, p in zip(d.Source, d.Page)}), "mode": "structured", "table": out}
        if self.M is None: return {"answer": "No documents loaded.", "sources": [], "mode": "none"}
        sc = (self.M @ self.vec.transform([q]).T).toarray().ravel(); top = sc.argsort()[::-1][:3]; top = [i for i in top if sc[i] > 0.05]
        if not top: return {"answer": "I could not find this in the uploaded documents.", "sources": [], "mode": "none"}
        ctx = "\n".join(f"[{self.c.chunks[i]['source']} p.{self.c.chunks[i]['page']}] {self.texts[i]}" for i in top)
        return {"answer": _llm(q, ctx) or "Most relevant passages:\n\n" + "\n\n".join(f"- {self.texts[i]}" for i in top),
                "sources": [f"{self.c.chunks[i]['source']} p.{self.c.chunks[i]['page']}" for i in top], "mode": "retrieval"}


# ---------------------------------------------------------------- report
def build_report(c: Corpus, findings: pd.DataFrame, tps, path):
    import docx, matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from docx.shared import Inches
    df = c.prod_clean(); d = docx.Document()
    d.add_heading("Coal Production & Geological Status Report", 0)
    d.add_paragraph("Auto-generated draft for review. Every figure is traceable to its source document (see Annexure).")
    tot_a, tot_t = df.Actual_MT.sum(), df.Target_MT.sum()
    bysub = df.groupby("Subsidiary")[["Actual_MT", "Target_MT"]].sum(); bysub["Ach"] = bysub.Actual_MT / bysub.Target_MT * 100
    d.add_heading("1. Executive summary", 1)
    d.add_paragraph(f"Across {df.Subsidiary.nunique()} subsidiaries, production for {df.Month.min():%b %Y}-{df.Month.max():%b %Y} was "
                    f"{tot_a:,.1f} MT against a target of {tot_t:,.1f} MT ({tot_a / tot_t * 100:.1f}% achievement). "
                    f"Best performer: {bysub.Ach.idxmax()} ({bysub.Ach.max():.1f}%); most behind: {bysub.Ach.idxmin()} ({bysub.Ach.min():.1f}%).")
    if not c.reserves.empty:
        d.add_paragraph(f"Proved reserves in the uploaded statement: {c.reserves.Proved_MT.sum():,.1f} MT across {len(c.reserves)} blocks.")
    if tps: d.add_paragraph("Key themes in the narrative: " + "; ".join(t["Keywords"].split(", ")[0] + " (" + t["Share of text"] + ")" for t in tps) + ".")
    d.add_heading("2. Subsidiary-wise performance", 1)
    t = d.add_table(rows=1, cols=4); t.style = "Light Grid Accent 1"
    for i, h in enumerate(["Subsidiary", "Target (MT)", "Actual (MT)", "Achievement (%)"]): t.rows[0].cells[i].text = h
    for s, r in bysub.iterrows():
        cells = t.add_row().cells; cells[0].text = s; cells[1].text = f"{r.Target_MT:,.2f}"; cells[2].text = f"{r.Actual_MT:,.2f}"; cells[3].text = f"{r.Ach:.1f}"
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for s, g in df.groupby("Subsidiary"): ax.plot(g.Month, g.Actual_MT, marker="o", label=s)
    ax.set_ylabel("Actual production (MT)"); ax.legend(); ax.grid(alpha=.3); fig.autofmt_xdate(); buf = io.BytesIO(); fig.savefig(buf, dpi=150, bbox_inches="tight"); buf.seek(0); plt.close(fig)
    d.add_picture(buf, width=Inches(6))
    d.add_heading("3. Data-quality & consistency flags", 1)
    if findings.empty: d.add_paragraph("No issues detected.")
    else:
        t = d.add_table(rows=1, cols=4); t.style = "Light Grid Accent 1"
        for i, h in enumerate(["Severity", "Check", "Where", "Detail"]): t.rows[0].cells[i].text = h
        for r in findings.itertuples():
            cells = t.add_row().cells; cells[0].text = r.Severity; cells[1].text = r.Check; cells[2].text = f"{r.Subsidiary} {r.Month}"; cells[3].text = r.Detail
    d.add_heading("Annexure - source traceability", 1)
    for f, g in df.groupby("Source"): d.add_paragraph(f"{f}: {len(g)} monthly records extracted (pages {', '.join(map(str, sorted(set(g.Page))))})", style="List Bullet")
    d.save(path); return path


# ---------------------------------------------------------------- KPIs
def kpis(c: Corpus, truth_csv, baseline_hours=6.0, report_seconds=5.0):
    out = {}
    if os.path.exists(truth_csv) and not c.production.empty:
        tr = pd.read_csv(truth_csv); tr["Month"] = pd.to_datetime(tr.Month, format="%b %Y")
        m = tr.merge(c.prod_clean(), on=["Subsidiary", "Month"], how="left", suffixes=("_t", ""))
        cols = ["Target_MT", "Actual_MT", "Despatch_MT", "OB_Removal_Mm3"]; ok = sum(int((m[k] - m[k + "_t"]).abs().lt(0.005).sum()) for k in cols)
        out["Extraction accuracy (%)"] = round(ok / (len(tr) * len(cols)) * 100, 1)
    manual = baseline_hours * 3600; auto = c.seconds + report_seconds
    out["Report time reduction (%)"] = round((1 - auto / manual) * 100, 2)
    out["Pipeline runtime (s)"] = round(auto, 1)
    steps = ["Ingest", "Extract", "Validate", "Chart", "Draft report", "Topics", "Q&A"]; human = ["Final review"]
    out["Workflow automation (%)"] = round(len(steps) / (len(steps) + len(human)) * 100, 1)
    return out
