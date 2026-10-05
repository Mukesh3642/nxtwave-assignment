"""
Placement Resume Gap Analyzer  —  NxtWave Growth Challenge asset
------------------------------------------------------------------
Lead magnet for the free workshop "Build Your First AI Project in 60 Minutes".

Funnel:  scan resume (free value)  ->  skill-credibility flags + radar vs placed benchmark
         ->  personalised workshop project CTA  ->  phone + consent to unlock full PDF report
         ->  webhook (n8n / Zapier -> WhatsApp API)  ->  referral link  ->  growth dashboard.
"""

import base64
import csv
import hashlib
import io
import os
import re
from datetime import datetime
from urllib.parse import quote

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

# Must be the FIRST Streamlit command in the script
st.set_page_config(page_title="Placement Resume Gap Analyzer", page_icon="🎯", layout="wide")
from fpdf import FPDF
from fpdf.enums import XPos, YPos

# ─────────────────────────────── CONFIG ───────────────────────────────


def _secrets_file_exists() -> bool:
    """Only touch st.secrets if a secrets.toml exists, otherwise Streamlit prints 'No secrets found' errors."""
    paths = [
        os.path.join(os.path.expanduser("~"), ".streamlit", "secrets.toml"),
        os.path.join(os.getcwd(), ".streamlit", "secrets.toml"),
    ]
    return any(os.path.exists(p) for p in paths)


def cfg(key: str, default: str = "") -> str:
    """Read from Streamlit secrets (if a secrets file exists), then environment, then default."""
    if _secrets_file_exists():
        try:
            if key in st.secrets:
                return str(st.secrets[key])
        except Exception:
            pass
    return os.getenv(key, default)


APP_URL = cfg("APP_URL", "https://placement-gap.streamlit.app")
WORKSHOP_URL = cfg("WORKSHOP_URL", "https://www.ccbp.in/")  # replace with real registration page
WA_BUSINESS_NUMBER = cfg("WA_BUSINESS_NUMBER", "919999999999")  # NxtWave WhatsApp number (no +)
WEBHOOK_URL = cfg("WEBHOOK_URL", "")  # n8n / Zapier / Make catch-hook
ADMIN_PASSWORD = cfg("ADMIN_PASSWORD", "nxtwave500")
TARGET_REGISTRATIONS = 500

DATA_DIR = "data"
LEADS_CSV = os.path.join(DATA_DIR, "leads.csv")
EVENTS_CSV = os.path.join(DATA_DIR, "events.csv")
os.makedirs(DATA_DIR, exist_ok=True)

# ───────────────────────────── MODELS ────────────────────────────────
from engine import BENCHMARKS, PROJECT_TRACKS, SAMPLE_RESUME, SkillClassifier, analyse


@st.cache_resource(show_spinner="Loading light skill model…")
def load_model():
    return SkillClassifier(prefer_embeddings=cfg("USE_EMBEDDINGS", "0") == "1")


def extract_text(uploaded) -> str:
    data = uploaded.read()
    name = uploaded.name.lower()
    if name.endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    if name.endswith(".docx"):
        import docx

        d = docx.Document(io.BytesIO(data))
        parts = [p.text for p in d.paragraphs]
        for table in d.tables:
            for row in table.rows:
                parts.extend(c.text for c in row.cells)
        return "\n".join(parts)
    return data.decode("utf-8", errors="ignore")


# ───────────────────────────── VISUALS ───────────────────────────────


def radar_chart(res: dict):
    cats = list(res["scores"].keys())
    you = [res["scores"][c] for c in cats]
    bench = [res["bench"][c] for c in cats]
    angles = np.linspace(0, 2 * np.pi, len(cats), endpoint=False).tolist()
    you_c, bench_c, ang_c = you + you[:1], bench + bench[:1], angles + angles[:1]

    fig, ax = plt.subplots(figsize=(5.6, 5.6), subplot_kw=dict(polar=True))
    ax.plot(ang_c, bench_c, color="#16a34a", linewidth=2, label="Placed 2026 candidate (benchmark)")
    ax.fill(ang_c, bench_c, color="#16a34a", alpha=0.12)
    ax.plot(ang_c, you_c, color="#dc2626", linewidth=2.2, label="You")
    ax.fill(ang_c, you_c, color="#dc2626", alpha=0.25)
    ax.set_xticks(angles)
    ax.set_xticklabels([c.replace(" & ", " &\n").replace(" ", "\n", 1) if len(c) > 14 else c for c in cats], fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25", "50", "75", "100"], fontsize=7, color="grey")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), fontsize=8, frameon=False, ncol=1)
    fig.tight_layout()
    return fig


def fig_to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    return buf.getvalue()


# ───────────────────────────── PDF REPORT ────────────────────────────


def latin(s: str) -> str:
    s = s.replace("—", "-").replace("–", "-").replace("’", "'").replace("·", "|").replace("→", "->")
    return s.encode("latin-1", "replace").decode("latin-1")


def build_pdf(name: str, res: dict, radar_png: bytes, workshop_link: str) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_page()
    NL = dict(new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def h(txt, size=13):
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", size)
        pdf.set_text_color(20, 20, 60)
        pdf.cell(0, 8, latin(txt), **NL)
        pdf.set_text_color(0, 0, 0)

    def p(txt, size=10, style=""):
        pdf.set_font("Helvetica", style, size)
        pdf.multi_cell(0, 5.5, latin(txt), **NL)

    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, latin("Placement Resume Gap Report"), **NL)
    p(f"Prepared for {name}  |  Target role: {res['role']}  |  {datetime.now():%d %b %Y}", 9)
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 7, latin(f"Placement readiness: {res['readiness']}/100    Skill credibility: {res['credibility']}/100"), **NL)

    pdf.image(io.BytesIO(radar_png), x=45, w=120)

    h("1. Skill coverage vs a placed 2026 candidate")
    pdf.set_font("Helvetica", "B", 9)
    widths = [62, 22, 26, 80]
    for i, col in enumerate(["Area", "You", "Benchmark", "Detected keywords"]):
        pdf.cell(widths[i], 7, col, border=1, **({"new_x": XPos.RIGHT} if i < 3 else NL))
    pdf.set_font("Helvetica", "", 8.5)
    for c in res["scores"]:
        kws = ", ".join(res["found"][c][:5]) or "-"
        row = [c, str(res["scores"][c]), str(res["bench"][c]), kws[:55]]
        for i, val in enumerate(row):
            pdf.cell(widths[i], 6.5, latin(val), border=1, **({"new_x": XPos.RIGHT} if i < 3 else NL))

    h("2. Skill-credibility flags (what a recruiter will notice)")
    if not res["flags"]:
        p("No credibility issues found. Your claims are backed by foundations and evidence.")
    for f in res["flags"]:
        p(f"[{f['severity']}] {f['title']}", 10, "B")
        p(f["detail"], 9.5)
        p(f"Fix: {f['fix']}", 9.5, "I")
        p(f"Interview question to prepare: \"{f['question']}\"", 9.5)
        pdf.ln(1)

    h("3. Claimed vs proven (ML model check)")
    p(f"Model: {res['model']}. Each project/experience bullet was classified into a skill area.", 8.5, "I")
    p("Listed but not proven: " + (", ".join(res["unproven"]) or "none"))
    p("Hidden strengths (proven but not listed): " + (", ".join(res["hidden"]) or "none"))
    if res["weak_bullets"]:
        p("Weak bullets to rewrite (formula: action verb + tool + measurable result):", 10, "B")
        for wb in res["weak_bullets"]:
            p(f"- {wb['text'][:160]}", 9)
        p("Example: 'Built a RAG chatbot over 120 syllabus PDFs using LangChain and FAISS; 87% answer accuracy on 50 test questions.'", 9, "I")

    h("4. Proof-of-work checklist")
    for k, v in res["evidence"].items():
        p(f"{'[x]' if v else '[ ]'}  {k}", 10)

    t = PROJECT_TRACKS[res["top_gap"]]
    h("5. Your recommended 60-minute project")
    p(t["title"], 11, "B")
    p(t["pitch"])
    p(f"Stack: {t['stack']}", 9.5, "I")

    h("6. Your 7-day fix plan")
    first_fix = res["flags"][0]["fix"] if res["flags"] else "Tighten bullets with measurable outcomes."
    plan = [
        f"Day 1: {first_fix}",
        f"Day 2: Attend 'Build Your First AI Project in 60 Minutes' and build: {t['title']}.",
        "Day 3: Push the project to GitHub with a README (problem, data, approach, result).",
        "Day 4: Deploy it (Streamlit / Hugging Face Spaces) and add the live link.",
        "Day 5: Add one measurable result to every project bullet.",
        "Day 6: Practise answers to the interview questions above, out loud.",
        "Day 7: Re-scan your resume here and compare your new score.",
    ]
    for line in plan:
        p(line)

    h("Register free")
    p(workshop_link, 10, "U")
    pdf.ln(3)
    p("Benchmarks are illustrative composites of placed fresher profiles, not guarantees of placement outcomes.", 7.5, "I")
    return bytes(pdf.output())


# ───────────────────────────── STORAGE ───────────────────────────────

LEAD_FIELDS = [
    "timestamp", "lead_id", "name", "phone", "college", "grad_year", "role",
    "readiness", "credibility", "top_gap", "project_track", "flags",
    "utm_source", "utm_campaign", "referred_by", "referral_code", "consent", "delivery_status",
]


def append_csv(path, fields, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})


def log_event(event: str, **meta):
    append_csv(EVENTS_CSV, ["timestamp", "session", "event", "utm_source", "detail"], dict(
        timestamp=datetime.now().isoformat(timespec="seconds"),
        session=st.session_state.get("sid", ""), event=event,
        utm_source=st.session_state.get("utm_source", "direct"), detail=meta.get("detail", ""),
    ))


def read_csv(path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str) if os.path.exists(path) else pd.DataFrame()


def normalise_phone(raw: str):
    digits = re.sub(r"[^\d]", "", raw)
    if digits.startswith("91") and len(digits) == 12:
        digits = digits[2:]
    if digits.startswith("0") and len(digits) == 11:
        digits = digits[1:]
    return f"+91{digits}" if re.fullmatch(r"[6-9]\d{9}", digits) else None


def referral_code(phone: str) -> str:
    return "NW" + hashlib.sha1(phone.encode()).hexdigest()[:6].upper()


def send_webhook(payload: dict) -> str:
    if not WEBHOOK_URL:
        return "demo_mode"
    try:
        r = requests.post(WEBHOOK_URL, json=payload, timeout=10)
        return "sent" if r.ok else f"error_{r.status_code}"
    except requests.RequestException:
        return "error_network"


# ──────────────────────────── STUDENT VIEW ───────────────────────────


def student_view():
    st.markdown(
        """
        <div style="padding:1.4rem 1.6rem;border-radius:14px;background:linear-gradient(120deg,#0f172a,#1e3a8a);color:white">
        <div style="font-size:0.85rem;opacity:.8;letter-spacing:.06em">FREE · 30 SECONDS · FOR FINAL-YEAR ENGINEERS</div>
        <div style="font-size:2rem;font-weight:800;line-height:1.2;margin:.3rem 0">Would your resume survive a 6-second recruiter scan?</div>
        <div style="opacity:.9">Find the skill claims recruiters won't believe, see how you compare to a placed 2026 candidate,
        and get the one project that closes your biggest gap.</div></div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    c1, c2 = st.columns([2, 1])
    with c1:
        tab_up, tab_paste = st.tabs(["📄 Upload resume", "📋 Paste text"])
        with tab_up:
            up = st.file_uploader("PDF, DOCX or TXT", type=["pdf", "docx", "txt"], label_visibility="collapsed")
        with tab_paste:
            pasted = st.text_area("Paste resume text", height=160, label_visibility="collapsed",
                                  placeholder="Paste your resume text here…")
    with c2:
        role = st.selectbox("Target role", list(BENCHMARKS.keys()))
        st.caption("🔒 Your resume is analysed in memory and never stored.")
        go = st.button("🎯 Scan my resume", type="primary", width="stretch")
        sample = st.button("Try a sample resume", width="stretch")

    if go or sample:
        text = SAMPLE_RESUME if sample else (extract_text(up) if up else pasted)
        if not text or len(text.strip()) < 80:
            st.warning("Please upload a resume or paste at least a few lines of text.")
            return
        st.session_state.res = analyse(text, role, load_model())
        st.session_state.lead_done = False
        log_event("resume_scanned", detail=f"{role}|sample={sample}")

    res = st.session_state.get("res")
    if not res:
        return

    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Placement readiness", f"{res['readiness']}/100")
    m2.metric("Skill credibility", f"{res['credibility']}/100",
              delta=f"{len(res['flags'])} red flags" if res["flags"] else "clean",
              delta_color="inverse" if res["flags"] else "normal")
    m3.metric("Skills detected", res["skills_detected"])
    behind = sum(1 for g in res["gaps"].values() if g > 0)
    m4.metric("Areas behind benchmark", f"{behind}/8")

    left, right = st.columns([1.05, 1])
    with left:
        st.subheader("📡 You vs a placed 2026 candidate")
        fig = radar_chart(res)
        st.pyplot(fig)
        st.session_state.radar_png = fig_to_png(fig)
        plt.close(fig)
    with right:
        st.subheader("🚩 Skill Hallucination Check")
        if not res["flags"]:
            st.success("Your claims are backed by foundations and evidence. Nice.")
        free, locked = res["flags"][:2], res["flags"][2:]
        for f in free:
            icon = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟡"}[f["severity"]]
            with st.container(border=True):
                st.markdown(f"**{icon} {f['title']}**")
                st.write(f["detail"])
        if locked:
            st.info(f"🔒 **+{len(locked)} more issue(s)**, fixes and the interview questions you'll face are in your full report.")
        st.markdown("**Proof-of-work signals**")
        st.write("  ".join(("✅ " if v else "❌ ") + k for k, v in res["evidence"].items()))

    # Claim-vs-proof view powered by the light classifier
    st.subheader("🧠 Claimed vs Proven")
    st.caption(f"A light ML model ({res['model']}) read each project/experience bullet and decided which skill it really demonstrates.")
    rows = []
    for c in res["scores"]:
        listed, proven = c in res["claimed"], c in res["proven"]
        status = ("✅ Proven" if proven else "❌ Listed, no proof") if listed else ("💎 Hidden strength" if proven else "—")
        ev = res["proof"][c][0][0] if res["proof"][c] else ""
        rows.append({"Skill area": c, "Listed": "✅" if listed else "", "Status": status,
                     "Strongest evidence found": (ev[:90] + "…") if len(ev) > 90 else ev})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if res["hidden"]:
        st.success("💎 **Hidden strengths:** your projects show " + ", ".join(res["hidden"])
                   + " but your skills section doesn't say so. Add them.")
    if res["weak_bullets"]:
        wb = res["weak_bullets"][0]
        missing = [n for n, ok in [("an action verb", wb["verb"]), ("a named tool", wb["tool"]), ("a measurable result", wb["number"])] if not ok]
        st.warning(f"✍️ **Weakest bullet:** “{wb['text']}”  \nMissing {', '.join(missing)}. "
                   f"{len(res['weak_bullets']) - 1} more weak bullet(s) with rewrite formula in your full report.")

    # Contextual project CTA
    t = PROJECT_TRACKS[res["top_gap"]]
    utm = f"utm_source=gap_analyzer&utm_medium={quote(st.session_state.get('utm_source', 'direct'))}&utm_content={quote(res['top_gap'])}"
    workshop_link = f"{WORKSHOP_URL}{'&' if '?' in WORKSHOP_URL else '?'}{utm}"
    st.markdown(
        f"""
        <div style="margin-top:1rem;padding:1.3rem 1.5rem;border-radius:14px;border:2px solid #f59e0b;background:#fffbeb;color:#1f2937">
        <div style="font-size:.8rem;font-weight:700;color:#b45309;letter-spacing:.05em">YOUR BIGGEST GAP: {res['top_gap'].upper()}</div>
        <div style="font-size:1.4rem;font-weight:800;margin:.3rem 0">Close it in 60 minutes: build a {t['title']}</div>
        <div>{t['pitch']}</div>
        <div style="font-size:.85rem;margin-top:.4rem;opacity:.75">Stack: {t['stack']} · Free live workshop · Certificate + GitHub-ready project</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    st.link_button("🚀 Reserve my free seat — Build Your First AI Project in 60 Minutes", workshop_link,
                   type="primary", width="stretch")

    # Gated report + lead capture
    st.divider()
    st.subheader("📲 Get your full report on WhatsApp")
    st.caption("Includes every red flag with fixes, the interview questions each claim will trigger, a proof-of-work checklist and a 7-day fix plan.")

    if not st.session_state.get("lead_done"):
        with st.form("lead"):
            a, b = st.columns(2)
            name = a.text_input("Name*")
            phone = b.text_input("WhatsApp number*", placeholder="98765 43210")
            college = a.text_input("College*")
            year = b.selectbox("Graduating in", ["2026", "2027", "2025", "Other"])
            consent = st.checkbox("I agree to receive my report and workshop updates from NxtWave on WhatsApp. I can reply STOP anytime.")
            submitted = st.form_submit_button("Send my report", type="primary", width="stretch")

        if submitted:
            ph = normalise_phone(phone)
            if not (name.strip() and college.strip()):
                st.error("Please fill in your name and college.")
            elif not ph:
                st.error("Please enter a valid 10-digit Indian mobile number.")
            elif not consent:
                st.error("Please tick the consent box so we can send the report on WhatsApp.")
            else:
                capture_lead(name.strip(), ph, college.strip(), year, res, workshop_link)

    if st.session_state.get("lead_done"):
        post_capture(res)


def capture_lead(name, phone, college, year, res, workshop_link):
    leads = read_csv(LEADS_CSV)
    existing = not leads.empty and phone in set(leads["phone"])
    code = referral_code(phone)
    lead_id = code + datetime.now().strftime("%H%M%S")
    pdf_bytes = build_pdf(name, res, st.session_state.radar_png, workshop_link)
    t = PROJECT_TRACKS[res["top_gap"]]

    payload = dict(
        event="gap_report_requested", lead_id=lead_id, name=name, phone=phone, college=college,
        grad_year=year, role=res["role"], readiness=res["readiness"], credibility=res["credibility"],
        top_gap=res["top_gap"], project_track=t["title"], flags=[f["title"] for f in res["flags"]],
        workshop_link=workshop_link, referral_link=f"{APP_URL}?ref={code}&utm_source=referral",
        utm_source=st.session_state.get("utm_source"), referred_by=st.session_state.get("ref"),
        report_filename=f"Placement_Gap_Report_{name.split()[0]}.pdf",
        report_pdf_base64=base64.b64encode(pdf_bytes).decode(),
        whatsapp_template=dict(name="gap_report_v1", vars=[name.split()[0], str(res["readiness"]), t["title"]]),
    )
    status = send_webhook(payload)

    if not existing:
        append_csv(LEADS_CSV, LEAD_FIELDS, dict(
            timestamp=datetime.now().isoformat(timespec="seconds"), lead_id=lead_id, name=name, phone=phone,
            college=college, grad_year=year, role=res["role"], readiness=res["readiness"],
            credibility=res["credibility"], top_gap=res["top_gap"], project_track=t["title"],
            flags="|".join(f["id"] for f in res["flags"]),
            utm_source=st.session_state.get("utm_source", "direct"),
            utm_campaign=st.session_state.get("utm_campaign", ""),
            referred_by=st.session_state.get("ref", ""), referral_code=code, consent="yes",
            delivery_status=status,
        ))
        log_event("lead_captured", detail=res["top_gap"])

    st.session_state.update(lead_done=True, pdf=pdf_bytes, code=code, lead_id=lead_id,
                            first_name=name.split()[0], status=status)
    st.rerun()


def post_capture(res):
    status = st.session_state.status
    if status == "sent":
        st.success(f"✅ Done, {st.session_state.first_name}! Your report is on its way to WhatsApp.")
    else:
        st.success(f"✅ Report ready, {st.session_state.first_name}!")
        wa = f"https://wa.me/{WA_BUSINESS_NUMBER}?text=" + quote(f"Hi NxtWave! Please send my Placement Gap Report. ID: {st.session_state.lead_id}")
        st.link_button("💬 Get it on WhatsApp", wa, width="stretch")

    st.download_button("⬇️ Download PDF now", st.session_state.pdf, file_name="Placement_Gap_Report.pdf",
                       mime="application/pdf", width="stretch")

    code = st.session_state.code
    ref_link = f"{APP_URL}?ref={code}&utm_source=referral"
    share = quote(f"I scored {res['readiness']}/100 on placement readiness 😬 This tool shows which resume skills recruiters won't believe. Check yours in 30 sec: {ref_link}")
    with st.container(border=True):
        st.markdown("### 🎁 Your batch is applying for the same jobs")
        st.write("Share your link. When **3 friends** scan their resumes, you unlock a **1:1 resume review slot** at the workshop.")
        st.code(ref_link, language=None)
        leads = read_csv(LEADS_CSV)
        count = 0 if leads.empty else int((leads["referred_by"] == code).sum())
        st.progress(min(count, 3) / 3, text=f"{count}/3 friends joined")
        st.link_button("📤 Share on WhatsApp", f"https://wa.me/?text={share}", width="stretch")


# ─────────────────────────── GROWTH DASHBOARD ────────────────────────


def dashboard():
    st.title("📈 Campaign Growth Dashboard")
    if st.text_input("Admin password", type="password") != ADMIN_PASSWORD:
        st.info("Enter the admin password to view campaign data.")
        return

    leads, events = read_csv(LEADS_CSV), read_csv(EVENTS_CSV)
    views = 0 if events.empty else int((events["event"] == "page_view").sum())
    scans = 0 if events.empty else int((events["event"] == "resume_scanned").sum())
    n = len(leads)

    a, b, c, d = st.columns(4)
    a.metric("Visitors", views)
    b.metric("Resumes scanned", scans, f"{scans / views:.0%} of visitors" if views else None)
    c.metric("Leads captured", n, f"{n / scans:.0%} of scans" if scans else None)
    d.metric("Goal", f"{n}/{TARGET_REGISTRATIONS}")
    st.progress(min(n / TARGET_REGISTRATIONS, 1.0), text=f"{n / TARGET_REGISTRATIONS:.0%} to 500 registrations")

    if leads.empty:
        st.caption("No leads yet. Scan a sample resume in the student view to see data flow in.")
        return

    leads["timestamp"] = pd.to_datetime(leads["timestamp"])
    x, y = st.columns(2)
    with x:
        st.markdown("**Leads by channel (utm_source)**")
        st.bar_chart(leads["utm_source"].value_counts())
        st.markdown("**Top colleges** (→ pick campus ambassadors)")
        st.bar_chart(leads["college"].str.title().value_counts().head(10))
    with y:
        st.markdown("**Biggest skill gap across leads** (→ workshop content focus)")
        st.bar_chart(leads["top_gap"].value_counts())
        st.markdown("**Leads per day**")
        st.line_chart(leads.set_index("timestamp").resample("D").size())

    st.markdown("**🏆 Referral leaderboard**")
    refs = leads[leads["referred_by"].fillna("") != ""]
    if refs.empty:
        st.caption("No referrals yet.")
    else:
        board = refs["referred_by"].value_counts().rename_axis("referral_code").reset_index(name="referrals")
        names = leads.set_index("referral_code")["name"].to_dict()
        board.insert(0, "student", board["referral_code"].map(names))
        st.dataframe(board, hide_index=True, width="stretch")

    st.markdown("**All leads**")
    st.dataframe(leads.drop(columns=["phone"]), hide_index=True, width="stretch")
    st.download_button("⬇️ Export leads CSV (for WhatsApp broadcast)", leads.to_csv(index=False), "leads.csv")


# ─────────────────────────────── MAIN ────────────────────────────────


def main():

    if "sid" not in st.session_state:
        qp = st.query_params
        st.session_state.sid = hashlib.md5(str(datetime.now().timestamp()).encode()).hexdigest()[:10]
        st.session_state.utm_source = qp.get("utm_source", "direct")
        st.session_state.utm_campaign = qp.get("utm_campaign", "")
        st.session_state.ref = qp.get("ref", "")
        log_event("page_view")

    page = st.sidebar.radio("View", ["🎯 Student view", "📈 Growth dashboard"])
    st.sidebar.caption("NxtWave · Build Your First AI Project in 60 Minutes")
    student_view() if page.startswith("🎯") else dashboard()


main()