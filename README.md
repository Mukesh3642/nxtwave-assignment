# 🎯 Placement Resume Gap Analyzer
Lead magnet for NxtWave's free workshop **"Build Your First AI Project in 60 Minutes"** — target: 500 final-year registrations in 7 days on ₹2,000.

## Funnel
1. **Scan (free value)** – upload PDF/DOCX or paste text, pick a target role.
2. **Skill Hallucination Check** – ML claim-vs-proof check + 7 rules flag claims recruiters won't believe (advanced AI without Python, models without data work, buzzwords without proof links, DL without ML basics, NLP without text basics, nothing shipped, no numbers).
3. **Peer Benchmark Radar** – matplotlib spider chart: you vs an illustrative placed 2026 candidate, per role.
4. **Contextual Project CTA** – biggest gap → the 60-minute project that fixes it (e.g. weak NLP → Customer-Support Ticket Classifier). UTM-tagged registration link.
5. **WhatsApp report gate** – name, number (+91 validated), college, explicit consent → PDF built → webhook → WhatsApp API. Free view shows top 2 flags; the PDF adds all flags, fixes, interview questions, proof-of-work checklist and a 7-day plan.
6. **Referral loop** – unique code + WhatsApp share text; 3 friends = 1:1 resume review.
7. **Growth dashboard** (sidebar, password) – visitors → scans → leads funnel, progress to 500, leads by channel, top colleges (ambassador targeting), most common gap (workshop content), referral leaderboard, CSV export.

## 🧠 Light models (CPU only, no paid API)
| Model | Size | What it does |
|---|---|---|
| **TF-IDF (word + char n-grams) + Logistic Regression** (default) | ~1 MB, trains in <1 s at startup | Classifies every project/experience bullet into 1 of 8 skill areas, even when no keyword is used ("built a chatbot that answers from PDFs" → GenAI) |
| **bge-small-en-v1.5 (ONNX via fastembed) + LogReg** (optional) | 33M params / ~67 MB | Same classifier on semantic embeddings for better paraphrase understanding. Auto-falls back to TF-IDF if unavailable |
| Claim-vs-Proof detector | rules on model output | Listed-but-never-demonstrated skills = skill hallucination; demonstrated-but-unlisted = hidden strength |
| Bullet strength scorer | transparent rules | Action verb + named tool + measurable result |

Training data = ~200 hand-written resume bullets in `engine.py` (`SEED`). Add more to improve accuracy.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Click **Try a sample resume** for an instant demo. Dashboard password defaults to `nxtwave500`.

## Deploy (live link in ~5 min)
Push to GitHub → share.streamlit.io → New app → `app.py` → paste `.streamlit/secrets.toml.example` values into **Secrets**.

## WhatsApp delivery (n8n / Zapier)
`WEBHOOK_URL` receives JSON: lead fields, scores, flags, `report_pdf_base64`, `whatsapp_template` vars.
Flow: **Webhook → Move Binary Data (base64→PDF) → WATI/Interakt/Twilio send-document (template `gap_report_v1`) → Google Sheets row**.
Without a webhook the app runs in demo mode: PDF download + click-to-chat `wa.me` button (student messages first, which opens a free 24-hour service window).

## Tracking links per channel
`?utm_source=whatsapp_groups`, `?utm_source=linkedin`, `?utm_source=club_<name>`, referrals add `?ref=CODE`.

> Note: benchmarks are illustrative composites; storage is CSV (ephemeral on Streamlit Cloud) — swap for Google Sheets/Supabase in production.
