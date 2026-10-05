"""
engine.py — light-model resume analysis (CPU only, no GPU, no paid API).

Models
------
1. Skill classifier: classifies every resume line/bullet into one of 8 skill areas (or "Other").
   - Default : TF-IDF (word 1-2 grams + char 3-5 grams) + Logistic Regression.
               Trains in < 1 s at startup on the seed set below. ~1 MB RAM. Zero downloads.
   - Upgrade : BAAI/bge-small-en-v1.5 (33M params, ONNX via fastembed, ~67 MB) + Logistic Regression.
               Used automatically if `fastembed` is installed and the model can download;
               otherwise falls back to TF-IDF silently.
2. Claim-vs-Proof detector: skills *listed* (skills/summary lines) vs skills *demonstrated*
   (project / experience bullets the classifier assigns to that area). Listed-but-never-shown =
   "skill hallucination". Shown-but-never-listed = "hidden strength".
3. Bullet strength scorer (rule-based, transparent): action verb + tool + measurable result.
"""

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion

# ───────────────────────────── TAXONOMY ──────────────────────────────
SKILLS = {
    "Programming Foundations": [
        "python", "java", "c++", "sql", "git", "github", "data structures",
        "algorithms", "oop", "object oriented", "dsa", "javascript",
    ],
    "Data Handling": [
        "pandas", "numpy", "data preprocessing", "preprocessing", "pre-processing",
        "data cleaning", "feature engineering", "eda", "exploratory data analysis",
        "matplotlib", "seaborn", "data visualization", "power bi", "tableau", "excel",
    ],
    "Core ML": [
        "machine learning", "scikit-learn", "sklearn", "regression", "classification",
        "clustering", "random forest", "xgboost", "decision tree", "svm",
        "cross-validation", "hyperparameter", "model evaluation",
    ],
    "Deep Learning": [
        "deep learning", "neural network", "neural networks", "cnn", "rnn", "lstm",
        "pytorch", "tensorflow", "keras", "transformer", "transformers", "attention", "bert",
    ],
    "NLP": [
        "nlp", "natural language processing", "text classification", "sentiment analysis",
        "tokenization", "bert", "spacy", "nltk", "hugging face", "huggingface",
        "tf-idf", "word embeddings", "named entity recognition", "ner",
    ],
    "Computer Vision": [
        "computer vision", "opencv", "image classification", "object detection",
        "yolo", "segmentation", "image processing", "resnet",
    ],
    "GenAI & LLMs": [
        "llm", "llms", "large language model", "generative ai", "genai",
        "prompt engineering", "rag", "retrieval augmented generation", "langchain",
        "llamaindex", "openai", "gpt", "fine-tuning", "fine tuning", "vector database",
        "embeddings", "ai agents", "agentic", "hallucination detection", "lora",
    ],
    "Deployment & MLOps": [
        "flask", "fastapi", "streamlit", "docker", "aws", "gcp", "azure", "rest api",
        "api", "mlops", "deployed", "deployment", "ci/cd", "hugging face spaces",
        "render", "vercel", "kubernetes",
    ],
}
CATEGORIES = list(SKILLS.keys())
ADVANCED_CATS = {"Deep Learning", "NLP", "Computer Vision", "GenAI & LLMs"}
SKILL_TARGET = 4

BENCHMARKS = {  # illustrative placed-2026-fresher composites, 0-100, order = CATEGORIES
    "AI / ML Engineer": [85, 75, 70, 60, 55, 45, 65, 60],
    "Data Analyst / Data Scientist": [80, 90, 70, 35, 35, 25, 40, 40],
    "Full-Stack Developer (AI-enabled)": [90, 50, 35, 25, 25, 20, 55, 85],
}

ADVANCED_TERMS = [
    "transformer", "transformers", "rag", "retrieval augmented generation", "fine-tuning",
    "fine tuning", "llm", "large language model", "hallucination detection", "ai agents",
    "agentic", "lora", "diffusion", "gan", "reinforcement learning", "attention", "vector database",
]

# ───────────────────── SEED TRAINING DATA (bullets) ──────────────────
# Written as realistic resume bullets so the classifier learns *descriptions of work*,
# not just keywords. Extend this list to improve accuracy.
SEED = {
    "Programming Foundations": [
        "Implemented a library management system in Java using object oriented design",
        "Solved 400+ data structures and algorithms problems on LeetCode",
        "Wrote Python scripts to automate file renaming and report generation",
        "Designed normalized SQL schemas and wrote complex joins for a college ERP",
        "Built a command line expense tracker in Python with unit tests",
        "Managed version control with Git branches and pull requests for a team of four",
        "Developed a C++ program for graph shortest path using Dijkstra's algorithm",
        "Wrote recursive and dynamic programming solutions for competitive programming contests",
        "Built a REST client in JavaScript to consume a weather service",
        "Refactored legacy Python code into reusable modules and classes",
        "Wrote efficient queries to aggregate student attendance from a MySQL database",
        "Implemented linked lists, stacks, queues and hash maps from scratch",
    ],
    "Data Handling": [
        "Cleaned a 50,000 row dataset by handling missing values and outliers with pandas",
        "Performed exploratory data analysis and visualised trends with matplotlib and seaborn",
        "Engineered features from timestamps and categorical columns to improve model input",
        "Scraped product data and merged multiple CSV files into a single clean dataset",
        "Built an interactive Power BI dashboard tracking monthly sales by region",
        "Normalised and encoded features using NumPy and one-hot encoding",
        "Removed duplicates and fixed inconsistent date formats across data sources",
        "Analysed placement data to find which factors correlate with getting hired",
        "Created data pipelines to collect and preprocess sensor readings",
        "Split the dataset into train, validation and test sets with stratification to avoid leakage",
        "Visualised class imbalance and applied SMOTE oversampling",
        "Prepared and labelled a custom dataset of 3,000 samples",
    ],
    "Core ML": [
        "Trained a random forest classifier to predict student placement with 87% accuracy",
        "Built a house price prediction model using linear regression and evaluated with RMSE",
        "Tuned hyperparameters with grid search and 5-fold cross-validation",
        "Compared logistic regression, SVM and XGBoost for credit default prediction",
        "Applied K-means clustering to segment customers by purchase behaviour",
        "Reduced overfitting using regularisation and feature selection",
        "Evaluated models with precision, recall, F1 score and ROC-AUC",
        "Built a churn prediction model with scikit-learn pipelines",
        "Used decision trees to explain which features drive loan approval",
        "Predicted crop yield from weather data using gradient boosting",
        "Handled class imbalance with class weights and threshold tuning",
        "Explained model predictions with SHAP feature importance",
    ],
    "Deep Learning": [
        "Designed and trained a convolutional neural network in PyTorch",
        "Built an LSTM network to forecast stock prices from time series data",
        "Implemented backpropagation and gradient descent for a neural network from scratch",
        "Trained a deep neural network in TensorFlow Keras with dropout and batch normalisation",
        "Fine-tuned a pretrained ResNet with transfer learning on a small dataset",
        "Implemented a transformer encoder with multi-head self attention",
        "Used learning rate scheduling and early stopping to stabilise training",
        "Trained an autoencoder for anomaly detection on network traffic",
        "Built a GAN to generate synthetic handwritten digits",
        "Optimised model training on GPU and reduced epoch time by 40%",
        "Experimented with activation functions and weight initialisation",
        "Trained a recurrent neural network for music generation",
    ],
    "NLP": [
        "Built a sentiment analysis model for product reviews using BERT",
        "Classified customer support tickets into categories using TF-IDF and logistic regression",
        "Performed tokenization, stop word removal and lemmatization with NLTK and spaCy",
        "Detected fake reviews in English and Telugu using multilingual transformers",
        "Built a named entity recognition model to extract names and dates from documents",
        "Created a spam email classifier using natural language processing",
        "Fine-tuned a Hugging Face model for text classification",
        "Summarised news articles automatically with an abstractive summarisation model",
        "Built a chatbot intent classifier for college admission queries",
        "Trained word embeddings to capture similarity between technical terms",
        "Detected toxic comments on social media posts",
        "Translated and transliterated text between Indian languages",
    ],
    "Computer Vision": [
        "Detected plant leaf diseases from smartphone photos using image classification",
        "Built a real-time face mask detector with OpenCV and a webcam",
        "Trained YOLO to detect vehicles and count traffic in CCTV footage",
        "Segmented brain tumours in MRI scans using U-Net",
        "Recognised handwritten digits from images",
        "Built an attendance system using face recognition",
        "Applied image augmentation such as rotation, flipping and cropping",
        "Detected cracks on bridges and buildings from drone images",
        "Tracked moving objects across video frames",
        "Extracted text from scanned documents using OCR",
        "Classified X-ray images to detect pneumonia",
        "Estimated human pose from video for fitness feedback",
    ],
    "GenAI & LLMs": [
        "Built a chatbot that answers questions from PDF documents using retrieval augmented generation",
        "Created a RAG pipeline with LangChain, embeddings and a vector database",
        "Engineered prompts for GPT to generate structured JSON from emails",
        "Fine-tuned an open source LLM with LoRA on domain specific data",
        "Built an AI agent that plans tasks and calls external tools",
        "Detected hallucinations in LLM answers by checking them against source documents",
        "Integrated the OpenAI API to auto-generate interview questions",
        "Indexed lecture notes into a vector store for semantic search",
        "Built a multi-agent workflow to research and summarise topics",
        "Evaluated LLM responses for factuality and relevance",
        "Generated images from text prompts with a diffusion model",
        "Built a resume reviewer powered by a large language model",
    ],
    "Deployment & MLOps": [
        "Deployed a machine learning model as a REST API with FastAPI",
        "Containerised the application with Docker and deployed it on AWS EC2",
        "Built and hosted an interactive Streamlit web app used by 200 students",
        "Set up CI/CD with GitHub Actions to run tests and deploy automatically",
        "Hosted the model on Hugging Face Spaces with a public demo link",
        "Served predictions through a Flask backend connected to a React frontend",
        "Monitored model performance and logged predictions in production",
        "Deployed the web application on Vercel and Render",
        "Reduced API latency from 800 ms to 120 ms with caching",
        "Orchestrated containers with Kubernetes",
        "Tracked experiments and model versions with MLflow",
        "Built a mobile app that calls the model through a cloud API",
    ],
    "Other": [
        "Member of the college cultural committee and organised the annual fest",
        "Strong communication and teamwork skills",
        "Volunteered at an NGO teaching children on weekends",
        "Captain of the college cricket team",
        "Fluent in English, Hindi and Telugu",
        "Hobbies include reading, travelling and music",
        "Class representative for two years",
        "Bachelor of Technology in Computer Science, CGPA 8.4",
        "Secured first prize in an inter-college debate competition",
        "Responsible, hardworking and quick learner",
        "Coordinated logistics for a 300-person technical symposium",
        "Completed Class XII with 92 percent",
    ],
}
for _c, _kws in SKILLS.items():  # keywords as short extra samples
    SEED[_c] = SEED[_c] + _kws


# ───────────────────────────── MODEL ─────────────────────────────────


class SkillClassifier:
    """Light skill-area classifier. bge-small ONNX if available, else TF-IDF; LogReg on top."""

    def __init__(self, prefer_embeddings: bool = False):
        texts, labels = [], []
        for cat, items in SEED.items():
            texts += items
            labels += [cat] * len(items)
        self.backend = "tfidf"
        self.embedder = None
        if prefer_embeddings:
            try:
                from fastembed import TextEmbedding

                self.embedder = TextEmbedding("BAAI/bge-small-en-v1.5")
                self.backend = "bge-small-en-v1.5 (ONNX)"
            except Exception:
                self.embedder = None
        if self.embedder is None:
            self.vec = FeatureUnion([
                ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
            ])
            X = self.vec.fit_transform([t.lower() for t in texts])
            self.backend_label = "TF-IDF + Logistic Regression"
        else:
            X = self._embed(texts)
            self.backend_label = "bge-small embeddings + Logistic Regression"
        self.clf = LogisticRegression(C=12, max_iter=3000, class_weight="balanced")
        self.clf.fit(X, labels)
        self.classes = list(self.clf.classes_)

    def _embed(self, texts):
        return np.array(list(self.embedder.embed(list(texts))))

    def predict(self, texts):
        """Return list of (category, confidence) for each text."""
        if not texts:
            return []
        X = self._embed(texts) if self.embedder is not None else self.vec.transform([t.lower() for t in texts])
        proba = self.clf.predict_proba(X)
        idx = proba.argmax(axis=1)
        return [(self.classes[i], float(proba[r, i])) for r, i in enumerate(idx)]


# ─────────────────────────── PARSING ─────────────────────────────────
SECTION_PATTERNS = {
    "claims": r"(technical\s+)?skills|summary|objective|profile|about me|certifications?|tools|technologies|core competencies|courses",
    "evidence": r"projects?|experience|work experience|internships?|publications?|research|achievements|hackathons?|open source",
    "other": r"education|academics|extra[- ]?curricular|activities|interests|hobbies|languages|personal details|declaration",
}
ACTION_VERBS = (
    "built|developed|designed|implemented|trained|deployed|created|engineered|automated|optimised|optimized|"
    "reduced|improved|increased|led|analysed|analyzed|integrated|fine-tuned|launched|shipped|scaled|architected|"
    "published|evaluated|benchmarked|wrote|migrated|achieved|delivered"
)


def has(text: str, kw: str) -> bool:
    return re.search(r"(?<![a-z0-9+#])" + re.escape(kw) + r"(?![a-z0-9+#])", text) is not None


def parse_sections(raw: str):
    """Split resume into lines tagged claims / evidence / other / unknown."""
    lines = []
    current = "unknown"
    for line in raw.splitlines():
        line = line.strip().lstrip("•●▪◦-*·–").strip()
        if not line:
            continue
        header = re.match(r"^([A-Za-z &/]{3,40})\s*[:|\-–]?\s*(.*)$", line)
        if header:
            head = header.group(1).strip().lower()
            for sec, pat in SECTION_PATTERNS.items():
                if re.fullmatch(pat, head):
                    current = sec
                    line = header.group(2).strip()  # "SKILLS: python, sql" keeps the rest
                    break
        if not line:
            continue
        for part in re.split(r"(?<=[.;])\s+(?=[A-Z])", line):  # split long lines into sentences
            if len(part.split()) >= 2:
                lines.append((current, part))
    return lines


def is_skill_list(text: str) -> bool:
    return text.count(",") >= 3 or text.count("|") >= 3


def bullet_strength(text: str) -> dict:
    t = text.lower()
    verb = bool(re.match(rf"^({ACTION_VERBS})\b", t)) or bool(re.search(rf"\b({ACTION_VERBS})\b", t))
    tool = any(has(t, k) for kws in SKILLS.values() for k in kws)
    number = bool(re.search(r"\d+(\.\d+)?\s?(%|x\b|ms|k\b|\+)|\b\d{2,}\b|accuracy|f1|latency", t))
    score = verb + tool + number
    return dict(text=text, verb=verb, tool=tool, number=number, score=score)


# ─────────────────────────── ANALYSIS ────────────────────────────────


def analyse(raw: str, role: str, model: SkillClassifier) -> dict:
    text = raw.lower()
    bench = dict(zip(CATEGORIES, BENCHMARKS[role]))
    lines = parse_sections(raw)
    has_sections = any(sec != "unknown" for sec, _ in lines)

    # Claims = skills/summary lines + any comma-separated skill list. Evidence = project/experience bullets.
    claim_lines, evidence_lines = [], []
    for sec, ln in lines:
        if sec == "claims" or is_skill_list(ln):
            claim_lines.append(ln)
        elif sec == "evidence" or (not has_sections and sec == "unknown"):
            evidence_lines.append(ln)
    claim_text = " ".join(claim_lines).lower() or text

    # 1) keyword layer (whole resume)
    found = {c: [k for k in SKILLS[c] if has(text, k)] for c in CATEGORIES}
    claimed = {c for c in CATEGORIES if any(has(claim_text, k) for k in SKILLS[c])}

    # 2) model layer: classify every evidence bullet
    preds = model.predict(evidence_lines)
    proof = {c: [] for c in CATEGORIES}
    tech_lines = []
    for ln, (cat, conf) in zip(evidence_lines, preds):
        if cat != "Other":
            tech_lines.append(ln)
        # Proof needs an action ("built", "trained"...). "Explored techniques" is interest, not evidence.
        if not re.search(rf"\b({ACTION_VERBS})\b", ln.lower()):
            continue
        if cat != "Other" and conf >= 0.30:
            proof[cat].append((ln, conf))
        # a bullet that explicitly names a category keyword also counts as proof
        for c in CATEGORIES:
            if c != cat and any(has(ln.lower(), k) for k in SKILLS[c]) and (ln, conf) not in proof[c]:
                proof[c].append((ln, 0.5))
    proven = {c for c in CATEGORIES if proof[c]}

    # 3) blended score: listing a skill gets you to max 60, demonstrating it gets the rest
    scores = {}
    for c in CATEGORIES:
        kw = min(1.0, len(found[c]) / SKILL_TARGET)
        pr = min(1.0, len(proof[c]) / 2)
        scores[c] = round(100 * (0.6 * kw + 0.4 * pr)) if (kw or pr) else 0
        if pr and not kw:  # demonstrated without naming tools
            scores[c] = max(scores[c], 40)

    unproven = sorted(c for c in claimed - proven)
    hidden = sorted(c for c in proven - claimed if c != "Other")

    evidence = {
        "GitHub / portfolio link": bool(re.search(r"github\.com/|gitlab\.com/|portfolio", text)),
        "Live / deployed link": bool(re.search(r"https?://|\.app\b|\.vercel|huggingface\.co/spaces|deployed", text)),
        "Quantified results": bool(re.search(r"\d+(\.\d+)?\s?%|accuracy|f1|precision|recall|latency|\d+x\b", text)),
        "Projects section": bool(evidence_lines),
        "Internship / hackathon / publication": bool(re.search(r"intern|hackathon|ieee|springer|publication|published", text)),
    }
    evidence_score = round(sum(evidence.values()) / len(evidence) * 100)
    advanced_found = [t for t in ADVANCED_TERMS if has(text, t)]

    bullets = [bullet_strength(ln) for ln in tech_lines if not is_skill_list(ln) and len(ln.split()) >= 4]
    weak_bullets = sorted([b for b in bullets if b["score"] <= 1], key=lambda b: b["score"])[:5]

    flags = hallucination_flags(text, scores, evidence, advanced_found, unproven, proof)
    penalty = sum({"HIGH": 8, "MEDIUM": 4, "LOW": 2}[f["severity"]] for f in flags)
    credibility = max(10, 100 - sum({"HIGH": 25, "MEDIUM": 12, "LOW": 6}[f["severity"]] for f in flags))

    weights = np.array(BENCHMARKS[role], dtype=float)
    coverage = float(np.dot([min(scores[c], bench[c]) / bench[c] for c in CATEGORIES], weights) / weights.sum() * 100)
    readiness = int(max(5, min(100, round(0.65 * coverage + 0.35 * evidence_score - penalty))))

    gaps = {c: bench[c] - scores[c] for c in CATEGORIES}
    ids = {f["id"] for f in flags}
    if "no_python" in ids:
        top_gap = "Programming Foundations"
    elif "no_data" in ids:
        top_gap = "Data Handling"
    elif unproven and any(c in ADVANCED_CATS for c in unproven):
        top_gap = max((c for c in unproven if c in ADVANCED_CATS), key=lambda c: bench[c])
    else:
        top_gap = max(gaps, key=lambda c: gaps[c] * bench[c])

    return dict(
        role=role, scores=scores, bench=bench, found=found, evidence=evidence,
        evidence_score=evidence_score, flags=flags, credibility=credibility, readiness=readiness,
        top_gap=top_gap, gaps=gaps, advanced=advanced_found,
        skills_detected=sum(len(v) for v in found.values()),
        claimed=sorted(claimed), proven=sorted(proven), unproven=unproven, hidden=hidden,
        proof={c: [(l, round(p, 2)) for l, p in v[:2]] for c, v in proof.items()},
        weak_bullets=weak_bullets, n_bullets=len(bullets), model=model.backend_label,
    )


def hallucination_flags(text, s, evidence, advanced, unproven, proof) -> list:
    flags = []
    advanced_claimed = max(s[c] for c in ADVANCED_CATS) >= 50

    if advanced_claimed and not has(text, "python"):
        flags.append(dict(
            id="no_python", severity="HIGH", title="Advanced AI claimed, but Python is missing",
            detail="Your resume leans on advanced AI concepts but never mentions Python, the language almost all of it is built in. Recruiters read this as copied keywords.",
            fix="List Python explicitly and link one project where you wrote the Python yourself.",
            question="Walk me through the Python code of your best AI project."))
    if (s["Deep Learning"] >= 50 or s["GenAI & LLMs"] >= 50) and s["Data Handling"] < 25:
        flags.append(dict(
            id="no_data", severity="HIGH", title="Models without data foundations",
            detail="Deep learning / LLM work is claimed, but there is no sign of pandas, preprocessing or data cleaning. Real ML is mostly data work.",
            fix="Add a project bullet describing how you collected, cleaned and split your data.",
            question="How did you clean and split the dataset, and how did you avoid data leakage?"))
    adv_unproven = [c for c in unproven if c in ADVANCED_CATS]
    if adv_unproven:
        flags.append(dict(
            id="unproven", severity="HIGH" if len(adv_unproven) >= 2 else "MEDIUM",
            title="Listed but never demonstrated: " + ", ".join(adv_unproven),
            detail="Our model read every project and experience bullet and found no work that actually uses these skills. A claim with no project behind it is the first thing interviewers attack.",
            fix=f"Add one project bullet that shows {adv_unproven[0]} in action (what you built, with which tool, and the result), or drop the claim.",
            question=f"Tell me about a project where you applied {adv_unproven[0]}."))
    if len(advanced) >= 3 and not (evidence["GitHub / portfolio link"] or evidence["Live / deployed link"]):
        flags.append(dict(
            id="buzzwords", severity="MEDIUM", title=f"{len(advanced)} advanced buzzwords, zero proof links",
            detail="Terms like " + ", ".join(advanced[:4]) + " appear with no GitHub or live link. Unverifiable claims are discounted heavily in screening.",
            fix="Add a GitHub link to every project. One working repo beats five buzzwords.",
            question="Can you share the repository for this project?"))
    if s["Deep Learning"] >= 50 and s["Core ML"] < 25:
        flags.append(dict(
            id="skipped_ml", severity="MEDIUM", title="Skipped ML fundamentals",
            detail="Deep learning without classical ML (regression, evaluation, cross-validation) suggests gaps interviewers love to probe.",
            fix="Show one classical ML project with a proper evaluation metric.",
            question="Explain the bias-variance trade-off with an example from your work."))
    if s["NLP"] >= 50 and not re.search(r"tokeni[sz]|nltk|spacy|tf-idf|embedding|preprocess|lemmati|stop ?words|text cleaning", text):
        flags.append(dict(
            id="nlp_no_basics", severity="MEDIUM", title="NLP claimed without text-processing basics",
            detail="NLP is listed but tokenisation, text preprocessing or embeddings are not.",
            fix="Mention how you prepared text data (tokenisation, cleaning, embeddings).",
            question="How did you preprocess the text before training?"))
    if advanced_claimed and s["Deployment & MLOps"] == 0:
        flags.append(dict(
            id="nothing_shipped", severity="LOW", title="Nothing shipped",
            detail="No API, app or deployment is mentioned, so recruiters can't see your work running.",
            fix="Deploy one project (Streamlit / Hugging Face Spaces) and add the live link.",
            question="Has anyone outside your team used what you built?"))
    if not evidence["Quantified results"]:
        flags.append(dict(
            id="no_numbers", severity="LOW", title="No measurable results",
            detail="No accuracy, %, latency or impact numbers. Bullets without numbers read as tasks, not achievements.",
            fix="Rewrite each project bullet as: action + tool + measurable result.",
            question="How did you measure whether your model was good?"))
    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    return sorted(flags, key=lambda f: order[f["severity"]])


# ─────────────────────────── PROJECT TRACKS ──────────────────────────
PROJECT_TRACKS = {
    "Programming Foundations": dict(title="AI Resume Screener in pure Python",
        pitch="Write the Python end-to-end (reading files, cleaning text, scoring) so 'Python' on your resume is backed by code you wrote.",
        stack="Python · regex · scikit-learn"),
    "Data Handling": dict(title="Placement Predictor with a real data pipeline",
        pitch="Load, clean and engineer features on a messy dataset before a single model is trained, exactly what interviewers probe.",
        stack="pandas · NumPy · scikit-learn"),
    "Core ML": dict(title="Placement Predictor (Classification)",
        pitch="Train, cross-validate and explain a classifier so you can answer overfitting and metric questions with your own numbers.",
        stack="scikit-learn · pandas"),
    "Deep Learning": dict(title="Handwritten Digit Recogniser (CNN)",
        pitch="Build and train your first neural network and see what each layer actually learns.",
        stack="PyTorch / Keras"),
    "NLP": dict(title="Smart Customer-Support Ticket Classifier",
        pitch="Turn raw support messages into routed categories: tokenisation, embeddings and a deployable classifier in one hour.",
        stack="Python · Hugging Face · scikit-learn"),
    "Computer Vision": dict(title="Crop Leaf Disease Detector",
        pitch="Use transfer learning to classify images from your phone camera, a visual project recruiters remember.",
        stack="PyTorch · torchvision · OpenCV"),
    "GenAI & LLMs": dict(title="RAG Chatbot over Your College Syllabus",
        pitch="Build a retrieval-augmented chatbot that answers from real documents, the GenAI project that proves you understand RAG, not just the acronym.",
        stack="Python · embeddings · vector store · LLM API"),
    "Deployment & MLOps": dict(title="Sentiment API, Live on the Internet",
        pitch="Wrap a model in an API and ship it publicly so your resume has a link a recruiter can actually click.",
        stack="FastAPI / Streamlit · Hugging Face Spaces"),
}

SAMPLE_RESUME = """Ravi Kumar | B.Tech CSE (AI & ML), 2026 | ravi@email.com
SUMMARY: Passionate AI engineer specialising in Large Language Models, RAG pipelines, transformer architectures and post-hoc hallucination detection frameworks for LLMs.
SKILLS: Generative AI, LLM fine-tuning, LoRA, LangChain, Transformers, Attention mechanisms, AI Agents, Agentic workflows, Vector Database, Prompt Engineering, Deep Learning, Computer Vision, NLP.
PROJECTS:
- Hallucination Detection Framework for LLMs - explored state-of-the-art techniques and research papers.
- Agentic RAG Assistant - designed an intelligent multi-agent system architecture.
- Organised a college AI awareness workshop for juniors.
CERTIFICATIONS: Generative AI Fundamentals, Prompt Engineering for Everyone.
"""
