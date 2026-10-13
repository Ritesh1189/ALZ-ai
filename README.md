# ALZ ai: AI Resume Intelligence & Career Recommendation System

An enterprise recruitment intelligence and resume optimization platform. Built with **Streamlit**, **Scikit-Learn**, **Plotly**, and modern **NLP** algorithms, it provides automated ATS compliance diagnostics, hybrid semantic career matching, interactive scenario simulators, and personalized upskilling roadmaps.

---

## Core Capabilities

### 1. 6-Pillar ATS Compliance Audit & Interactive What-If Simulator
- **Automated ATS Scoring (0–100)** with letter grades (**A, B, C, D**) and tier classifications.
- **Section Architecture Analysis (25 pts):** Segments and validates standard corporate sections (*Summary*, *Skills*, *Experience*, *Education*, *Projects*, *Certifications*).
- **Strong Action Verbs Density (20 pts):** Detects 100+ high-impact action verbs (e.g., *architected*, *spearheaded*, *automated*) while flagging weak passive phrasing.
- **Quantifiable Results & Metrics (20 pts):** Identifies percentages, dollar metrics, and scale indicators to verify evidence-based bullet points.
- **Length & Readability Health (15 pts):** Gauges optimal document volume and reading times.
- **Contact & Portfolio Visibility (10 pts):** Verifies email, phone, LinkedIn, and GitHub profile visibility.
- **Technical Keyword Saturation (10 pts):** Benchmarks domain-specific vocabulary saturation.
- **Interactive What-If Simulator:** Real-time projected score calculator allowing candidates to simulate revisions and quantify score lift.

### 2. Hybrid Job Role Recommendation Engine
- **Dual-Factor Match Algorithm:**
  $$\text{Match Score} = (\alpha \times \text{Skill Overlap}) + ((1 - \alpha) \times \text{Semantic TF-IDF Cosine Similarity})$$
- Evaluates resumes across 10 curated tech career roles:
  - *Machine Learning Engineer*
  - *AI Engineer*
  - *Data Scientist*
  - *Data Analyst*
  - *NLP Engineer*
  - *Computer Vision Engineer*
  - *Full Stack Developer*
  - *Software Engineer*
  - *Cloud Engineer*
  - *DevOps Engineer*
- **Interactive Role Filter & Comparison Matrix:** Search by job title, filter by minimum match score, and toggle between role cards and structured tabular comparison matrices.

### 3. Competency Gap Analysis & Bridge-the-Gap Simulator
- Select any career benchmark to evaluate readiness and skill deltas.
- Prioritized missing skills breakdown: **Critical**, **Important**, and **Recommended**.
- **Interactive Bridge-the-Gap Simulator:** Select skills you plan to acquire to view live projected readiness percentage increases.
- Practical portfolio projects, core concepts, and recognized industry credentials for each gap.
- **Phased 3-Milestone Learning Timeline** with estimated study completion weeks.

### 4. Technical Competency Matrix & Radar Profiler
- **Plotly Polar Radar Chart:** Visualizes candidate strengths across 7 technology domains:
  - *Programming Languages*
  - *Machine Learning & AI*
  - *Web & Frontend*
  - *Web & Backend*
  - *Data Science & Analytics*
  - *Cloud & DevOps*
  - *Database Technologies*
- Interactive skill search filter and structured document section inspector.

### 5. STAR Method Bullet Point Studio
- Interactive editor to evaluate and rewrite resume accomplishment statements.
- Real-time diagnostic: Action verb presence, quantifiable impact metrics, and technical keyword extraction.
- Generates 3 enhanced executive alternatives across *Scale & Performance*, *Business ROI*, and *Systems Architecture*.

### 6. Executive Report Export
- Single-click export of complete analysis as **Markdown (`.md`)** or **JSON (`.json`)**.

---

## Architecture & Modules

```
AI resume analyzer/
├── app.py                  # Enterprise Streamlit web application
├── resume_parser.py        # In-memory document parser (PDF, DOCX, TXT)
├── text_cleaner.py         # Text normalization & section boundary regex
├── skill_extractor.py      # High-precision multi-word token & alias matching
├── recommender.py          # TF-IDF cosine similarity & hybrid role matcher
├── ats_scorer.py           # 6-pillar ATS evaluation engine
├── gap_analyzer.py         # Upskilling roadmaps, milestones & projects
├── test_pipeline.py        # Automated test suite
├── requirements.txt        # Dependency definitions
├── .env                    # Environment configuration
├── data/
│   ├── job_roles.csv       # Curated role descriptions & skill sets
│   └── skill_dictionary.csv# 110+ technical skills with regex aliases
└── samples/
    ├── ai_ml_engineer.txt       # Senior AI & ML Engineer sample profile
    ├── full_stack_developer.txt # Lead Full Stack Developer sample profile
    └── data_analyst.txt         # Senior Data Analyst sample profile
```

---

## Installation & Execution

### 1. Prerequisites
- Python 3.10+ (recommended Python 3.11 or 3.12)
- Virtual environment (`venv`)

### 2. Setup
```bash
# Clone or navigate to the repository
cd "AI resume analyzer"

# Activate the virtual environment
# Windows:
.\venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run Self-Test Suite
```bash
python test_pipeline.py
```

### 4. Launch the Web Application
```bash
streamlit run app.py
```

The application will be accessible at: **`http://localhost:8501`**

---

## Privacy & Security
- **100% In-Memory Processing:** Uploaded documents are parsed strictly in memory and are never saved to permanent disk storage or shared with external third-party model APIs.
- **Zero External API Cost:** Runs entirely locally using Scikit-Learn TF-IDF vectorization and curated knowledge bases.
