# CineMatch: A Hybrid Content-Based and Collaborative Filtering Movie Recommendation System

**Academic Project by Pavni Pawar (Enrollment: 24B5044)**  
**Supervisor:** Dr. Chandra Prakash Patidar  
**Department of Computer Science and Business Systems, Institute of Engineering & Technology, Devi Ahilya Vishwavidyalaya, Indore (M.P.)**  
**Academic Session:** 2025 – 2026  

---

## 🎬 Project Overview

**CineMatch** is an end-to-end, academic-grade movie recommendation engine wrapped in a cinematic, dark-themed OTT streaming interface. It combines two foundational recommendation paradigms—**Content-Based Filtering** (using TF-IDF vectorization and cosine similarity over rich metadata) and **Collaborative Filtering** (using Regularized Singular Value Decomposition Matrix Factorization and Memory-Based K-Nearest Neighbors)—into a unified, **Weighted Hybrid Recommendation Engine** equipped with an adaptive cold-start fallback mechanism.

### Key Highlights
- **Real ML Implementation:** Trained and evaluated on the official GroupLens **MovieLens latest-small** dataset (100,836 ratings across 9,742 movies and 610 users).
- **Leak-Free Evaluation:** Rigorous 80/20 train/test split computing **RMSE, MAE, Precision@K (K=5,10,20), Recall@K**, and Catalog Coverage.
- **Adaptive Cold-Start Solution:** Dynamically shifts recommendation weight to Content-Based profiling when user rating volume is below threshold ($N < 5$), with an interactive onboarding quiz for brand-new users.
- **Explainable Recommendations:** Every recommended title shows the exact reasoning (e.g. *"Hybrid Recommendation: 60% taste pattern from similar users + 40% content match on Sci-Fi themes"*).
- **Cinematic OTT UI:** Ultra-modern dark-theme interface built with Streamlit and custom CSS, featuring hero billboards, movie poster cards, glassmorphism badges, and modal dialogs.
- **Resilient Poster Architecture:** Built-in curated metadata for top classic blockbusters, live TMDB v3 API querying with disk caching, and dynamic SVG poster card generation when offline or missing keys.

---

## 🚀 Quickstart & Execution (Windows)

### Prerequisites
- Python 3.10+ (Tested and verified on Python 3.14)
- Git (optional)

### 1. Navigate to Project Directory
```powershell
cd C:\Users\Hp\.gemini\antigravity\scratch\cinematch
```

### 2. Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Launch the Application (Single Command)
```powershell
python run.py
```
*Alternatively, you can launch directly via Streamlit:*
```powershell
python -m streamlit run app.py
```
Open your browser at **`http://localhost:8501`**.

---

## 📁 Project Architecture

```
cinematch/
│
├── data/                               # Dataset & persistent storage
│   ├── ml-latest-small/                # Auto-downloaded MovieLens dataset
│   │   ├── movies.csv                  # 9,742 movies with genres
│   │   ├── ratings.csv                 # 100,836 ratings (0.5 to 5.0)
│   │   ├── links.csv                   # IMDb and TMDB identifiers
│   │   └── tags.csv                    # 3,683 user-contributed tags
│   ├── evaluation_results.json         # Real pre-computed model evaluation metrics
│   └── tmdb_cache.json                 # Disk cache for live TMDB queries
│
├── src/                                # Modular core algorithms
│   ├── __init__.py                     # Package export declarations
│   ├── data_loader.py                  # Acquisition, cleaning, title parsing, EDA aggregations
│   ├── curated_data.py                 # High-res curated metadata & dynamic SVG poster generator
│   ├── tmdb_client.py                  # TMDB API client with fallback & caching
│   ├── content_based.py                # TF-IDF vectorizer, metadata soup & cosine similarity
│   ├── collaborative.py                # Surprise SVD, KNNWithMeans & pure-NumPy FunkSVD
│   ├── hybrid.py                       # Adaptive weighted hybrid engine & cold-start handler
│   └── evaluation.py                   # Leak-free train/test split & quantitative metrics
│
├── app.py                              # OTT dark-themed Streamlit application
├── run.py                              # One-click health check and launch script
├── requirements.txt                    # Exact production dependencies
├── .env.example                        # Environment variable template for TMDB API key
├── .gitignore                          # Standard Python & environment exclusions
├── VIVA_QUESTIONS.md                   # Comprehensive 25+ viva questions and answers
└── README.md                           # Complete project documentation
```

---

## 🧪 Model Evaluation & Real Benchmark Results

Evaluated on held-out test data (80% Train: 80,668 ratings / 20% Test: 20,168 ratings) with relevance threshold $r \ge 3.5$:

| Recommender Model | Paradigm | RMSE (↓) | MAE (↓) | Precision@10 (↑) | Recall@10 (↑) | Catalog Coverage |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Collaborative (SVD)** | Matrix Factorization ($k=40$) | **0.8859** | **0.6789** | 0.6800 | 0.6277 | 6.69% |
| **Collaborative (KNN)** | Memory-Based Cosine | 0.9204 | 0.7050 | 0.6787 | 0.6321 | 7.48% |
| **Content-Based (TF-IDF)** | Metadata Soup Similarity | 1.0391 | 0.8466 | 0.6213 | 0.6112 | **8.30%** |
| **CineMatch Hybrid** | **Adaptive Weighted Blend** | **0.9032** | **0.7090** | **0.6813** | **0.6302** | 6.71% |

### Key Findings
1. **SVD** achieves the lowest prediction error (**RMSE: 0.8859**), significantly outperforming neighborhood KNN because its 40 latent factors overcome the **98.3% matrix sparsity**.
2. **CineMatch Hybrid** achieves the highest top-10 precision (**0.6813**) and top-20 recall (**0.7861**), balancing personalized serendipity from SVD with thematic consistency from TF-IDF.
3. **Content-Based Filtering** exhibits the highest catalog coverage (**8.30%**), effectively pulling hidden gems from the long tail.

---

## 🛠️ Optional: Configuring TMDB Live API

CineMatch works seamlessly offline using built-in curated posters and dynamic SVG cards. If you wish to connect live to TMDB for all 9,742 catalog titles:
1. Create a free account at [The Movie Database (TMDB)](https://www.themoviedb.org/).
2. Generate an API Key under **Account Settings → API**.
3. Create a `.env` file in the project root:
   ```env
   TMDB_API_KEY=your_actual_tmdb_api_key_here
   ```
4. *Or simply enter your key live inside the CineMatch web interface under **Settings & TMDB API**.*

---

## 🎓 Academic Defense & Viva Preparation

A detailed study guide covering mathematical equations, SGD derivations, cold-start strategies, and 25 viva questions with answers is available in **`VIVA_QUESTIONS.md`**.
