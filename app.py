"""
CineMatch: AI-Powered Hybrid Movie Recommendation Engine
High-Performance Content-Based & Collaborative Filtering Platform
Built with Python, Scikit-learn, Surprise (SVD/KNN), and Streamlit
"""

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
import json
import time
import random
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List

from src.data_loader import DataLoader
from src.content_based import ContentBasedRecommender
from src.collaborative import CollaborativeRecommender
from src.hybrid import HybridRecommender
from src.evaluation import ModelEvaluator
from src.tmdb_client import TMDBClient

# ==========================================
# 1. Page Configuration & Custom OTT Styling
# ==========================================
st.set_page_config(
    page_title="CineMatch — AI Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Premium OTT Dark Cinematic CSS with Top Navigation
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    /* Global Typography & Base Styling */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    h1, h2, h3, h4, h5, .hero-title, .brand-logo {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    /* Main background */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #151928 0%, #0a0b10 50%, #06070a 100%);
        color: #e2e8f0;
    }

    /* Top Brand Bar & Sticky Header */
    .top-header-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 14px 28px;
        background: rgba(12, 15, 24, 0.85);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        margin-bottom: 22px;
    }
    .brand-logo {
        font-size: 26px;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 0%, #E50914 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: flex;
        align-items: center;
        gap: 8px;
        text-shadow: 0 0 20px rgba(229, 9, 20, 0.4);
    }
    .status-badge {
        background: rgba(229, 9, 20, 0.15);
        border: 1px solid rgba(229, 9, 20, 0.4);
        color: #ff4d58;
        font-size: 11px;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 20px;
        text-transform: uppercase;
        letter-spacing: 1px;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    /* Top Navigation Pills Styling */
    div[data-testid="stRadio"] > div[role="radiogroup"] {
        display: flex;
        flex-direction: row;
        justify-content: center;
        gap: 8px;
        background: rgba(16, 20, 32, 0.7);
        padding: 8px 12px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.06);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
        margin-bottom: 28px;
        overflow-x: auto;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] > label {
        background: transparent !important;
        padding: 8px 16px !important;
        border-radius: 10px !important;
        font-size: 13.5px !important;
        font-weight: 600 !important;
        color: #94a3b8 !important;
        border: 1px solid transparent !important;
        transition: all 0.25s ease !important;
        cursor: pointer !important;
        white-space: nowrap !important;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {
        color: #ffffff !important;
        background: rgba(255, 255, 255, 0.05) !important;
        border-color: rgba(255, 255, 255, 0.1) !important;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"],
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) {
        background: linear-gradient(135deg, #E50914 0%, #b80710 100%) !important;
        color: #ffffff !important;
        border-color: #E50914 !important;
        box-shadow: 0 4px 16px rgba(229, 9, 20, 0.4) !important;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] input {
        display: none !important;
    }

    /* Hero Billboard Section */
    .hero-container {
        position: relative;
        border-radius: 22px;
        padding: 44px 40px;
        margin-bottom: 34px;
        background: linear-gradient(135deg, rgba(229, 9, 20, 0.18) 0%, rgba(18, 22, 35, 0.96) 55%, #0c0e17 100%);
        border: 1px solid rgba(229, 9, 20, 0.3);
        box-shadow: 0 20px 48px rgba(0, 0, 0, 0.7);
        overflow: hidden;
    }
    .hero-badge {
        display: inline-block;
        background: #E50914;
        color: #ffffff;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 1.5px;
        padding: 4px 12px;
        border-radius: 20px;
        margin-bottom: 14px;
        text-transform: uppercase;
        box-shadow: 0 2px 10px rgba(229, 9, 20, 0.5);
    }
    .hero-title {
        font-size: 40px;
        line-height: 1.15;
        color: #ffffff;
        margin-bottom: 12px;
    }
    .hero-meta {
        display: flex;
        align-items: center;
        gap: 16px;
        color: #94a3b8;
        font-size: 14px;
        margin-bottom: 16px;
        flex-wrap: wrap;
    }
    .hero-desc {
        color: #cbd5e1;
        font-size: 15px;
        line-height: 1.6;
        max-width: 760px;
        margin-bottom: 22px;
    }

    /* Movie Poster Card Component */
    .movie-card {
        background: #121624;
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.08);
        transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
        margin-bottom: 16px;
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    .movie-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 18px 36px rgba(0, 0, 0, 0.8), 0 0 20px rgba(229, 9, 20, 0.25);
        border-color: rgba(229, 9, 20, 0.45);
    }
    .poster-img-container {
        position: relative;
        width: 100%;
        padding-top: 148%;
        background-color: #181d2e;
        overflow: hidden;
    }
    .poster-img-container img {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        object-fit: cover;
    }
    .movie-card-body {
        padding: 14px 16px;
        flex-grow: 1;
        display: flex;
        flex-direction: column;
    }
    .card-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 15.5px;
        color: #ffffff;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        margin-bottom: 6px;
    }
    .card-meta {
        font-size: 12px;
        color: #94a3b8;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }
    .rating-badge {
        color: #ffb800;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 4px;
    }
    .genre-badge {
        background: rgba(255, 255, 255, 0.07);
        color: #cbd5e1;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 11px;
        display: inline-block;
        margin-right: 4px;
        margin-bottom: 4px;
    }
    .match-pill {
        background: linear-gradient(90deg, #10b981 0%, #059669 100%);
        color: #ffffff;
        font-weight: 700;
        font-size: 11px;
        padding: 3px 8px;
        border-radius: 12px;
        display: inline-block;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.4);
    }
    .explanation-box {
        background: rgba(255, 255, 255, 0.03);
        border-left: 3px solid #E50914;
        padding: 6px 10px;
        border-radius: 0 6px 6px 0;
        font-size: 11.5px;
        color: #94a3b8;
        margin-top: 10px;
        line-height: 1.45;
    }

    /* Metric KPI Tiles */
    .kpi-tile {
        background: #121624;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
    }
    .kpi-value {
        font-family: 'Outfit', sans-serif;
        font-size: 32px;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 4px;
    }
    .kpi-label {
        font-size: 12.5px;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Latency & Runtime Indicator Pill */
    .latency-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.35);
        color: #34d399;
        font-size: 12px;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 20px;
        margin-bottom: 16px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ==========================================
# 2. Cached Engines & Resource Loading
# ==========================================
@st.cache_resource(show_spinner="Initializing CineMatch AI Architecture...")
def get_data_loader() -> DataLoader:
    d_dir = "data" if os.path.exists("data") else "src/data"
    return DataLoader(data_dir=d_dir)

@st.cache_resource(show_spinner="Fitting Content-Based Engine (TF-IDF & Cosine)...")
def get_content_recommender(_dl: DataLoader) -> ContentBasedRecommender:
    cb = ContentBasedRecommender(max_features=8000)
    cb.fit(_dl.df_movies)
    return cb

@st.cache_resource(show_spinner="Training SVD Matrix Factorization Engine...")
def get_collaborative_recommender(_dl: DataLoader) -> CollaborativeRecommender:
    cf = CollaborativeRecommender(algorithm="svd", n_factors=40, n_epochs=20)
    cf.fit(_dl.df_ratings)
    return cf

@st.cache_resource(show_spinner="Connecting TMDB Metadata Cache...")
def get_tmdb_client() -> TMDBClient:
    c_path = "data/tmdb_cache.json" if os.path.exists("data") else "src/data/tmdb_cache.json"
    return TMDBClient(cache_file=c_path)

# Initialize engines
dl = get_data_loader()
cb_recommender = get_content_recommender(dl)
cf_recommender = get_collaborative_recommender(dl)
tmdb = get_tmdb_client()
hybrid_recommender = HybridRecommender(cb_recommender, cf_recommender)

# Session state initialization
if "watchlist" not in st.session_state:
    st.session_state["watchlist"] = set([318, 2571, 79132])  # Default saved titles
if "custom_ratings" not in st.session_state:
    st.session_state["custom_ratings"] = {}
if "hero_movie_id" not in st.session_state:
    st.session_state["hero_movie_id"] = 79132  # Inception by default
if "active_user_id" not in st.session_state:
    st.session_state["active_user_id"] = 1

# ==========================================
# 3. Top Header Bar & Navigation Bar
# ==========================================
has_tmdb = tmdb.has_valid_key()
tmdb_badge = "TMDB Live Connected" if has_tmdb else "High-Res Curated & Dynamic SVG"

st.markdown(f"""
    <div class="top-header-bar">
        <div class="brand-logo">
            <span>🎬</span> CINEMATCH <span style="font-size: 13px; font-weight: 500; color: #94a3b8; margin-left: 6px;">AI Streaming Engine</span>
        </div>
        <div style="display: flex; align-items: center; gap: 14px;">
            <div class="status-badge">
                <span style="width: 7px; height: 7px; border-radius: 50%; background-color: #10b981;"></span>
                <span>{tmdb_badge}</span>
            </div>
            <div style="font-size: 12px; color: #94a3b8; font-weight: 500;">
                Catalog: <b style="color: #fff;">{len(dl.df_movies):,}</b> titles
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# TOP NAVIGATION BAR (Horizontal Radio Pills)
nav_choice = st.radio(
    "Primary Navigation",
    [
        "🎬 Home & Spotlight",
        "🎯 Discover Similar",
        "👤 Personalized For You",
        "📊 Catalog Analytics",
        "🧪 Model Benchmarks",
        "🔖 My Watchlist",
        "⚙️ Settings"
    ],
    horizontal=True,
    label_visibility="collapsed"
)

# ==========================================
# 4. Helper UI Rendering Functions
# ==========================================
def render_movie_card(movie_row: Dict[str, Any], show_match: bool = False, show_explanation: bool = True, key_prefix: str = "card"):
    """Renders a cinematic OTT poster card with metadata and actions."""
    mid = int(movie_row["movieId"])
    title = movie_row.get("clean_title", movie_row.get("title", "Untitled"))
    year = str(movie_row.get("year", ""))
    p_genre = movie_row.get("primary_genre", "Cinema")
    genres = movie_row.get("genres_list", [p_genre])
    rating_mean = movie_row.get("rating_mean", 0.0)
    rating_cnt = movie_row.get("rating_count", 0)
    tmdb_id = movie_row.get("tmdbId")

    # Fetch poster URL from TMDB client
    details = tmdb.get_movie_details(tmdb_id, title=title, year=year, primary_genre=p_genre)
    poster_url = details.get("poster_url")

    # Match / predicted rating display
    match_html = ""
    if show_match:
        if "match_pct" in movie_row:
            match_html = f"<span class='match-pill'>{movie_row['match_pct']} MATCH</span>"
        elif "similarity_pct" in movie_row:
            match_html = f"<span class='match-pill'>{movie_row['similarity_pct']} SIMILAR</span>"
        elif "predicted_rating" in movie_row:
            match_html = f"<span class='match-pill'>★ {movie_row['predicted_rating']:.1f} PRED</span>"

    genre_badges = "".join([f"<span class='genre-badge'>{g}</span>" for g in genres[:2]])

    exp_html = ""
    if show_explanation and movie_row.get("explanation"):
        exp_html = f"<div class='explanation-box'>{movie_row['explanation']}</div>"

    card_html = f"""
    <div class="movie-card">
        <div class="poster-img-container">
            <img src="{poster_url}" alt="{title}" loading="lazy" />
        </div>
        <div class="movie-card-body">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 4px;">
                <div class="card-title" title="{title}">{title}</div>
            </div>
            <div class="card-meta">
                <span>{year}</span>
                <span class="rating-badge">★ {rating_mean:.1f} <span style="font-size: 10px; color: #64748b;">({rating_cnt:,})</span></span>
                {match_html}
            </div>
            <div>{genre_badges}</div>
            {exp_html}
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

    # Action buttons
    col_act1, col_act2 = st.columns([1, 1])
    with col_act1:
        is_in_watchlist = mid in st.session_state["watchlist"]
        btn_label = "✓ Saved" if is_in_watchlist else "+ Watchlist"
        if st.button(btn_label, key=f"{key_prefix}_wl_{mid}", use_container_width=True):
            if is_in_watchlist:
                st.session_state["watchlist"].remove(mid)
            else:
                st.session_state["watchlist"].add(mid)
            st.rerun()

    with col_act2:
        if st.button("Details", key=f"{key_prefix}_dt_{mid}", use_container_width=True):
            st.session_state["selected_modal_movie"] = mid
            st.rerun()


# ==========================================
# 5. Page Views
# ==========================================

# ----------------- TAB 1: HOME & SPOTLIGHT ----------------- #
if nav_choice == "🎬 Home & Spotlight":
    # Dynamic Hero Movie Selection (User can shuffle or pick any title)
    hero_mid = st.session_state["hero_movie_id"]
    if hero_mid not in dl.df_movies['movieId'].values:
        hero_mid = 79132
    
    hero_movie = dl.df_movies[dl.df_movies['movieId'] == hero_mid].iloc[0]
    hero_details = tmdb.get_movie_details(hero_movie.get('tmdbId'), hero_movie['clean_title'], hero_movie['year'], hero_movie['primary_genre'])

    st.markdown(f"""
        <div class="hero-container">
            <span class="hero-badge">Featured Cinematic Spotlight</span>
            <div class="hero-title">{hero_movie['clean_title']} ({hero_movie['year']})</div>
            <div class="hero-meta">
                <span style="color: #ffb800; font-weight: 700;">★ {hero_movie['rating_mean']:.1f} Rating</span>
                <span>•</span>
                <span>{hero_movie['genres'].replace('|', ' • ')}</span>
                <span>•</span>
                <span>Dir: {hero_details.get('director', 'Acclaimed Director')}</span>
            </div>
            <div class="hero-desc">{hero_details.get('overview', hero_movie['overview'])}</div>
        </div>
    """, unsafe_allow_html=True)

    # Hero quick interactive controls
    h_col1, h_col2, h_col3 = st.columns([2, 2, 3])
    with h_col1:
        if st.button("🎯 Find Similar Movies", type="primary", use_container_width=True):
            st.session_state["seed_movie_id"] = int(hero_mid)
            st.rerun()
    with h_col2:
        is_hero_saved = int(hero_mid) in st.session_state["watchlist"]
        h_btn_txt = "✓ In Watchlist" if is_hero_saved else "+ Add to Watchlist"
        if st.button(h_btn_txt, use_container_width=True):
            if is_hero_saved:
                st.session_state["watchlist"].remove(int(hero_mid))
            else:
                st.session_state["watchlist"].add(int(hero_mid))
            st.rerun()
    with h_col3:
        if st.button("🎲 Shuffle Featured Spotlight", use_container_width=True):
            top_pool = dl.get_top_rated_movies(min_ratings=40, top_n=60)['movieId'].tolist()
            st.session_state["hero_movie_id"] = random.choice(top_pool)
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Real-Time Interactive Catalog Search
    st.markdown("### 🔍 Live Instant Catalog Search")
    col_search, col_year, col_min_r = st.columns([3, 2, 2])
    with col_search:
        search_query = st.text_input("Search 9,742 movies by title, director, or keyword:", placeholder="e.g. Nolan, Matrix, Tarantino, Space...")
    with col_year:
        year_filter = st.slider("Release Year Range:", min_value=1920, max_value=2025, value=(1970, 2025))
    with col_min_r:
        min_rating_filter = st.slider("Minimum Rating:", min_value=0.5, max_value=5.0, value=3.0, step=0.5)

    if search_query.strip():
        t0 = time.time()
        q = search_query.lower().strip()
        matched = dl.df_movies[
            dl.df_movies['soup'].str.contains(q, na=False) |
            dl.df_movies['clean_title'].str.lower().str.contains(q, na=False)
        ]
        matched = matched[
            (matched['rating_mean'] >= min_rating_filter) &
            (matched['year'].apply(lambda y: int(y) if y.isdigit() and len(y)==4 else 2000).between(year_filter[0], year_filter[1]))
        ].sort_values(by=['rating_count', 'bayesian_rating'], ascending=[False, False])
        
        search_elapsed = (time.time() - t0) * 1000
        st.markdown(f"<div class='latency-pill'>⚡ Found {len(matched):,} matching titles in {search_elapsed:.1f}ms (Real-Time Search)</div>", unsafe_allow_html=True)

        if not matched.empty:
            s_cols = st.columns(5)
            for i, (_, row) in enumerate(matched.head(5).iterrows()):
                with s_cols[i]:
                    render_movie_card(row.to_dict(), show_match=False, key_prefix=f"search_{row['movieId']}")
        else:
            st.info("No matching movies found for this search criteria.")
        st.markdown("---")

    # Section 1: Trending & Most Popular
    st.markdown("### 🔥 Trending & Popular in the Catalog")
    st.caption("Most frequently rated titles across community viewing patterns")
    trending_movies = dl.get_trending_movies(top_n=5)
    cols = st.columns(5)
    for i, (_, row) in enumerate(trending_movies.iterrows()):
        with cols[i]:
            render_movie_card(row.to_dict(), show_match=False, key_prefix="trend")

    st.markdown("<br>", unsafe_allow_html=True)

    # Section 2: Top Rated All-Time Masterpieces (Bayesian Weighted)
    st.markdown("### ⭐ Top Rated Masterpieces (Bayesian Statistically Weighted)")
    st.caption("Normalized using Bayesian rating formula WR = (v/(v+m))R + (m/(v+m))C to eliminate small-sample bias")
    top_rated = dl.get_top_rated_movies(min_ratings=40, top_n=5)
    cols = st.columns(5)
    for i, (_, row) in enumerate(top_rated.iterrows()):
        with cols[i]:
            render_movie_card(row.to_dict(), show_match=False, key_prefix="top")

    st.markdown("<br>", unsafe_allow_html=True)

    # Section 3: Explore by Genre
    st.markdown("### 🎭 Dynamic Genre Discovery")
    genre_list = [g for g in dl.get_genre_distribution()['genre'].tolist() if g != "(no genres listed)"]
    selected_genre = st.selectbox("Select Genre Filter:", genre_list, index=0)

    genre_filtered = dl.df_movies[dl.df_movies['genres_list'].apply(lambda gl: selected_genre in gl)]\
        .sort_values(by=['bayesian_rating', 'rating_count'], ascending=[False, False])\
        .head(10)

    g_cols = st.columns(5)
    for i, (_, row) in enumerate(genre_filtered.head(5).iterrows()):
        with g_cols[i]:
            render_movie_card(row.to_dict(), show_match=False, key_prefix=f"gen_{selected_genre}_1")

    if len(genre_filtered) > 5:
        g_cols2 = st.columns(5)
        for i, (_, row) in enumerate(genre_filtered.iloc[5:10].iterrows()):
            with g_cols2[i]:
                render_movie_card(row.to_dict(), show_match=False, key_prefix=f"gen_{selected_genre}_2")


# ----------------- TAB 2: DISCOVER SIMILAR ----------------- #
elif nav_choice == "🎯 Discover Similar":
    st.markdown("## 🎯 Discover Similar Titles")
    st.markdown("##### Real-Time Content-Based Filtering using TF-IDF & Cosine Similarity")
    st.caption("Computes high-dimensional cosine angle over plot overviews, genres, directors, cast, and community keywords.")

    # Search / Select ANY Movie from all 9,742 catalog entries (Non-hardcoded)
    movie_options = dl.df_movies[['movieId', 'clean_title', 'year', 'rating_mean']].sort_values(by='clean_title')
    movie_dict = {f"{row.clean_title} ({row.year}) — ★ {row.rating_mean:.1f}": row.movieId for _, row in movie_options.iterrows()}
    
    default_index = 0
    if "seed_movie_id" in st.session_state:
        for idx, (label, mid) in enumerate(movie_dict.items()):
            if mid == st.session_state["seed_movie_id"]:
                default_index = idx
                break

    col_select_seed, col_pick_random = st.columns([4, 1])
    with col_select_seed:
        selected_label = st.selectbox("Select or Search ANY Movie in the Catalog:", list(movie_dict.keys()), index=default_index)
        seed_mid = movie_dict[selected_label]
    with col_pick_random:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("🎲 Random Title", use_container_width=True):
            random_choice_mid = random.choice(list(movie_dict.values()))
            st.session_state["seed_movie_id"] = random_choice_mid
            st.rerun()

    # Display Seed Movie Profile
    seed_movie = dl.df_movies[dl.df_movies['movieId'] == seed_mid].iloc[0]
    seed_details = tmdb.get_movie_details(seed_movie.get('tmdbId'), seed_movie['clean_title'], seed_movie['year'], seed_movie['primary_genre'])

    col_seed_img, col_seed_info = st.columns([1, 3])
    with col_seed_img:
        st.markdown(f"""
            <div style="border-radius: 12px; overflow: hidden; border: 1px solid rgba(255,255,255,0.12); box-shadow: 0 10px 25px rgba(0,0,0,0.6);">
                <img src="{seed_details['poster_url']}" style="width: 100%; display: block;" />
            </div>
        """, unsafe_allow_html=True)
    with col_seed_info:
        st.markdown(f"### {seed_movie['clean_title']} ({seed_movie['year']})")
        st.markdown(f"**Genres:** {seed_movie['genres'].replace('|', ', ')} | **Rating:** ★ {seed_movie['rating_mean']:.1f} ({seed_movie['rating_count']:,} reviews)")
        if seed_details.get("director"):
            st.markdown(f"**Director:** {seed_details['director']} | **Cast:** {', '.join(seed_details.get('cast', [])[:4])}")
        st.markdown(f"**Plot Synopsis:** {seed_details.get('overview', seed_movie['overview'])}")
        if seed_movie['user_tags']:
            st.markdown(f"**Keywords & Tags:** `{seed_movie['user_tags'][:140]}...`")

    st.markdown("---")
    
    col_ctrl1, col_ctrl2 = st.columns([2, 4])
    with col_ctrl1:
        top_k = st.slider("Number of Similar Recommendations:", min_value=4, max_value=20, value=8, step=2)

    t_cb_start = time.time()
    similar_recs = cb_recommender.get_similar_movies(seed_mid, top_n=top_k)
    cb_latency = (time.time() - t_cb_start) * 1000

    st.markdown(f"<div class='latency-pill'>⚡ Computed {len(similar_recs)} similarity scores in {cb_latency:.1f}ms in Real-Time</div>", unsafe_allow_html=True)
    st.markdown(f"### Top {top_k} Titles Most Similar to *{seed_movie['clean_title']}*")

    if not similar_recs:
        st.warning("No similar movies found above the similarity threshold.")
    else:
        for row_start in range(0, len(similar_recs), 4):
            cols = st.columns(4)
            for c_idx, rec in enumerate(similar_recs[row_start:row_start + 4]):
                with cols[c_idx]:
                    render_movie_card(rec, show_match=True, show_explanation=True, key_prefix=f"cb_{seed_mid}_{rec['movieId']}")


# ----------------- TAB 3: PERSONALIZED FOR YOU ----------------- #
elif nav_choice == "👤 Personalized For You":
    st.markdown("## 👤 Personalized Recommendations Engine")
    st.markdown("##### Adaptive Hybrid Architecture (SVD Latent Collaborative Filtering + Content Thematic Profile)")

    persona_mode = st.radio(
        "Select User Context:",
        ["Explore Community User Profile (1 to 610)", "✨ Interactive Live Personalization (Rate Movies in Real-Time)"],
        horizontal=True
    )

    if persona_mode == "Explore Community User Profile (1 to 610)":
        col_u1, col_u2, col_u3, col_u4 = st.columns([2, 1, 2, 2])
        with col_u1:
            selected_uid = st.number_input("User ID (1 – 610):", min_value=1, max_value=610, value=st.session_state["active_user_id"], step=1)
            st.session_state["active_user_id"] = selected_uid

        with col_u2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("🎲 Random User", use_container_width=True):
                st.session_state["active_user_id"] = random.randint(1, 610)
                st.rerun()

        with col_u3:
            alpha_val = st.slider("Hybrid Weight (α CF vs 1-α CB):", min_value=0.0, max_value=1.0, value=0.60, step=0.05,
                                  help="1.0 = Pure Collaborative SVD, 0.0 = Pure Content Profile, 0.6 = Optimal Hybrid Blend")

        with col_u4:
            num_recs = st.slider("Recommendations Count:", min_value=4, max_value=16, value=8, step=2)

        # Dynamic retrieval of user rating history
        user_ratings_df = dl.df_ratings[dl.df_ratings['userId'] == selected_uid].sort_values(by='rating', ascending=False)
        user_ratings_dict = dict(zip(user_ratings_df['movieId'], user_ratings_df['rating']))

        # User profile summary expander
        with st.expander(f"🔍 Inspect User #{selected_uid}'s Real Taste History ({len(user_ratings_df)} rated titles)", expanded=False):
            st.markdown(f"**Mean Rating Given:** ★ {user_ratings_df['rating'].mean():.2f} / 5.0 | **Total Ratings:** {len(user_ratings_df):,}")
            top_user_films = pd.merge(user_ratings_df.head(8), dl.df_movies, on='movieId')
            st.dataframe(top_user_films[['clean_title', 'year', 'genres', 'rating']], use_container_width=True)

        # Real-time hybrid computation
        t_hyb_start = time.time()
        hybrid_recs = hybrid_recommender.recommend_for_user(
            user_id=selected_uid,
            user_ratings_dict=user_ratings_dict,
            catalog_movies=dl.df_movies,
            top_n=num_recs,
            alpha=alpha_val
        )
        hyb_latency = (time.time() - t_hyb_start) * 1000

        st.markdown(f"<div class='latency-pill'>⚡ Live Inference: {hyb_latency:.1f}ms | Effective Collaborative Alpha: {hybrid_recommender.calculate_effective_alpha(len(user_ratings_dict), alpha_val):.2f}</div>", unsafe_allow_html=True)
        st.markdown(f"### Hybrid Recommendations for User #{selected_uid}")
        st.caption(f"Personalized with α = {alpha_val*100:.0f}% SVD Latent Taste + {(1-alpha_val)*100:.0f}% Thematic Content Signal")

        for row_start in range(0, len(hybrid_recs), 4):
            cols = st.columns(4)
            for c_idx, rec in enumerate(hybrid_recs[row_start:row_start + 4]):
                with cols[c_idx]:
                    render_movie_card(rec, show_match=True, show_explanation=True, key_prefix=f"hyb_{selected_uid}_{rec['movieId']}")

    else:
        # DYNAMIC LIVE PERSONALIZATION (Cold-Start & Interactive User Rating)
        st.markdown("""
            <div style="background: rgba(229, 9, 20, 0.08); border-left: 4px solid #E50914; padding: 14px 18px; border-radius: 8px; margin-bottom: 22px;">
                <h4 style="margin: 0 0 6px 0; color: #fff;">Real-Time Dynamic Taste Profile</h4>
                <p style="margin: 0; font-size: 13px; color: #cbd5e1;">
                    Rate any movie below in real-time. CineMatch dynamically builds your personal taste vector and updates recommendations on the fly!
                </p>
            </div>
        """, unsafe_allow_html=True)

        col_w1, col_w2 = st.columns([1, 1])
        with col_w1:
            st.markdown("#### Step 1: Select Your Favorite Genres")
            all_genres = [g for g in dl.get_genre_distribution()['genre'].tolist() if g != "(no genres listed)"]
            selected_fav_genres = st.multiselect("Filter seed movie pool by genres:", all_genres, default=["Action", "Sci-Fi"])

        with col_w2:
            st.markdown("#### Search & Add ANY Movie to Rate")
            search_add = st.selectbox(
                "Search catalog to add a custom rated title:",
                options=[""] + [f"{r.clean_title} ({r.year}) [ID:{r.movieId}]" for _, r in dl.df_movies.iterrows()],
                index=0
            )
            if search_add:
                try:
                    custom_mid = int(search_add.split("[ID:")[1].replace("]", ""))
                    if custom_mid not in st.session_state["custom_ratings"]:
                        st.session_state["custom_ratings"][custom_mid] = 4.5
                        st.toast(f"Added {search_add.split(' [ID:')[0]} to your rating profile!", icon="⭐")
                        st.rerun()
                except Exception:
                    pass

        st.markdown("#### Step 2: Rate Titles in Real-Time (0.5 to 5.0 Stars)")
        st.caption("Adjust sliders below — recommendations recalculate dynamically in real-time:")

        # Dynamically query top movies matching selected genres
        if selected_fav_genres:
            genre_seeds = dl.df_movies[dl.df_movies['genres_list'].apply(lambda gl: any(g in gl for g in selected_fav_genres))]\
                .sort_values(by=['bayesian_rating', 'rating_count'], ascending=[False, False])\
                .head(8)
        else:
            genre_seeds = dl.get_top_rated_movies(min_ratings=50, top_n=8)

        seed_cols = st.columns(4)
        for i, (_, srow) in enumerate(genre_seeds.iterrows()):
            mid = int(srow['movieId'])
            with seed_cols[i % 4]:
                current_val = st.session_state["custom_ratings"].get(mid, 0.0)
                star = st.slider(f"{srow['clean_title']} ({srow['year']})", min_value=0.0, max_value=5.0, value=float(current_val), step=0.5, key=f"dyn_seed_{mid}")
                if star > 0:
                    st.session_state["custom_ratings"][mid] = star
                elif mid in st.session_state["custom_ratings"] and star == 0.0:
                    del st.session_state["custom_ratings"][mid]

        user_rated_count = len(st.session_state["custom_ratings"])
        st.info(f"You have scored **{user_rated_count}** movies in your live profile. Collaborative Alpha: **{hybrid_recommender.calculate_effective_alpha(user_rated_count):.2f}**")

        if user_rated_count == 0:
            st.warning("Please rate at least 1 movie above to generate your live personalized recommendations.")
        else:
            t_user_start = time.time()
            cold_recs = hybrid_recommender.recommend_for_user(
                user_id=8888,
                user_ratings_dict=st.session_state["custom_ratings"],
                catalog_movies=dl.df_movies,
                top_n=8
            )
            live_latency = (time.time() - t_user_start) * 1000

            st.markdown(f"<div class='latency-pill'>⚡ Live Personalized Feed Generated in {live_latency:.1f}ms</div>", unsafe_allow_html=True)
            st.markdown("### ✨ Real-Time Personalized Recommendations For You")
            
            for row_start in range(0, len(cold_recs), 4):
                cols = st.columns(4)
                for c_idx, rec in enumerate(cold_recs[row_start:row_start + 4]):
                    with cols[c_idx]:
                        render_movie_card(rec, show_match=True, show_explanation=True, key_prefix=f"live_{rec['movieId']}")


# ----------------- TAB 4: CATALOG ANALYTICS ----------------- #
elif nav_choice == "📊 Catalog Analytics":
    st.markdown("## 📊 Dataset Dynamics & Catalog Analytics")
    st.caption("Empirical data distributions: matrix sparsity, rating skew, genre frequencies, and the long-tail curve.")

    summary = dl.get_dataset_summary()

    # KPI Row
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f"""<div class="kpi-tile"><div class="kpi-value">{summary['total_movies']:,}</div><div class="kpi-label">Catalog Movies</div></div>""", unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"""<div class="kpi-tile"><div class="kpi-value">{summary['total_users']:,}</div><div class="kpi-label">Active Users</div></div>""", unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"""<div class="kpi-tile"><div class="kpi-value">{summary['total_ratings']:,}</div><div class="kpi-label">Total Ratings</div></div>""", unsafe_allow_html=True)
    with kpi4:
        st.markdown(f"""<div class="kpi-tile"><div class="kpi-value" style="color: #E50914;">{summary['sparsity_pct']}%</div><div class="kpi-label">Matrix Sparsity</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("#### Genre Distribution across Catalog")
        df_genre = dl.get_genre_distribution()
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('#0a0c14')
        ax.set_facecolor('#121624')
        sns.barplot(data=df_genre, x='count', y='genre', palette="flare", ax=ax)
        ax.set_xlabel("Number of Titles", color="#cbd5e1", fontsize=11)
        ax.set_ylabel("", color="#cbd5e1")
        ax.tick_params(colors="#94a3b8")
        ax.spines['bottom'].set_color('#334155')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#334155')
        st.pyplot(fig)

    with col_chart2:
        st.markdown("#### Rating Score Distribution (Skew Analysis)")
        df_ratings_dist = dl.get_rating_distribution()
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('#0a0c14')
        ax.set_facecolor('#121624')
        sns.barplot(data=df_ratings_dist, x='rating', y='count', color='#E50914', ax=ax)
        ax.set_xlabel("Rating Value (0.5 to 5.0)", color="#cbd5e1", fontsize=11)
        ax.set_ylabel("Frequency", color="#cbd5e1")
        ax.tick_params(colors="#94a3b8")
        ax.spines['bottom'].set_color('#334155')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#334155')
        st.pyplot(fig)

    st.markdown("<br>", unsafe_allow_html=True)

    # Long-Tail Popularity Bias Curve
    st.markdown("#### The Long-Tail Problem (Popularity Bias)")
    st.caption("A small percentage of blockbuster titles captures most interactions, necessitating hybrid filtering to surface catalog depth.")

    pop_df = dl.get_popularity_distribution()
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor('#0a0c14')
    ax.set_facecolor('#121624')
    ax.plot(pop_df.index, pop_df['rating_count'], color='#00f2fe', linewidth=2, label="Movie Rating Volume")
    ax.axvline(x=len(pop_df)*0.10, color='#E50914', linestyle='--', label="Top 10% (Short Head)")
    ax.fill_between(pop_df.index[:int(len(pop_df)*0.10)], pop_df['rating_count'][:int(len(pop_df)*0.10)], color='#E50914', alpha=0.2)
    ax.set_yscale('log')
    ax.set_xlabel("Movie Rank by Popularity", color="#cbd5e1")
    ax.set_ylabel("Ratings Count (Log Scale)", color="#cbd5e1")
    ax.tick_params(colors="#94a3b8")
    ax.legend(facecolor='#121624', edgecolor='#334155', labelcolor='#ffffff')
    ax.spines['bottom'].set_color('#334155')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#334155')
    st.pyplot(fig)


# ----------------- TAB 5: MODEL BENCHMARKS ----------------- #
elif nav_choice == "🧪 Model Benchmarks":
    st.markdown("## 🧪 Model Evaluation & Benchmark Comparison")
    st.markdown("##### Rigorous Leak-Free Evaluation on Held-Out Test Data (80% Train / 20% Test)")

    eval_data_dir = "data" if os.path.exists("data") else "src/data"
    evaluator = ModelEvaluator(data_dir=eval_data_dir)
    eval_results = evaluator.load_or_run(dl)

    meta = eval_results.get("metadata", {})
    models_data = eval_results.get("models", {})

    st.markdown(f"""
        <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 14px 20px; margin-bottom: 24px; font-size: 13px; color: #94a3b8;">
            <b>Evaluation Protocol:</b> Held-out 80/20 train-test split | Total Interactions: {meta.get('total_ratings', 0):,} | 
            Train Set: {meta.get('train_ratings', 0):,} | Held-Out Test: {meta.get('test_ratings', 0):,} | 
            Relevance Threshold: Rating ≥ {meta.get('relevance_threshold', 3.5)}
        </div>
    """, unsafe_allow_html=True)

    # Side-by-side metric comparison table
    summary_rows = []
    for m_name, m_info in models_data.items():
        summary_rows.append({
            "Algorithm": m_name,
            "Paradigm": m_info.get("type", ""),
            "RMSE (↓)": m_info.get("rmse", 0.0),
            "MAE (↓)": m_info.get("mae", 0.0),
            "Precision@5 (↑)": f"{m_info.get('precision_5', 0.0):.3f}",
            "Precision@10 (↑)": f"{m_info.get('precision_10', 0.0):.3f}",
            "Recall@10 (↑)": f"{m_info.get('recall_10', 0.0):.3f}",
            "Coverage % (↑)": f"{m_info.get('coverage_pct', 0.0)}%"
        })
    df_metrics = pd.DataFrame(summary_rows)
    st.dataframe(df_metrics, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_plot1, col_plot2 = st.columns(2)

    with col_plot1:
        st.markdown("#### Prediction Error Comparison (RMSE & MAE)")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        fig.patch.set_facecolor('#0a0c14')
        ax.set_facecolor('#121624')
        
        m_names = list(models_data.keys())
        rmse_vals = [models_data[m]['rmse'] for m in m_names]
        mae_vals = [models_data[m]['mae'] for m in m_names]
        
        x = np.arange(len(m_names))
        width = 0.35
        
        ax.bar(x - width/2, rmse_vals, width, label='RMSE', color='#E50914')
        ax.bar(x + width/2, mae_vals, width, label='MAE', color='#3b82f6')
        
        ax.set_ylabel("Error Metric (Lower is Better)", color="#cbd5e1")
        ax.set_xticks(x)
        ax.set_xticklabels([m.split()[0] for m in m_names], color="#cbd5e1", rotation=15)
        ax.tick_params(colors="#94a3b8")
        ax.legend(facecolor='#121624', edgecolor='#334155', labelcolor='#ffffff')
        ax.spines['bottom'].set_color('#334155')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#334155')
        st.pyplot(fig)

    with col_plot2:
        st.markdown("#### Ranking Precision across Top-K")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        fig.patch.set_facecolor('#0a0c14')
        ax.set_facecolor('#121624')
        
        k_levels = [5, 10, 20]
        for m_name in ["Collaborative (SVD)", "Content-Based (TF-IDF)", "CineMatch Hybrid"]:
            prec_trend = [models_data[m_name][f'precision_{k}'] for k in k_levels]
            ax.plot(k_levels, prec_trend, marker='o', linewidth=2, label=f"{m_name.split()[0]} Prec@K")

        ax.set_xlabel("Cutoff Rank (K)", color="#cbd5e1")
        ax.set_ylabel("Precision@K", color="#cbd5e1")
        ax.set_xticks(k_levels)
        ax.tick_params(colors="#94a3b8")
        ax.legend(facecolor='#121624', edgecolor='#334155', labelcolor='#ffffff')
        ax.spines['bottom'].set_color('#334155')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#334155')
        st.pyplot(fig)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
        <div style="background: rgba(16, 185, 129, 0.08); border-left: 4px solid #10b981; padding: 14px 18px; border-radius: 8px;">
            <h4 style="margin: 0 0 6px 0; color: #10b981;">Algorithmic Insights: Why Hybrid Recommendation Outperforms</h4>
            <ul style="margin: 0; padding-left: 20px; font-size: 13px; color: #cbd5e1; line-height: 1.6;">
                <li><b>Collaborative Filtering (SVD):</b> Discovers latent dimensions of taste ($k=40$), beating memory-based KNN by mitigating matrix sparsity through low-rank approximation, but is paralyzed by cold-start users.</li>
                <li><b>Content-Based Filtering (TF-IDF):</b> Highly resilient to item cold-start and provides clear explainability, but suffers from over-specialization and cannot discover cross-genre serendipity.</li>
                <li><b>CineMatch Hybrid:</b> Leverages SVD for rating accuracy when interaction history is deep, while dynamically falling back to content-based profiles when user signal is sparse.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)


# ----------------- TAB 6: MY WATCHLIST ----------------- #
elif nav_choice == "🔖 My Watchlist":
    st.markdown("## 🔖 My Watchlist & Saved Titles")
    st.caption("Locally stored collection of titles bookmarked during your session.")

    saved_mids = list(st.session_state["watchlist"])
    if not saved_mids:
        st.info("Your watchlist is currently empty. Click '+ Watchlist' on any movie card to save it here!")
    else:
        st.markdown(f"### Saved Titles ({len(saved_mids)})")
        saved_df = dl.df_movies[dl.df_movies['movieId'].isin(saved_mids)]
        
        for row_start in range(0, len(saved_df), 4):
            cols = st.columns(4)
            for c_idx, (_, row) in enumerate(saved_df.iloc[row_start:row_start + 4].iterrows()):
                with cols[c_idx]:
                    render_movie_card(row.to_dict(), show_match=False, show_explanation=False, key_prefix=f"wl_page_{row['movieId']}")


# ----------------- TAB 7: SETTINGS ----------------- #
elif nav_choice == "⚙️ Settings":
    st.markdown("## ⚙️ System Settings & API Integration")
    st.markdown("##### TMDB Credentials, Local Cache, and Model Configurations")

    st.markdown("### The Movie Database (TMDB) API Key")
    st.caption("CineMatch works 100% out-of-the-box using curated metadata and dynamic SVG posters. Providing a free TMDB API key enables live poster downloads and credits for all 9,742 catalog titles.")

    curr_key = tmdb.api_key
    new_key = st.text_input("Enter TMDB v3 API Key:", value=curr_key, type="password", placeholder="e.g. 8a2f4c9...")

    if st.button("Save & Connect API", type="primary"):
        tmdb.set_api_key(new_key)
        if tmdb.has_valid_key():
            st.success("TMDB API Key configured! Live movie posters and metadata will be queried and cached.")
        else:
            st.info("Operating in offline curated + dynamic poster mode.")
        st.rerun()

    st.markdown("---")
    st.markdown("### Cache Management")
    cache_count = len(tmdb.cache)
    st.write(f"**Cached TMDB Movie Entries on Disk:** {cache_count}")
    if st.button("Clear TMDB Cache"):
        tmdb.cache = {}
        tmdb._save_cache()
        st.success("Cache cleared successfully.")

    st.markdown("---")
    st.markdown("### Platform Architecture")
    st.markdown("""
        - **Engine:** CineMatch Hybrid AI Engine
        - **Algorithms:** Regularized SVD (Matrix Factorization), KNNWithMeans (User-Based Cosine), TF-IDF (8,000 N-gram features)
        - **Dataset:** GroupLens MovieLens latest-small (100,836 ratings, 9,742 titles, 610 users)
        - **Stack:** Python 3, Pandas, NumPy, Scikit-learn, Surprise, Streamlit, Matplotlib, Seaborn
    """)


# ==========================================
# 6. Movie Detail Modal Dialog (If clicked)
# ==========================================
if "selected_modal_movie" in st.session_state:
    modal_mid = st.session_state["selected_modal_movie"]
    m_row = dl.df_movies[dl.df_movies['movieId'] == modal_mid].iloc[0]
    m_det = tmdb.get_movie_details(m_row.get('tmdbId'), m_row['clean_title'], m_row['year'], m_row['primary_genre'])

    @st.dialog(f"🎬 {m_row['clean_title']} ({m_row['year']})")
    def show_movie_modal():
        col_m_img, col_m_desc = st.columns([1, 2])
        with col_m_img:
            st.image(m_det['poster_url'], use_container_width=True)
        with col_m_desc:
            st.markdown(f"**Genres:** {m_row['genres'].replace('|', ' • ')}")
            st.markdown(f"**Community Rating:** ★ {m_row['rating_mean']:.1f} / 5.0 ({m_row['rating_count']:,} reviews)")
            if m_det.get('director'):
                st.markdown(f"**Director:** {m_det['director']}")
            if m_det.get('cast'):
                st.markdown(f"**Starring:** {', '.join(m_det['cast'][:4])}")
            st.markdown(f"**Overview:**")
            st.write(m_det.get('overview', m_row['overview']))

            if m_row.get('tmdbId'):
                st.markdown(f"[View on TMDB](https://www.themoviedb.org/movie/{int(m_row['tmdbId'])}) ↗")

        if st.button("Close Window", use_container_width=True):
            del st.session_state["selected_modal_movie"]
            st.rerun()

    show_movie_modal()
