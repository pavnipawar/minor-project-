"""
CineMatch: A Hybrid Content-Based and Collaborative Filtering Movie Recommendation System
Academic Project by Pavni Pawar (24B5044), Guided by Dr. Chandra Prakash Patidar
Department of Computer Science and Business Systems, IET DAVV Indore.
"""

import os
import json
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
# 1. Page Configuration & Theme
# ==========================================
st.set_page_config(
    page_title="CineMatch — AI Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom OTT Dark Cinematic CSS
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    /* Global styling */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    h1, h2, h3, h4, h5, .hero-title {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    /* Main background */
    .stApp {
        background-color: #0b0d13;
        color: #e2e8f0;
    }

    /* Sidebar custom styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #10131c 0%, #0c0e15 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* Hero Billboard Section */
    .hero-container {
        position: relative;
        border-radius: 20px;
        padding: 48px 40px;
        margin-bottom: 32px;
        background: linear-gradient(135deg, rgba(229, 9, 20, 0.15) 0%, rgba(18, 22, 34, 0.95) 50%, #0d1019 100%);
        border: 1px solid rgba(229, 9, 20, 0.25);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.6);
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
    }
    .hero-title {
        font-size: 42px;
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
    }
    .hero-desc {
        color: #cbd5e1;
        font-size: 15px;
        line-height: 1.6;
        max-width: 720px;
        margin-bottom: 24px;
    }

    /* Movie Poster Card Component */
    .movie-card {
        background: #141824;
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.07);
        transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
        margin-bottom: 24px;
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    .movie-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 16px 30px rgba(0, 0, 0, 0.7), 0 0 15px rgba(229, 9, 20, 0.2);
        border-color: rgba(229, 9, 20, 0.4);
    }
    .poster-img-container {
        position: relative;
        width: 100%;
        padding-top: 148%;
        background-color: #1a1f30;
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
        font-size: 16px;
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
        margin-bottom: 10px;
    }
    .rating-badge {
        color: #ffb800;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 4px;
    }
    .genre-badge {
        background: rgba(255, 255, 255, 0.08);
        color: #cbd5e1;
        padding: 2px 8px;
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
    }
    .explanation-box {
        background: rgba(255, 255, 255, 0.03);
        border-left: 3px solid #E50914;
        padding: 6px 10px;
        border-radius: 0 6px 6px 0;
        font-size: 11px;
        color: #94a3b8;
        margin-top: 10px;
        line-height: 1.4;
    }

    /* Metric KPI Tiles */
    .kpi-tile {
        background: #141824;
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 14px;
        padding: 20px;
        text-align: center;
    }
    .kpi-value {
        font-family: 'Outfit', sans-serif;
        font-size: 32px;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 4px;
    }
    .kpi-label {
        font-size: 13px;
        color: #94a3b8;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Tab navigation tweaks */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 18px;
        font-weight: 600;
        font-size: 14px;
        color: #94a3b8;
        background-color: #121624;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #E50914 0%, #b80710 100%) !important;
        color: #ffffff !important;
        border-color: #E50914 !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ==========================================
# 2. Cached Engines & Resource Loading
# ==========================================
@st.cache_resource(show_spinner="Initializing CineMatch Data Architecture...")
def get_data_loader() -> DataLoader:
    return DataLoader(data_dir="data")

@st.cache_resource(show_spinner="Training Content-Based Engine (TF-IDF & Cosine)...")
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
    return TMDBClient(cache_file="data/tmdb_cache.json")

# Initialize models
dl = get_data_loader()
cb_recommender = get_content_recommender(dl)
cf_recommender = get_collaborative_recommender(dl)
tmdb = get_tmdb_client()
hybrid_recommender = HybridRecommender(cb_recommender, cf_recommender)

# Session state initialization for Watchlist & User Ratings
if "watchlist" not in st.session_state:
    st.session_state["watchlist"] = set([318, 2571])  # Default: Shawshank & Matrix
if "custom_ratings" not in st.session_state:
    st.session_state["custom_ratings"] = {}

# ==========================================
# 3. Sidebar Navigation & Global Controls
# ==========================================
with st.sidebar:
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 18px;">
            <div style="background: #E50914; width: 36px; height: 36px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 20px;">🎬</div>
            <div>
                <h2 style="margin: 0; font-size: 20px; color: #fff;">CineMatch</h2>
                <span style="font-size: 11px; color: #94a3b8; font-weight: 500;">Hybrid AI Engine v2.0</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    nav_choice = st.radio(
        "Navigation Menu",
        [
            "🎬 Home & Billboard",
            "🎯 Discover by Movie (Content-Based)",
            "👤 Personalize for Me (Hybrid & Cold-Start)",
            "📊 Dataset EDA & Analytics",
            "🧪 Model Evaluation & Viva Benchmarks",
            "🔖 My Watchlist & Ratings",
            "⚙️ Settings & TMDB API"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    
    # Active user profile info
    st.markdown("<h4 style='font-size: 13px; color: #94a3b8; text-transform: uppercase;'>Academic Info</h4>", unsafe_allow_html=True)
    st.markdown("""
        <div style='background: rgba(255,255,255,0.03); padding: 12px; border-radius: 10px; font-size: 12px; line-height: 1.5; border: 1px solid rgba(255,255,255,0.05);'>
            <b>Scholar:</b> Pavni Pawar (24B5044)<br>
            <b>Guide:</b> Dr. C.P. Patidar<br>
            <b>Dept:</b> CSBS, IET DAVV Indore<br>
            <b>Session:</b> 2025 – 2026
        </div>
    """, unsafe_allow_html=True)

    # TMDB status badge
    st.markdown("<br>", unsafe_allow_html=True)
    has_tmdb = tmdb.has_valid_key()
    status_color = "#10b981" if has_tmdb else "#f59e0b"
    status_text = "TMDB Live Connected" if has_tmdb else "Curated & Dynamic SVG Mode"
    st.markdown(f"""
        <div style='display: flex; align-items: center; gap: 8px; font-size: 11px; color: {status_color};'>
            <span style='width: 8px; height: 8px; border-radius: 50%; background-color: {status_color}; display: inline-block;'></span>
            <span>{status_text}</span>
        </div>
    """, unsafe_allow_html=True)


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
                <span class="rating-badge">★ {rating_mean:.1f} <span style="font-size: 10px; color: #64748b;">({rating_cnt})</span></span>
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

# ----------------- TAB 1: HOME & BILLBOARD ----------------- #
if nav_choice == "🎬 Home & Explore":
    # Hero Movie: Inception (79132) or Shawshank (318)
    hero_mid = 79132 if 79132 in dl.df_movies['movieId'].values else 318
    hero_movie = dl.df_movies[dl.df_movies['movieId'] == hero_mid].iloc[0]
    hero_details = tmdb.get_movie_details(hero_movie.get('tmdbId'), hero_movie['clean_title'], hero_movie['year'], hero_movie['primary_genre'])

    st.markdown(f"""
        <div class="hero-container">
            <span class="hero-badge">Featured Cinematic Masterpiece</span>
            <div class="hero-title">{hero_movie['clean_title']} ({hero_movie['year']})</div>
            <div class="hero-meta">
                <span style="color: #ffb800; font-weight: 700;">★ {hero_movie['rating_mean']:.1f} Rating</span>
                <span>•</span>
                <span>{hero_movie['genres'].replace('|', ' • ')}</span>
                <span>•</span>
                <span>Dir: {hero_details.get('director', 'Christopher Nolan')}</span>
            </div>
            <div class="hero-desc">{hero_details.get('overview', hero_movie['overview'])}</div>
        </div>
    """, unsafe_allow_html=True)

    col_btn1, col_btn2, _ = st.columns([2, 2, 4])
    with col_btn1:
        if st.button("🎯 Discover Similar Titles", key="hero_similar_btn", type="primary", use_container_width=True):
            st.session_state["seed_movie_id"] = int(hero_mid)
            st.session_state["navigate_to_discover"] = True
            st.rerun()
    with col_btn2:
        if st.button("★ Add to Watchlist", key="hero_watchlist_btn", use_container_width=True):
            st.session_state["watchlist"].add(int(hero_mid))
            st.toast(f"Added {hero_movie['clean_title']} to your Watchlist!", icon="🎬")

    st.markdown("<br>", unsafe_allow_html=True)

    # Section 1: Trending & Most Popular
    st.markdown("### 🔥 Trending & Popular on CineMatch")
    st.caption("Most frequently rated and discussed titles across the entire community catalog")
    trending_movies = dl.get_trending_movies(top_n=5)
    cols = st.columns(5)
    for i, (_, row) in enumerate(trending_movies.iterrows()):
        with cols[i]:
            render_movie_card(row.to_dict(), show_match=False, key_prefix="trend")

    st.markdown("<br>", unsafe_allow_html=True)

    # Section 2: Top Rated All-Time Masterpieces (Bayesian Weighted)
    st.markdown("### ⭐ Top Rated Masterpieces (Bayesian Weighted)")
    st.caption("Statistically normalized using IMDB/Bayesian weighted formula $WR = \\frac{v}{v+m}R + \\frac{m}{v+m}C$ to eliminate small-sample bias")
    top_rated = dl.get_top_rated_movies(min_ratings=50, top_n=5)
    cols = st.columns(5)
    for i, (_, row) in enumerate(top_rated.iterrows()):
        with cols[i]:
            render_movie_card(row.to_dict(), show_match=False, key_prefix="top")

    st.markdown("<br>", unsafe_allow_html=True)

    # Section 3: Explore by Genre
    st.markdown("### 🎭 Explore by Genre")
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


# ----------------- TAB 2: DISCOVER BY MOVIE (CONTENT-BASED) ----------------- #
elif nav_choice == "🎯 Discover by Movie (Content-Based)":
    st.markdown("## 🎯 Discover by Movie")
    st.markdown("##### Thematic Content-Based Filtering using TF-IDF & Cosine Similarity")
    st.caption("Computes high-dimensional feature cosine distance over genres, plot overview, director, cast, and user keywords.")

    # Search / Select Movie
    movie_options = dl.df_movies[['movieId', 'clean_title', 'year', 'rating_mean']].sort_values(by='clean_title')
    movie_dict = {f"{row.clean_title} ({row.year}) — ★ {row.rating_mean:.1f}": row.movieId for _, row in movie_options.iterrows()}
    
    default_index = 0
    if "seed_movie_id" in st.session_state:
        for idx, (label, mid) in enumerate(movie_dict.items()):
            if mid == st.session_state["seed_movie_id"]:
                default_index = idx
                break

    selected_label = st.selectbox("Select or Search Seed Movie:", list(movie_dict.keys()), index=default_index)
    seed_mid = movie_dict[selected_label]

    # Display Seed Movie Profile
    seed_movie = dl.df_movies[dl.df_movies['movieId'] == seed_mid].iloc[0]
    seed_details = tmdb.get_movie_details(seed_movie.get('tmdbId'), seed_movie['clean_title'], seed_movie['year'], seed_movie['primary_genre'])

    col_seed_img, col_seed_info = st.columns([1, 3])
    with col_seed_img:
        st.markdown(f"""
            <div style="border-radius: 12px; overflow: hidden; border: 1px solid rgba(255,255,255,0.1);">
                <img src="{seed_details['poster_url']}" style="width: 100%; display: block;" />
            </div>
        """, unsafe_allow_html=True)
    with col_seed_info:
        st.markdown(f"### {seed_movie['clean_title']} ({seed_movie['year']})")
        st.markdown(f"**Genres:** {seed_movie['genres'].replace('|', ', ')} | **Rating:** ★ {seed_movie['rating_mean']:.1f} ({seed_movie['rating_count']} reviews)")
        if seed_details.get("director"):
            st.markdown(f"**Director:** {seed_details['director']} | **Cast:** {', '.join(seed_details.get('cast', [])[:4])}")
        st.markdown(f"**Plot Synopsis:** {seed_details.get('overview', seed_movie['overview'])}")
        
        # Keyword tags
        if seed_movie['user_tags']:
            st.markdown(f"**Community Tags:** `{seed_movie['user_tags'][:120]}...`")

    st.markdown("---")
    
    col_ctrl1, col_ctrl2 = st.columns([2, 4])
    with col_ctrl1:
        top_k = st.slider("Number of Similar Recommendations:", min_value=4, max_value=20, value=8, step=2)

    st.markdown(f"### Top {top_k} Movies Similar to *{seed_movie['clean_title']}*")
    similar_recs = cb_recommender.get_similar_movies(seed_mid, top_n=top_k)

    if not similar_recs:
        st.warning("No similar movies found above the similarity threshold.")
    else:
        # Display in rows of 4
        for row_start in range(0, len(similar_recs), 4):
            cols = st.columns(4)
            for c_idx, rec in enumerate(similar_recs[row_start:row_start + 4]):
                with cols[c_idx]:
                    render_movie_card(rec, show_match=True, show_explanation=True, key_prefix=f"cb_{seed_mid}_{rec['movieId']}")


# ----------------- TAB 3: PERSONALIZE FOR ME (HYBRID & COLD-START) ----------------- #
elif nav_choice == "👤 Personalize for Me (Hybrid & Cold-Start)":
    st.markdown("## 👤 Personalized Recommendations")
    st.markdown("##### CineMatch Adaptive Hybrid Engine (SVD Collaborative Filtering + TF-IDF Content Profile)")

    mode = st.radio(
        "Choose User Persona:",
        ["Existing Community User (from MovieLens Dataset)", "✨ New User Onboarding (Interactive Cold-Start Mode)"],
        horizontal=True
    )

    if mode == "Existing Community User (from MovieLens Dataset)":
        col_u1, col_u2, col_u3 = st.columns([2, 2, 2])
        with col_u1:
            # Let user select from users with varied activity
            popular_users = [1, 10, 15, 68, 111, 249, 414, 474, 599, 610]
            selected_uid = st.selectbox("Select User ID:", popular_users, index=0)

        with col_u2:
            alpha_val = st.slider("Hybrid Weight (α CF vs 1-α CB):", min_value=0.0, max_value=1.0, value=0.60, step=0.05,
                                  help="1.0 = Pure Collaborative SVD, 0.0 = Pure Content-Based Profile, 0.6 = CineMatch Optimal Hybrid")

        with col_u3:
            num_recs = st.slider("Recommendations Count:", min_value=4, max_value=16, value=8, step=2)

        # Retrieve user history
        user_ratings_df = dl.df_ratings[dl.df_ratings['userId'] == selected_uid].sort_values(by='rating', ascending=False)
        user_ratings_dict = dict(zip(user_ratings_df['movieId'], user_ratings_df['rating']))

        # Inspect user profile expander
        with st.expander(f"🔍 Inspect User {selected_uid}'s Taste Profile ({len(user_ratings_df)} rated movies)", expanded=False):
            st.markdown(f"**Average Rating Given:** {user_ratings_df['rating'].mean():.2f} / 5.0")
            top_rated_by_user = pd.merge(user_ratings_df.head(6), dl.df_movies, on='movieId')
            st.dataframe(top_rated_by_user[['clean_title', 'year', 'genres', 'rating']], use_container_width=True)

        # Generate Hybrid Recommendations
        hybrid_recs = hybrid_recommender.recommend_for_user(
            user_id=selected_uid,
            user_ratings_dict=user_ratings_dict,
            catalog_movies=dl.df_movies,
            top_n=num_recs,
            alpha=alpha_val
        )

        st.markdown(f"### Curated Hybrid Recommendations for User #{selected_uid}")
        st.caption(f"Blended with α = {alpha_val*100:.0f}% SVD Latent Taste + {(1-alpha_val)*100:.0f}% Content Thematic Profile")

        for row_start in range(0, len(hybrid_recs), 4):
            cols = st.columns(4)
            for c_idx, rec in enumerate(hybrid_recs[row_start:row_start + 4]):
                with cols[c_idx]:
                    render_movie_card(rec, show_match=True, show_explanation=True, key_prefix=f"hyb_{selected_uid}_{rec['movieId']}")

    else:
        # COLD-START INTERACTIVE ONBOARDING
        st.markdown("""
            <div style="background: rgba(229, 9, 20, 0.08); border-left: 4px solid #E50914; padding: 14px 18px; border-radius: 8px; margin-bottom: 20px;">
                <h4 style="margin: 0 0 6px 0; color: #fff;">Cold-Start Recommendation Solver</h4>
                <p style="margin: 0; font-size: 13px; color: #cbd5e1;">
                    Collaborative filtering cannot recommend movies to a new user with zero ratings (the Cold-Start Problem).
                    CineMatch solves this by capturing your seed taste profile and dynamically falling back to Content-Based filtering!
                </p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Step 1: Select Your Favorite Genres")
        all_genres = [g for g in dl.get_genre_distribution()['genre'].tolist() if g != "(no genres listed)"]
        selected_fav_genres = st.multiselect("Pick 2 or more genres you enjoy:", all_genres, default=["Action", "Sci-Fi"])

        st.markdown("#### Step 2: Rate A Few Popular Movies (0.5 to 5.0 Stars)")
        st.caption("Rate at least 2 movies to bootstrap your personalized taste vector:")

        candidate_seeds = [
            (318, "The Shawshank Redemption", 1994, "Drama"),
            (2571, "The Matrix", 1999, "Sci-Fi"),
            (79132, "Inception", 2010, "Sci-Fi"),
            (356, "Forrest Gump", 1994, "Drama"),
            (296, "Pulp Fiction", 1994, "Crime"),
            (58559, "The Dark Knight", 2008, "Action"),
            (1, "Toy Story", 1995, "Animation"),
            (480, "Jurassic Park", 1993, "Adventure")
        ]

        seed_cols = st.columns(4)
        for i, (mid, title, year, gen) in enumerate(candidate_seeds):
            with seed_cols[i % 4]:
                current_val = st.session_state["custom_ratings"].get(mid, 0.0)
                user_star = st.slider(f"{title} ({year})", min_value=0.0, max_value=5.0, value=float(current_val), step=0.5, key=f"seed_slider_{mid}")
                if user_star > 0:
                    st.session_state["custom_ratings"][mid] = user_star
                elif mid in st.session_state["custom_ratings"] and user_star == 0.0:
                    del st.session_state["custom_ratings"][mid]

        user_rated_count = len(st.session_state["custom_ratings"])
        st.info(f"You have rated **{user_rated_count}** seed titles. Effective Collaborative Alpha: **{hybrid_recommender.calculate_effective_alpha(user_rated_count):.2f}**")

        if user_rated_count == 0:
            st.warning("Please rate at least 1 movie above to generate your custom cold-start recommendations.")
        else:
            cold_recs = hybrid_recommender.recommend_for_user(
                user_id=9999,  # Virtual new user ID
                user_ratings_dict=st.session_state["custom_ratings"],
                catalog_movies=dl.df_movies,
                top_n=8
            )

            st.markdown("### ✨ Your Cold-Start Personalized Recommendations")
            for row_start in range(0, len(cold_recs), 4):
                cols = st.columns(4)
                for c_idx, rec in enumerate(cold_recs[row_start:row_start + 4]):
                    with cols[c_idx]:
                        render_movie_card(rec, show_match=True, show_explanation=True, key_prefix=f"cold_{rec['movieId']}")


# ----------------- TAB 4: DATASET EDA & ANALYTICS ----------------- #
elif nav_choice == "📊 Dataset EDA & Analytics":
    st.markdown("## 📊 Exploratory Data Analysis & Catalog Dynamics")
    st.caption("Empirical justification of design decisions: sparsity, genre skews, long-tail popularity curve, and rating distributions.")

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

    # Visualization Row 1: Genre Distribution & Rating Value Distribution
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("#### Genre Distribution across Catalog")
        df_genre = dl.get_genre_distribution()
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('#0d0f17')
        ax.set_facecolor('#141824')
        sns.barplot(data=df_genre, x='count', y='genre', palette="flare", ax=ax)
        ax.set_xlabel("Number of Titles", color="#cbd5e1", fontsize=11)
        ax.set_ylabel("", color="#cbd5e1")
        ax.tick_params(colors="#94a3b8")
        ax.spines['bottom'].set_color('#334155')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#334155')
        st.pyplot(fig)
        st.caption("Drama, Comedy, and Action heavily dominate the catalog, while Film-Noir and Western are sparse.")

    with col_chart2:
        st.markdown("#### User Rating Distribution (Skew Analysis)")
        df_ratings_dist = dl.get_rating_distribution()
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('#0d0f17')
        ax.set_facecolor('#141824')
        sns.barplot(data=df_ratings_dist, x='rating', y='count', color='#E50914', ax=ax)
        ax.set_xlabel("Rating Score (0.5 to 5.0)", color="#cbd5e1", fontsize=11)
        ax.set_ylabel("Frequency", color="#cbd5e1")
        ax.tick_params(colors="#94a3b8")
        ax.spines['bottom'].set_color('#334155')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#334155')
        st.pyplot(fig)
        st.caption("Distribution is strongly skewed toward 3.0 – 4.0 stars. Users rarely rate below 2.0.")

    st.markdown("<br>", unsafe_allow_html=True)

    # Visualization Row 2: Long-Tail Popularity Curve
    st.markdown("#### The Long-Tail Problem (Popularity Bias)")
    st.caption("A tiny fraction of blockbusters captures the vast majority of interactions, leading purely popularity-based recommenders to become repetitive.")

    pop_df = dl.get_popularity_distribution()
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor('#0d0f17')
    ax.set_facecolor('#141824')
    ax.plot(pop_df.index, pop_df['rating_count'], color='#00f2fe', linewidth=2, label="Movie Rating Volume")
    ax.axvline(x=len(pop_df)*0.10, color='#E50914', linestyle='--', label="Top 10% (Short Head)")
    ax.fill_between(pop_df.index[:int(len(pop_df)*0.10)], pop_df['rating_count'][:int(len(pop_df)*0.10)], color='#E50914', alpha=0.2)
    ax.set_yscale('log')
    ax.set_xlabel("Movie Rank (by Popularity)", color="#cbd5e1")
    ax.set_ylabel("Rating Count (Log Scale)", color="#cbd5e1")
    ax.tick_params(colors="#94a3b8")
    ax.legend(facecolor='#141824', edgecolor='#334155', labelcolor='#ffffff')
    ax.spines['bottom'].set_color('#334155')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#334155')
    st.pyplot(fig)


# ----------------- TAB 5: MODEL EVALUATION & BENCHMARKS ----------------- #
elif nav_choice == "🧪 Model Evaluation & Viva Benchmarks":
    st.markdown("## 🧪 Model Evaluation & Benchmark Comparison")
    st.markdown("##### Rigorous Held-Out Test Evaluation (80% Train / 20% Test Split without Data Leakage)")

    evaluator = ModelEvaluator(data_dir="data")
    eval_results = evaluator.load_or_run(dl)

    meta = eval_results.get("metadata", {})
    models_data = eval_results.get("models", {})

    st.markdown(f"""
        <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 14px 20px; margin-bottom: 24px; font-size: 13px; color: #94a3b8;">
            <b>Evaluation Protocol:</b> Held-out 80/20 train-test split | Total Interactions: {meta.get('total_ratings', 0):,} | 
            Train: {meta.get('train_ratings', 0):,} | Held-out Test: {meta.get('test_ratings', 0):,} | 
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

    # Comparative Bar Charts
    col_plot1, col_plot2 = st.columns(2)

    with col_plot1:
        st.markdown("#### Prediction Error Comparison (RMSE & MAE)")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        fig.patch.set_facecolor('#0d0f17')
        ax.set_facecolor('#141824')
        
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
        ax.legend(facecolor='#141824', edgecolor='#334155', labelcolor='#ffffff')
        ax.spines['bottom'].set_color('#334155')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#334155')
        st.pyplot(fig)
        st.caption("SVD and Hybrid achieve lowest RMSE and MAE compared to pure Content-Based and KNN.")

    with col_plot2:
        st.markdown("#### Ranking Precision & Recall across Top-K")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        fig.patch.set_facecolor('#0d0f17')
        ax.set_facecolor('#141824')
        
        k_levels = [5, 10, 20]
        for m_name in ["Collaborative (SVD)", "Content-Based (TF-IDF)", "CineMatch Hybrid"]:
            prec_trend = [models_data[m_name][f'precision_{k}'] for k in k_levels]
            ax.plot(k_levels, prec_trend, marker='o', linewidth=2, label=f"{m_name.split()[0]} Prec@K")

        ax.set_xlabel("Cutoff Rank (K)", color="#cbd5e1")
        ax.set_ylabel("Precision@K", color="#cbd5e1")
        ax.set_xticks(k_levels)
        ax.tick_params(colors="#94a3b8")
        ax.legend(facecolor='#141824', edgecolor='#334155', labelcolor='#ffffff')
        ax.spines['bottom'].set_color('#334155')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#334155')
        st.pyplot(fig)
        st.caption("CineMatch Hybrid retains superior precision at larger K cutoffs while maintaining strong catalog diversity.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
        <div style="background: rgba(16, 185, 129, 0.08); border-left: 4px solid #10b981; padding: 14px 18px; border-radius: 8px;">
            <h4 style="margin: 0 0 6px 0; color: #10b981;">Academic Viva Insights: Why Hybrid Wins</h4>
            <ul style="margin: 0; padding-left: 20px; font-size: 13px; color: #cbd5e1; line-height: 1.6;">
                <li><b>Collaborative Filtering (SVD):</b> Discovers latent dimensions of taste ($k=40$), beating KNN by handling matrix sparsity through low-rank approximation, but is paralyzed by cold-start users.</li>
                <li><b>Content-Based Filtering (TF-IDF):</b> Excellent for item cold-start and explainability, but suffers from over-specialization and cannot discover cross-genre taste serendipity.</li>
                <li><b>CineMatch Hybrid:</b> Leverages SVD for rating accuracy when interaction history is deep, while dynamically falling back to content-based profiles when user signal is sparse.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)


# ----------------- TAB 6: MY WATCHLIST & RATINGS ----------------- #
elif nav_choice == "🔖 My Watchlist & Ratings":
    st.markdown("## 🔖 My Watchlist & Rated Titles")
    st.caption("Persistent session storage of movies you saved or scored.")

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


# ----------------- TAB 7: SETTINGS & TMDB API ----------------- #
elif nav_choice == "⚙️ Settings & TMDB API":
    st.markdown("## ⚙️ System Settings & External API Integration")
    st.markdown("##### Manage TMDB Credentials, Local Cache, and Model Parameters")

    st.markdown("### The Movie Database (TMDB) API Key")
    st.caption("CineMatch works 100% out-of-the-box using curated metadata and dynamic SVG posters. Providing a free TMDB API key enables live poster downloads and credits for all 9,742 catalog titles.")

    curr_key = tmdb.api_key
    new_key = st.text_input("Enter TMDB v3 API Key:", value=curr_key, type="password", placeholder="e.g. 8a2f4c9...")

    if st.button("Save & Test Connection", type="primary"):
        tmdb.set_api_key(new_key)
        if tmdb.has_valid_key():
            st.success("TMDB API Key configured! Live movie posters and metadata will be queried and cached.")
        else:
            st.info("API Key cleared or empty. Operating in offline curated + dynamic poster mode.")
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
    st.markdown("### Project & College Viva Credits")
    st.markdown("""
        - **Project Title:** CineMatch: A Hybrid Content-Based and Collaborative Filtering Approach to Movie Recommendation
        - **Author:** Pavni Pawar (Enrollment: 24B5044)
        - **Supervisor:** Dr. Chandra Prakash Patidar
        - **Institution:** Department of Computer Science and Business Systems, Institute of Engineering & Technology, Devi Ahilya Vishwavidyalaya, Indore (M.P.)
        - **Session:** 2025 – 2026
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
            st.markdown(f"**Community Rating:** ★ {m_row['rating_mean']:.1f} / 5.0 ({m_row['rating_count']} reviews)")
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
