"""
CineMatch: TMDB Client and Metadata Enricher
Handles live API queries to The Movie Database (TMDB), persistent local caching,
and seamless fallback to curated metadata and high-fidelity dynamic SVG posters.
"""

import os
import json
import base64
import requests
from typing import Optional, Dict, Any
from .curated_data import CURATED_TMDB_DATA, generate_svg_poster_card

TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p"

class TMDBClient:
    """
    Robust TMDB client with local caching and offline fallback capabilities.
    """
    def __init__(self, api_key: Optional[str] = None, cache_file: str = "data/tmdb_cache.json"):
        # Check explicit key, or environment variable TMDB_API_KEY
        self.api_key = api_key or os.getenv("TMDB_API_KEY", "").strip()
        self.cache_file = cache_file
        self.cache: Dict[str, Dict[str, Any]] = {}
        self._load_cache()

    def _load_cache(self):
        """Loads cached metadata from disk."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except Exception:
                self.cache = {}

    def _save_cache(self):
        """Persists cached metadata to disk."""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2)
        except Exception:
            pass

    def set_api_key(self, api_key: str):
        """Updates the API key dynamically from Streamlit UI."""
        self.api_key = api_key.strip()

    def has_valid_key(self) -> bool:
        """Checks if a non-empty API key is configured."""
        return bool(self.api_key and len(self.api_key) >= 15)

    def get_movie_details(self, tmdb_id: Optional[float], title: str = "", year: str = "", primary_genre: str = "Default") -> Dict[str, Any]:
        """
        Retrieves movie metadata including overview, director, cast, and posters.
        Order of resolution:
        1. Local disk cache
        2. Curated built-in metadata (top popular movies)
        3. TMDB Live API (if API key present)
        4. Synthesized fallback with dynamic SVG poster
        """
        clean_tmdb_id = int(tmdb_id) if tmdb_id and not (isinstance(tmdb_id, float) and tmdb_id != tmdb_id) else None
        cache_key = str(clean_tmdb_id) if clean_tmdb_id else f"title_{title}_{year}"

        # 1. Return from disk cache if present
        if cache_key in self.cache:
            return self.cache[cache_key]

        # 2. Check built-in curated data
        if clean_tmdb_id and clean_tmdb_id in CURATED_TMDB_DATA:
            curated = CURATED_TMDB_DATA[clean_tmdb_id].copy()
            curated["title"] = title
            curated["year"] = year
            curated["tmdb_id"] = clean_tmdb_id
            if curated.get("poster_path"):
                curated["poster_url"] = f"{TMDB_IMAGE_BASE}/w500{curated['poster_path']}"
            if curated.get("backdrop_path"):
                curated["backdrop_url"] = f"{TMDB_IMAGE_BASE}/w1280{curated['backdrop_path']}"
            
            self.cache[cache_key] = curated
            self._save_cache()
            return curated

        # 3. Live TMDB API call if API key is configured
        if self.has_valid_key() and clean_tmdb_id:
            try:
                url = f"{TMDB_BASE_URL}/movie/{clean_tmdb_id}"
                params = {
                    "api_key": self.api_key,
                    "append_to_response": "credits,keywords"
                }
                # verify=False handles Windows local issuer certificate verification issues gracefully
                resp = requests.get(url, params=params, timeout=5, verify=False)
                if resp.status_code == 200:
                    data = resp.json()
                    
                    # Extract credits
                    director = ""
                    cast_list = []
                    credits = data.get("credits", {})
                    for crew in credits.get("crew", []):
                        if crew.get("job") == "Director":
                            director = crew.get("name", "")
                            break
                    for cast_member in credits.get("cast", [])[:5]:
                        cast_list.append(cast_member.get("name", ""))

                    # Extract keywords
                    keywords_raw = data.get("keywords", {}).get("keywords", [])
                    keywords = [k.get("name", "") for k in keywords_raw]

                    poster_path = data.get("poster_path")
                    backdrop_path = data.get("backdrop_path")

                    details = {
                        "tmdb_id": clean_tmdb_id,
                        "title": data.get("title", title),
                        "year": data.get("release_date", "")[:4] if data.get("release_date") else year,
                        "overview": data.get("overview", "") or f"{title} ({year}) is an acclaimed {primary_genre} film.",
                        "director": director or "Unknown",
                        "cast": cast_list,
                        "keywords": keywords,
                        "poster_path": poster_path,
                        "backdrop_path": backdrop_path,
                        "poster_url": f"{TMDB_IMAGE_BASE}/w500{poster_path}" if poster_path else None,
                        "backdrop_url": f"{TMDB_IMAGE_BASE}/w1280{backdrop_path}" if backdrop_path else None,
                        "vote_average": data.get("vote_average", 0.0),
                        "vote_count": data.get("vote_count", 0)
                    }

                    # If no poster image from API, generate dynamic SVG poster
                    if not details["poster_url"]:
                        svg = generate_svg_poster_card(title, year or "Movie", primary_genre)
                        encoded = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
                        details["poster_url"] = f"data:image/svg+xml;base64,{encoded}"

                    self.cache[cache_key] = details
                    self._save_cache()
                    return details
            except Exception:
                pass  # Fall through to fallback synthesis

        # 4. Fallback synthesis
        svg = generate_svg_poster_card(title, year or "Movie", primary_genre)
        encoded = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
        fallback_details = {
            "tmdb_id": clean_tmdb_id,
            "title": title,
            "year": year,
            "overview": f"{title} ({year}) is a notable {primary_genre} feature film in the MovieLens catalog.",
            "director": "Acclaimed Filmmaker",
            "cast": ["Featured Ensemble", "Lead Actor"],
            "keywords": [primary_genre.lower(), "cinema", "movielens"],
            "poster_path": None,
            "backdrop_path": None,
            "poster_url": f"data:image/svg+xml;base64,{encoded}",
            "backdrop_url": None,
            "vote_average": 0.0,
            "vote_count": 0
        }
        self.cache[cache_key] = fallback_details
        self._save_cache()
        return fallback_details
