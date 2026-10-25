"""
CineMatch: Single-Command Application Launcher
Runs system health checks, ensures dataset availability, and boots the Streamlit OTT app.
"""

import sys
import os
import subprocess

def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)
    print("=" * 65)
    print("🎬 CINEMATCH: Hybrid Movie Recommendation Engine")
    print("   AI-Powered Content-Based & Collaborative Filtering Platform")
    print("=" * 65)
    
    # Check dependencies
    print("\n[1/3] Verifying core libraries...")
    try:
        import streamlit
        import pandas
        import numpy
        import sklearn
        import surprise
        import matplotlib
        import seaborn
        print("  ✓ All required libraries are installed and verified.")
    except ImportError as e:
        print(f"  ✗ Missing required package: {e}")
        print("  Please run: pip install -r requirements.txt")
        sys.exit(1)

    # Check data directory
    print("\n[2/3] Checking dataset files...")
    movies_csv = os.path.join(project_dir, "data", "ml-latest-small", "movies.csv")
    ratings_csv = os.path.join(project_dir, "data", "ml-latest-small", "ratings.csv")
    if os.path.exists(movies_csv) and os.path.exists(ratings_csv):
        print("  ✓ MovieLens dataset verified locally.")
    else:
        print("  ! Dataset not detected locally. CineMatch will automatically download it on boot.")

    # Boot Streamlit
    print("\n[3/3] Launching CineMatch Streamlit Interface...")
    print("  Access the app in your browser at http://localhost:8501")
    print("=" * 65 + "\n")
    
    cmd = [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port=8501", "--server.headless=false"]
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nCineMatch stopped. Goodbye!")

if __name__ == "__main__":
    main()
