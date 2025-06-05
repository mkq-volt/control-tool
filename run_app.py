#!/usr/bin/env python3
"""
launcher script for depin controller wizard

run: uv run python run_app.py
"""

import subprocess
import sys


def main():
    """launch the streamlit app"""
    try:
        # run streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 shutting down...")
    except Exception as e:
        print(f"❌ error starting app: {e}")


if __name__ == "__main__":
    main() 