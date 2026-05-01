import subprocess
import sys
import os


def main():
    app_path = os.path.join(os.path.dirname(__file__), "streamlit_app.py")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", app_path],
        check=True,
    )


if __name__ == "__main__":
    main()
