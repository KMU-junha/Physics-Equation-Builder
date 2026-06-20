"""Physics Equation Builder — 런처"""
import subprocess, sys, os, webbrowser, time
from pathlib import Path

BASE = Path(__file__).parent
PYTHON = sys.executable

def check_deps():
    try:
        import fastapi, uvicorn
    except ImportError:
        print("의존성 설치 중...")
        subprocess.run([PYTHON, "-m", "pip", "install", "fastapi", "uvicorn[standard]", "python-multipart"], check=True)

if __name__ == "__main__":
    check_deps()
    print("=" * 50)
    print("  Physics Equation Builder")
    print("  http://localhost:8000")
    print("=" * 50)
    time.sleep(1)
    webbrowser.open("http://localhost:8000")
    os.chdir(BASE)
    subprocess.run([PYTHON, "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"])
