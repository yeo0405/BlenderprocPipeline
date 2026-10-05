#!/usr/bin/env python3
"""
Startup script for the 6D Detection UI system
Starts both backend and frontend servers
"""

import subprocess
import sys
import time
import signal
from pathlib import Path


print("current python:", sys.executable)

UI_DIR = Path(__file__).parent.absolute()
BACKEND_DIR = UI_DIR / "backend"
FRONTEND_DIR = UI_DIR / "frontend"

BACKEND_PORT = 5000
FRONTEND_PORT = 8000
FRONTEND_HOST = "0.0.0.0"

print("=" * 70)
print("6D Detection Data Generation UI - Startup")
print("=" * 70)

print("\n[*] Checking required files...")
backend_app = BACKEND_DIR / "app.py"
frontend_index = FRONTEND_DIR / "index.html"

if not backend_app.exists():
    print(f"[✗] Backend app not found: {backend_app}")
    sys.exit(1)

if not frontend_index.exists():
    print(f"[✗] Frontend index not found: {frontend_index}")
    sys.exit(1)

print(f"[✓] Backend app found: {backend_app}")
print(f"[✓] Frontend index found: {frontend_index}")

print("\n[*] Checking Python dependencies...")
try:
    import flask
    print("[✓] Flask is installed")
except ImportError:
    print("[✗] Flask is not installed. Install with: pip install flask flask-cors")
    sys.exit(1)

try:
    import flask_cors
    print("[✓] Flask-CORS is installed")
except ImportError:
    print("[✗] Flask-CORS is not installed. Install with: pip install flask-cors")
    sys.exit(1)

try:
    import numpy
    print("[✓] NumPy is installed")
except ImportError:
    print("[✗] NumPy is not installed. Install with: pip install numpy")
    sys.exit(1)

print(f"\n[*] Starting backend server on port {BACKEND_PORT}...")
print(f"    Command: python3 {backend_app}")

backend_process = subprocess.Popen(
    [sys.executable, str(backend_app)],
    cwd=str(BACKEND_DIR),
    stdout=None,
    stderr=None,
    text=True,
)

print("[*] Waiting for backend to start...")
time.sleep(2)

if backend_process.poll() is not None:
    print("[✗] Backend failed to start!")
    print(f"Backend exited with code: {backend_process.returncode}")
    sys.exit(1)

print("[✓] Backend server started successfully!")

print(f"\n[*] Starting frontend server on port {FRONTEND_PORT}...")

try:
    frontend_process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "http.server",
            str(FRONTEND_PORT),
            "--directory",
            str(FRONTEND_DIR),
            "--bind",
            FRONTEND_HOST,
        ],
        stdout=None,
        stderr=None,
        text=True,
    )

    time.sleep(1)

    if frontend_process.poll() is not None:
        print("[✗] Frontend failed to start!")
        print(f"Frontend exited with code: {frontend_process.returncode}")
        backend_process.terminate()
        sys.exit(1)

    print("[✓] Frontend server started successfully!")

except Exception as e:
    print(f"[✗] Failed to start frontend: {e}")
    backend_process.terminate()
    sys.exit(1)

print("\n" + "=" * 70)
print("✓ All services started successfully!")
print("=" * 70)
print(f"\n📱 Web UI:    http://localhost:{FRONTEND_PORT}")
print(f"⚙️  Backend:   http://localhost:{BACKEND_PORT}")
print(f"\n📂 Backend directory:  {BACKEND_DIR}")
print(f"📂 Frontend directory: {FRONTEND_DIR}")
print("\n[*] Press Ctrl+C to stop all services...")
print("=" * 70 + "\n")


def signal_handler(signum, frame):
    print("\n\n[*] Shutting down services...")

    try:
        print("[*] Stopping backend server...")
        backend_process.terminate()
        backend_process.wait(timeout=5)
        print("[✓] Backend stopped")
    except Exception as e:
        print(f"[!] Error stopping backend: {e}")
        backend_process.kill()

    try:
        print("[*] Stopping frontend server...")
        frontend_process.terminate()
        frontend_process.wait(timeout=5)
        print("[✓] Frontend stopped")
    except Exception as e:
        print(f"[!] Error stopping frontend: {e}")
        frontend_process.kill()

    print("\n[✓] All services stopped. Goodbye!")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

try:
    while True:
        backend_status = backend_process.poll()
        frontend_status = frontend_process.poll()

        if backend_status is not None:
            print("[✗] Backend server crashed! Exiting...")
            frontend_process.terminate()
            sys.exit(1)

        if frontend_status is not None:
            print("[✗] Frontend server crashed! Exiting...")
            backend_process.terminate()
            sys.exit(1)

        time.sleep(1)

except KeyboardInterrupt:
    signal_handler(None, None)
