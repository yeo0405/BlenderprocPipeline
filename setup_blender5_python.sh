#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BLENDER_DIR="${BLENDER_DIR:-$BASE_DIR/../blender-5.2.1-linux-x64}"
BLENDER_PYTHON="$BLENDER_DIR/5.2/python/bin/python3.13"

BOP_TOOLKIT_DIR="${BOP_TOOLKIT_DIR:-$BASE_DIR/../bop_toolkit}"

echo "========================================"
echo " Blender 5.x Python Environment Setup"
echo "========================================"
echo
echo "[BASE]          $BASE_DIR"
echo "[BLENDER]       $BLENDER_DIR"
echo "[BLENDER PYTHON] $BLENDER_PYTHON"
echo "[BOP TOOLKIT]   $BOP_TOOLKIT_DIR"
echo

if [ ! -x "$BLENDER_PYTHON" ]; then
    echo "[ERROR] Blender Python not found:"
    echo "        $BLENDER_PYTHON"
    echo
    echo "Set BLENDER_DIR manually, for example:"
    echo "  export BLENDER_DIR=/home/yeo/Downloads/blender-5.2.1-linux-x64"
    exit 1
fi

if [ ! -d "$BOP_TOOLKIT_DIR/bop_toolkit_lib" ]; then
    echo "[INFO] BOP Toolkit not found."

    if command -v git >/dev/null 2>&1; then
        echo "[INFO] Cloning BOP Toolkit..."
        git clone \
            https://github.com/thodan/bop_toolkit.git \
            "$BOP_TOOLKIT_DIR"
    else
        echo "[ERROR] git is not installed."
        exit 1
    fi
fi

echo
echo "[1/5] Blender Python version"
"$BLENDER_PYTHON" --version

echo
echo "[2/5] Checking pip"

if ! "$BLENDER_PYTHON" -m pip --version >/dev/null 2>&1; then
    echo "[INFO] Installing pip..."
    curl -sS https://bootstrap.pypa.io/get-pip.py | "$BLENDER_PYTHON"
fi

"$BLENDER_PYTHON" -m pip install --upgrade pip setuptools wheel

echo
echo "[3/5] Installing Blender 5.x Python dependencies"

"$BLENDER_PYTHON" -m pip install \
    "numpy<2.0" \
    "Pillow>=8.2" \
    "pypng>=0.20220715" \
    "pytz>=2025.2" \
    "webdataset>=0.2.100" \
    "PyOpenGL>=3.1.0" \
    "imageio>=2.35.1" \
    "scikit-image>=0.21.0" \
    "scipy>=1.10.1" \
    "vispy>=0.14.2" \
    "opencv-python>=4.11.0.86" \
    "matplotlib>=3.7.5" \
    "tqdm>=4.67.1"

echo
echo "[4/5] Installing bop_toolkit_lib source"

SITE_PACKAGES="$("$BLENDER_PYTHON" -c \
    "import site; print(site.getsitepackages()[0])")"

echo "[SITE-PACKAGES] $SITE_PACKAGES"

TARGET="$SITE_PACKAGES/bop_toolkit_lib"

rm -rf "$TARGET"

cp -a \
    "$BOP_TOOLKIT_DIR/bop_toolkit_lib" \
    "$TARGET"

echo
echo "[5/5] Verifying installation"

"$BLENDER_PYTHON" - <<'PY'
import sys

print("Python:", sys.version)

modules = [
    "numpy",
    "PIL",
    "png",
    "OpenGL",
    "imageio",
    "skimage",
    "scipy",
    "vispy",
    "cv2",
    "matplotlib",
    "tqdm",
    "webdataset",
    "bop_toolkit_lib",
]

failed = []

for name in modules:
    try:
        module = __import__(name)
        path = getattr(module, "__file__", "built-in")
        print(f"[OK]   {name:<18} {path}")
    except Exception as exc:
        failed.append((name, str(exc)))
        print(f"[FAIL] {name:<18} {exc}")

print()

if failed:
    print("Installation verification FAILED:")
    for name, error in failed:
        print(f"  - {name}: {error}")
    sys.exit(1)

from bop_toolkit_lib import inout, misc

print("[OK] bop_toolkit_lib.inout")
print("[OK] bop_toolkit_lib.misc")

print()
print("========================================")
print(" Blender 5.x Python setup completed")
print("========================================")
PY