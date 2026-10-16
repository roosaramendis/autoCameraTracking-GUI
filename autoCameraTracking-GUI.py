#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AutoTracker GUI (Linux)
=======================
Front-end for the video -> camera-solve workflow:

    ffmpeg (frames) -> colmap feature_extractor -> colmap sequential_matcher
    -> colmap global_mapper (GLOMAP, built into COLMAP 4+) -> colmap model_converter (TXT)

Videos can live anywhere (nothing is copied), the output goes to any folder you
pick, and there are options to keep disk usage down. If COLMAP is missing or too
old, "Set up…" builds COLMAP from the official source (no AUR).
Everything a run creates goes into the project folder you pick: one folder per video.

Run:   python3 autotracker_gui.py
Needs: PySide6   (Arch/CachyOS: sudo pacman -S pyside6)
"""
from __future__ import annotations

import codecs
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

try:
    from PySide6.QtCore import QProcess, QSettings, Qt, QTimer, QUrl
    from PySide6.QtGui import QDesktopServices, QFontDatabase, QPainter
    from PySide6.QtWidgets import (
        QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog,
        QFileDialog, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
        QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
        QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSpinBox, QSplitter,
        QVBoxLayout, QWidget,
    )
except ImportError:
    sys.stderr.write(
        "PySide6 is not installed.\n"
        "  Arch / CachyOS : sudo pacman -S pyside6\n"
        "  anything else  : python3 -m pip install --user PySide6\n"
    )
    sys.exit(1)

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mts", ".m2ts", ".mxf"}
VIDEO_FILTER = "Videos (" + " ".join("*" + e for e in sorted(VIDEO_EXTS)) + ");;All files (*)"
FFMPEG_PROGRESS_RE = re.compile(
    r"^(frame|fps|stream_\d+_\d+_q|bitrate|total_size|out_time_us|out_time_ms|"
    r"out_time|dup_frames|drop_frames|speed|progress)="
)
DB_BYTES_PER_FRAME = 1.5e6  # ~8192 SIFT features/frame (128-byte descriptors) + matches; an upper-ish bound

COLMAP_TAG = "4.2.1"  # keep in sync with the default in BUILD_SCRIPT
CUDA_NVCC = Path("/opt/cuda/bin/nvcc")
# Where "Set up…" keeps COLMAP: src/ (source + build), colmap/ (installed COLMAP) and build_colmap.sh.
DEFAULT_BASE = Path.home() / "AutoTracker"   # same place as before (a folder picked earlier still wins)
MIN_FREE_FOR_BUILD = 5 * 1024 ** 3   # sources + CUDA builds + cuDSS + install, with headroom
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")   # terminal colour codes in COLMAP's log output


def self_built_colmap(base):
    return Path(base) / "colmap" / "bin" / "colmap"


# The setup script that "Set up…" runs (written to DEFAULT_BASE/build_colmap.sh so you can read it).
BUILD_SCRIPT = r'''#!/usr/bin/env bash
# =============================================================================
# AutoTracker setup: build COLMAP from the official source, with CUDA.
#
#  * COLMAP 4.0+ already contains the GLOMAP solver (`colmap global_mapper`),
#    so there is no separate GLOMAP to build.
#  * Packages come only from your official repos (pacman). NOTHING from the AUR:
#    METIS (AUR-only on Arch) is built from its official source if you don't have it.
#  * Everything that gets compiled is downloaded first and the script pauses so
#    you can review it.
#  * Everything (source, build files, the installed COLMAP) goes into ONE folder,
#    AUTOTRACKER_DIR, on whatever drive you choose. No sudo, nothing in /usr.
#    Only the pacman packages go on the system drive, like any package.
#
# Settings (environment variables), e.g.  env AUTOTRACKER_DIR=/mnt/big/AutoTracker ./build_colmap.sh
#   AUTOTRACKER_DIR  the one folder everything goes into   (default ~/AutoTracker)
#   COLMAP_TAG  release to build               (default 4.2.1)
#   SRC_DIR     source + build folders         (default $AUTOTRACKER_DIR/src/colmap)
#   PREFIX      install location               (default $AUTOTRACKER_DIR/colmap)
#   CUDA        ON / OFF                       (default ON)
#   CERES_GPU   ON / OFF: GPU bundle adjustment (Ceres + cuDSS)  (default: same as CUDA)
#   CUDA_ARCH   GPU architecture               (default native = the GPU in this PC)
#   JOBS        parallel compile jobs          (default: all CPU threads)
#
# GPU bundle adjustment: the repo's ceres-solver has no CUDA, so COLMAP's final solve step
# would run on the CPU. With CERES_GPU=ON, Ceres is built from its official source (pinned
# commit; cuDSS support isn't in a Ceres release yet) together with NVIDIA's cuDSS library
# (closed-source, downloaded from NVIDIA and checked against NVIDIA's published SHA256).
# =============================================================================
set -euo pipefail

AUTOTRACKER_DIR="${AUTOTRACKER_DIR:-$HOME/AutoTracker}"
COLMAP_TAG="${COLMAP_TAG:-4.2.1}"
SRC_DIR="${SRC_DIR:-$AUTOTRACKER_DIR/src/colmap}"
PREFIX="${PREFIX:-$AUTOTRACKER_DIR/colmap}"
MIN_FREE_GB=5   # sources + CUDA builds + cuDSS + install, with headroom
CUDA="${CUDA:-ON}"
CERES_GPU="${CERES_GPU:-$CUDA}"
CUDA_ARCH="${CUDA_ARCH:-native}"
JOBS="${JOBS:-$(nproc)}"
BUILD_DIR="$SRC_DIR/build-$COLMAP_TAG"   # one build folder per release: nothing needs deleting
REPO="https://github.com/colmap/colmap.git"

# METIS is not in Arch's official repos, so if it's missing it is built from the official
# KarypisLab source, pinned to exact versions, into $DEPS (static libraries).
METIS_TAG="v5.2.1"
GKLIB_COMMIT="3b7d61b9f885063c89901f3901fb4426f9cfb58f"   # GKlib has no release tag; METIS 5.2.1 uses GKlib master
DEPS="$PREFIX/deps"
DEPS_SRC="$SRC_DIR-deps"
BUILD_METIS=0

# GPU bundle adjustment (CERES_GPU=ON)
CERES_COMMIT="dd4199f718d27cf92eaf259ce52dca0abd0b9626"   # ceres-solver master, 2026-10-10 (has cuDSS)
CERES_INSTALL="$DEPS/ceres"
CUDSS_VERSION="0.8.0.10"
CUDSS_URL="https://developer.download.nvidia.com/compute/cudss/redist/libcudss/linux-x86_64"
CUDSS_SHA256_12="29716a05c64b28531309afcb7a9dd5044081a43197e72bdd0fbf29e9b4ee4707"   # from NVIDIA's
CUDSS_SHA256_13="ba18f5fd80dcbbe905d158caac5b3061d848442bb5abd477b5f296b4257a4937"   # redistrib_0.8.0.json
CUDSS_DIR="$DEPS/cudss"

step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
die()  { printf '\n\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

command -v pacman >/dev/null || die "This script is for Arch-based systems (pacman not found)."
[ "$(id -u)" -ne 0 ] || die "Run this as your normal user, not with sudo (sudo is only used for pacman)."

mkdir -p "$AUTOTRACKER_DIR" || die "Can't create $AUTOTRACKER_DIR"
FREE_GB=$(( $(df -P -k "$AUTOTRACKER_DIR" | awk 'NR==2 {print $4}') / 1024 / 1024 ))
echo "AutoTracker folder: $AUTOTRACKER_DIR   (${FREE_GB} GB free on that drive)"
if [ "$FREE_GB" -lt "$MIN_FREE_GB" ]; then
    echo "WARNING: building COLMAP needs about ${MIN_FREE_GB} GB there."
    if [ -t 0 ]; then
        read -rp "Press Enter to continue anyway, or Ctrl+C to stop... " _
    else
        die "Not enough free space in $AUTOTRACKER_DIR"
    fi
fi

# -----------------------------------------------------------------------------
step "1/5  Build dependencies (official repos only, no AUR)"
PKGS=(base-devel git cmake ninja boost boost-libs eigen openimageio metis google-glog gflags
      ceres-solver suitesparse cgal glew sqlite curl openssl qt6-base qt6-svg blas lapack ffmpeg)
if [ "$CUDA" = ON ] && [ ! -x /opt/cuda/bin/nvcc ]; then
    PKGS+=(cuda)
fi
if [ "$CUDA" = ON ] && [ "$CERES_GPU" = ON ]; then
    PKGS+=(abseil-cpp)   # the Ceres version with cuDSS uses abseil
fi
# pacman -T lists only what isn't satisfied yet (it understands "provides", e.g. openblas for blas)
mapfile -t MISSING < <(pacman -T "${PKGS[@]}" || true)
TO_INSTALL=() NOT_IN_REPOS=()
for p in "${MISSING[@]}"; do
    if pacman -Si "$p" >/dev/null 2>&1; then
        TO_INSTALL+=("$p")
    elif [ "$p" = metis ]; then
        BUILD_METIS=1
    else
        NOT_IN_REPOS+=("$p")
    fi
done
if [ "${#NOT_IN_REPOS[@]}" -gt 0 ]; then
    die "Not in your official repos: ${NOT_IN_REPOS[*]}. Install them yourself, then re-run."
fi
if [ "${#TO_INSTALL[@]}" -gt 0 ]; then
    echo "From the official repos: ${TO_INSTALL[*]}"
    sudo pacman -S --needed "${TO_INSTALL[@]}"
else
    echo "All official-repo packages are already installed."
fi
if [ "$BUILD_METIS" = 1 ] && [ -f "$DEPS/lib/libmetis.a" ]; then
    BUILD_METIS=0   # built on an earlier run
fi
if [ "$BUILD_METIS" = 1 ]; then
    echo "metis isn't in the official repos. Instead of using the AUR, it will be built straight from"
    echo "the official METIS source (KarypisLab on GitHub). You can review that source at step 3."
fi
if [ "$CUDA" = ON ] && [ ! -x /opt/cuda/bin/nvcc ]; then
    die "CUDA=ON but /opt/cuda/bin/nvcc is missing. Install 'cuda', or re-run with: env CUDA=OFF ./build_colmap.sh"
fi
[ "$CUDA" = ON ] || CERES_GPU=OFF
BUILD_CERES=0
if [ "$CERES_GPU" = ON ]; then
    CUDA_MAJOR="$(/opt/cuda/bin/nvcc --version | sed -nE 's/.*release ([0-9]+)\..*/\1/p')"
    case "$CUDA_MAJOR" in
        12) CUDSS_SHA256="$CUDSS_SHA256_12" ;;
        13) CUDSS_SHA256="$CUDSS_SHA256_13" ;;
        *)  die "cuDSS $CUDSS_VERSION is available for CUDA 12 and 13, but nvcc reports CUDA '$CUDA_MAJOR'. Re-run with: env CERES_GPU=OFF ./build_colmap.sh" ;;
    esac
    CUDSS_FILE="libcudss-linux-x86_64-${CUDSS_VERSION}_cuda${CUDA_MAJOR}-archive.tar.xz"
    CERES_STAMP="$CERES_COMMIT cudss-$CUDSS_VERSION cuda$CUDA_MAJOR"
    if [ "$(cat "$CERES_INSTALL/.autotracker-ceres" 2>/dev/null || true)" != "$CERES_STAMP" ]; then
        BUILD_CERES=1
    fi
fi

# -----------------------------------------------------------------------------
step "2/5  Download sources"
echo "COLMAP $COLMAP_TAG  ->  $SRC_DIR"
if [ -d "$SRC_DIR/.git" ]; then
    git -C "$SRC_DIR" fetch -q --depth 1 origin tag "$COLMAP_TAG"
else
    mkdir -p "$(dirname "$SRC_DIR")"
    git -c advice.detachedHead=false clone -q --depth 1 --branch "$COLMAP_TAG" "$REPO" "$SRC_DIR"
fi
git -C "$SRC_DIR" -c advice.detachedHead=false checkout -q "tags/$COLMAP_TAG" \
    || die "git checkout failed. If you edited files in $SRC_DIR, commit/stash them or remove that folder."
COMMIT="$(git -C "$SRC_DIR" rev-parse HEAD)"

if [ "$BUILD_METIS" = 1 ]; then
    echo "METIS $METIS_TAG + GKlib ${GKLIB_COMMIT:0:12}  ->  $DEPS_SRC"
    mkdir -p "$DEPS_SRC"
    if [ ! -d "$DEPS_SRC/GKlib/.git" ]; then
        git init -q "$DEPS_SRC/GKlib"
    fi
    if [ "$(git -C "$DEPS_SRC/GKlib" rev-parse -q --verify HEAD 2>/dev/null || true)" != "$GKLIB_COMMIT" ]; then
        git -C "$DEPS_SRC/GKlib" fetch -q --depth 1 https://github.com/KarypisLab/GKlib.git "$GKLIB_COMMIT"
        git -C "$DEPS_SRC/GKlib" -c advice.detachedHead=false checkout -q FETCH_HEAD
    fi
    if [ ! -d "$DEPS_SRC/METIS/.git" ]; then
        git -c advice.detachedHead=false clone -q --depth 1 --branch "$METIS_TAG" \
            https://github.com/KarypisLab/METIS.git "$DEPS_SRC/METIS"
    fi
fi

if [ "$BUILD_CERES" = 1 ]; then
    echo "Ceres ${CERES_COMMIT:0:12}  ->  $DEPS_SRC/ceres"
    mkdir -p "$DEPS_SRC"
    if [ ! -d "$DEPS_SRC/ceres/.git" ]; then
        git init -q "$DEPS_SRC/ceres"
    fi
    if [ "$(git -C "$DEPS_SRC/ceres" rev-parse -q --verify HEAD 2>/dev/null || true)" != "$CERES_COMMIT" ]; then
        git -C "$DEPS_SRC/ceres" fetch -q --depth 1 https://github.com/ceres-solver/ceres-solver.git "$CERES_COMMIT"
        git -C "$DEPS_SRC/ceres" -c advice.detachedHead=false checkout -q FETCH_HEAD
    fi

    echo "cuDSS $CUDSS_VERSION (CUDA $CUDA_MAJOR)  ->  $CUDSS_DIR"
    DL="$DEPS_SRC/downloads"
    mkdir -p "$DL"
    if [ ! -f "$DL/$CUDSS_FILE" ] || ! echo "$CUDSS_SHA256  $DL/$CUDSS_FILE" | sha256sum -c --quiet - 2>/dev/null; then
        curl -fL --progress-bar -o "$DL/$CUDSS_FILE.part" "$CUDSS_URL/$CUDSS_FILE" \
            || die "Downloading cuDSS failed: $CUDSS_URL/$CUDSS_FILE"
        mv "$DL/$CUDSS_FILE.part" "$DL/$CUDSS_FILE"
    fi
    echo "$CUDSS_SHA256  $DL/$CUDSS_FILE" | sha256sum -c --quiet - \
        || die "cuDSS download does not match NVIDIA's published SHA256. Not using it. ($DL/$CUDSS_FILE)"
    echo "    SHA256 OK (matches NVIDIA's published checksum)"
    mkdir -p "$CUDSS_DIR"
    tar -xJf "$DL/$CUDSS_FILE" -C "$CUDSS_DIR" --strip-components=1
fi

# -----------------------------------------------------------------------------
step "3/5  Review before building"
echo "Release:    COLMAP $COLMAP_TAG   (commit $COMMIT, from $REPO)"
echo "Install to: $PREFIX   (no sudo; delete that folder to uninstall)"
echo "Build in:   $BUILD_DIR"
echo "Options:    CUDA=$CUDA (arch: $CUDA_ARCH), GUI on, tests off,"
echo "            ONNX off (only needed for learned features, not SIFT; avoids downloading a prebuilt binary)"
echo
echo "While building, CMake downloads these source archives and checks their SHA256:"
grep -hE '^\s*URL(_HASH)?\s' "$SRC_DIR/src/thirdparty/CMakeLists.txt" | sed -E 's/^\s+/    /'
AUDIT=("$SRC_DIR    (COLMAP $COLMAP_TAG, $REPO)")
if [ "$BUILD_METIS" = 1 ]; then
    AUDIT+=("$DEPS_SRC/METIS    (METIS $METIS_TAG, https://github.com/KarypisLab/METIS)"
            "$DEPS_SRC/GKlib    (GKlib commit $GKLIB_COMMIT, https://github.com/KarypisLab/GKlib)")
    echo
    echo "METIS is built from source, not the AUR, so there is no PKGBUILD: the source itself is what to review."
    echo "These exact commands will run for it (into $DEPS, static libraries):"
    echo "    cd $DEPS_SRC/GKlib && make config cc=gcc prefix=$DEPS && make && make install"
    echo "    cd $DEPS_SRC/METIS && make config cc=gcc prefix=$DEPS gklib_path=$DEPS && make && make install"
    echo "    (with CFLAGS=\"-O2 -fPIC -std=gnu17\" and CMAKE_POLICY_VERSION_MINIMUM=3.5)"
fi
if [ "$BUILD_CERES" = 1 ]; then
    AUDIT+=("$DEPS_SRC/ceres    (Ceres commit $CERES_COMMIT, https://github.com/ceres-solver/ceres-solver)")
    echo
    echo "GPU bundle adjustment:"
    echo "  Ceres is built from source with CUDA + cuDSS (the repo's ceres-solver has neither)."
    echo "  cuDSS is NVIDIA's closed-source library, so there is no source to review. It came from"
    echo "    $CUDSS_URL/$CUDSS_FILE"
    echo "  and matches NVIDIA's published SHA256 ($CUDSS_SHA256)."
    echo "  Unpacked to $CUDSS_DIR   (license: $CUDSS_DIR/LICENSE.txt)"
fi
echo
echo "Everything that will be compiled is downloaded now and sits in these folders, ready to review:"
for a in "${AUDIT[@]}"; do echo "    $a"; done
echo
if [ -t 0 ]; then
    read -rp "Review them now if you like. Press Enter to build, or Ctrl+C to stop... " _
fi

# -----------------------------------------------------------------------------
step "4/5  Configure and compile (this takes a while)"
if [ "$BUILD_METIS" = 1 ]; then
    METIS_LOG="$DEPS_SRC/metis-build.log"
    echo "Building GKlib + METIS (static) -> $DEPS   (log: $METIS_LOG)"
    # -std=gnu17: this older C code predates C23 (the default in GCC 15+)
    # -fPIC: it gets linked into a PIE binary
    # CMAKE_POLICY_VERSION_MINIMUM: METIS asks for CMake 2.8, which CMake 4 refuses without this
    export CFLAGS="-O2 -fPIC -std=gnu17" CMAKE_POLICY_VERSION_MINIMUM=3.5
    if ! { (cd "$DEPS_SRC/GKlib" && make config cc=gcc prefix="$DEPS" && make -j "$JOBS" && make install) \
        && (cd "$DEPS_SRC/METIS" && make config cc=gcc prefix="$DEPS" gklib_path="$DEPS" \
            && make -j "$JOBS" && make install); } > "$METIS_LOG" 2>&1; then
        tail -n 30 "$METIS_LOG"
        die "Building METIS failed. Full log: $METIS_LOG"
    fi
    unset CFLAGS CMAKE_POLICY_VERSION_MINIMUM   # don't let these leak into the COLMAP build
    echo "METIS OK."
fi

if [ "$CUDA" = ON ]; then
    export CUDACXX=/opt/cuda/bin/nvcc
    # Arch's cuda package links the GCC that nvcc officially supports as /opt/cuda/bin/g++.
    # Using it as the CUDA host compiler avoids "unsupported GNU version" errors with newer GCC.
    if [ -e /opt/cuda/bin/g++ ]; then
        CUDAHOSTCXX="$(readlink -f /opt/cuda/bin/g++)"
        export CUDAHOSTCXX
    fi
fi

if [ "$BUILD_CERES" = 1 ]; then
    CERES_LOG="$DEPS_SRC/ceres-build.log"
    CERES_BUILD="$DEPS_SRC/ceres/build"
    echo "Building Ceres with CUDA + cuDSS (static) -> $CERES_INSTALL   (log: $CERES_LOG)"
    if ! cmake -S "$DEPS_SRC/ceres" -B "$CERES_BUILD" -G Ninja \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_INSTALL_PREFIX="$CERES_INSTALL" \
            -DCMAKE_PREFIX_PATH="$CUDSS_DIR" \
            -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
            -DWITH_CUDA=default -DWITH_SUITESPARSE=ON \
            -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
            -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF \
            > "$CERES_LOG" 2>&1; then
        tail -n 30 "$CERES_LOG"
        die "Configuring Ceres failed. Full log: $CERES_LOG"
    fi
    # Don't build a Ceres that would still fall back to the CPU: check it really got CUDA + cuDSS.
    CFG_H="$CERES_BUILD/include/ceres/internal/config.h"
    if grep -q '^#define CERES_NO_CUDA$' "$CFG_H"; then
        die "Ceres didn't enable CUDA (see $CERES_LOG). Nothing was built."
    fi
    if grep -q '^#define CERES_NO_CUDSS$' "$CFG_H"; then
        grep -iE 'cudss' "$CERES_LOG" | head -5
        die "Ceres didn't find cuDSS in $CUDSS_DIR (see $CERES_LOG). Nothing was built."
    fi
    echo "    CUDA: ON, cuDSS: ON"
    if ! { ninja -C "$CERES_BUILD" -j "$JOBS" && ninja -C "$CERES_BUILD" install; } >> "$CERES_LOG" 2>&1; then
        tail -n 30 "$CERES_LOG"
        die "Building Ceres failed. Full log: $CERES_LOG"
    fi
    echo "$CERES_STAMP" > "$CERES_INSTALL/.autotracker-ceres"
    echo "Ceres OK."
fi

CMAKE_ARGS=(-S "$SRC_DIR" -B "$BUILD_DIR" -G Ninja
            -DCMAKE_BUILD_TYPE=Release
            -DCMAKE_INSTALL_PREFIX="$PREFIX"
            -DONNX_ENABLED=OFF
            -DTESTS_ENABLED=OFF
            -DCCACHE_ENABLED=OFF   # ccache would keep a cache in ~/.cache; keep everything in one folder
            -DGUI_ENABLED=ON)
PREFIX_PATHS=()
if [ -f "$DEPS/lib/libmetis.a" ]; then
    PREFIX_PATHS+=("$DEPS")
    CMAKE_ARGS+=(-DMETIS_INCLUDE_DIR_HINTS="$DEPS/include"
                 -DMETIS_LIBRARY_DIR_HINTS="$DEPS/lib")
fi
CERES_CFG="$(find "$CERES_INSTALL" -name CeresConfig.cmake -print -quit 2>/dev/null || true)"
if [ "$CERES_GPU" = ON ] && [ -n "$CERES_CFG" ]; then
    PREFIX_PATHS+=("$CUDSS_DIR")
    # use the GPU Ceres instead of the repo one, and let colmap find libcudss.so at runtime
    CMAKE_ARGS+=(-DCeres_DIR="$(dirname "$CERES_CFG")"
                 -DCMAKE_INSTALL_RPATH="$CUDSS_DIR/lib"
                 -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON)
fi
if [ "${#PREFIX_PATHS[@]}" -gt 0 ]; then
    CMAKE_ARGS+=(-DCMAKE_PREFIX_PATH="$(IFS=';'; echo "${PREFIX_PATHS[*]}")")
fi
if [ "$CUDA" = ON ]; then
    CMAKE_ARGS+=(-DCUDA_ENABLED=ON -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH")
else
    CMAKE_ARGS+=(-DCUDA_ENABLED=OFF)
fi

cmake "${CMAKE_ARGS[@]}" 2>&1 | tee "$BUILD_DIR.configure.log"
if [ "$CUDA" = ON ] && ! grep -q 'CUDA: ON' "$BUILD_DIR.configure.log"; then
    echo
    echo "WARNING: CMake did not enable CUDA (see $BUILD_DIR.configure.log)."
    echo "         COLMAP would build CPU-only, which makes feature extraction much slower."
    if [ -t 0 ]; then
        read -rp "Press Enter to continue anyway, or Ctrl+C to stop... " _
    fi
fi

ninja -C "$BUILD_DIR" -j "$JOBS"
ninja -C "$BUILD_DIR" install

# -----------------------------------------------------------------------------
step "5/5  Check"
BIN="$PREFIX/bin/colmap"
"$BIN" help 2>&1 | head -2
if "$BIN" help 2>&1 | grep -q global_mapper; then
    echo "OK: global_mapper (the GLOMAP solver) is available."
else
    echo "WARNING: global_mapper not found in this build."
fi
if [ "$CERES_GPU" = ON ]; then
    if ldd "$BIN" | grep -q 'libcudss.*=> /'; then
        echo "OK: GPU bundle adjustment (Ceres + cuDSS) is linked in."
        echo "$CERES_COMMIT cudss-$CUDSS_VERSION" > "$PREFIX/.ceres-gpu"
    else
        echo "WARNING: colmap isn't linked against cuDSS; bundle adjustment will still run on the CPU."
        ldd "$BIN" | grep -iE 'cudss|not found' || true
    fi
fi
echo
echo "Installed: $BIN"
echo "AutoTracker GUI finds it there automatically (press Re-check)."
echo "To use colmap in your terminal too (fish):  fish_add_path $PREFIX/bin"
echo "If colmap stops starting after a big system update (library version bumps), re-run this script."
'''


# --------------------------------------------------------------------------- helpers
def run_quiet(cmd, timeout=20):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout or "", p.stderr or ""
    except (OSError, subprocess.TimeoutExpired) as e:
        return -1, "", str(e)


def truthy(v):
    return v in (True, 1, "true", "True", "1")


def safe_int(v, default, lo, hi):
    """Parse a stored setting; fall back to default if it's missing/corrupt, clamp to [lo, hi]."""
    try:
        return max(lo, min(hi, int(v)))
    except (TypeError, ValueError):
        return default


def fmt_bytes(n):
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{int(n)} B" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def nearest_existing(p: Path) -> Path:
    p = p.expanduser()
    while not p.exists() and p != p.parent:
        p = p.parent
    return p


def distro_family():
    info = {}
    try:
        for line in Path("/etc/os-release").read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                info[k] = v.strip().strip('"')
    except OSError:
        pass
    ids = [info.get("ID", "")] + info.get("ID_LIKE", "").split()
    if any(i in ("arch", "cachyos", "manjaro", "endeavouros") for i in ids):
        return "arch"
    if any(i in ("debian", "ubuntu") for i in ids):
        return "debian"
    return "unknown"


def is_exe(p):
    return p is not None and Path(p).is_file() and os.access(p, os.X_OK)


def find_tool(name, override, fallbacks=()):
    """Custom path if set; otherwise known install locations, then PATH."""
    if override:
        p = Path(override).expanduser()
        return str(p) if is_exe(p) else None
    for f in fallbacks:
        if is_exe(f):
            return str(f)
    return shutil.which(name)


def probe_tools(overrides, base=None, extra=()):
    t = SimpleNamespace(ffmpeg=None, ffprobe=None, colmap=None, glomap=None, glomap_forced=False,
                        version="", cuda=None, gpu_kind="", global_mapper=False,
                        opt_feat_gpu="SiftExtraction.use_gpu",
                        opt_feat_size="SiftExtraction.max_image_size",
                        opt_match_gpu="SiftMatching.use_gpu")
    t.ffmpeg = find_tool("ffmpeg", overrides.get("ffmpeg"))
    if t.ffmpeg:
        sib = Path(t.ffmpeg).with_name("ffprobe")
        t.ffprobe = str(sib) if sib.is_file() else shutil.which("ffprobe")
    # A COLMAP built by "Set up…" wins over one on PATH (which may be an older package).
    t.colmap = find_tool("colmap", overrides.get("colmap"),
                         fallbacks=([self_built_colmap(base)] if base else []) + list(extra))
    if t.colmap:
        _, out, err = run_quiet([t.colmap, "help"], timeout=30)
        text = out + err
        m = re.search(r"COLMAP\s+([0-9][^\s]*)", text)
        t.version = m.group(1) if m else ""
        # COLMAP 4.x: "with CUDA" / "with HIP" / "without GPU support";  3.x: "with CUDA" / "without CUDA"
        if re.search(r"\bwith\s+(CUDA|HIP)\b", text):
            t.gpu_kind = re.search(r"\bwith\s+(CUDA|HIP)\b", text).group(1)
            t.cuda = True
        elif re.search(r"\bwithout\s+(CUDA|GPU)", text):
            t.cuda = False
        else:
            t.cuda = None  # couldn't tell
        t.global_mapper = "global_mapper" in text
        # Newer COLMAP renamed some SiftExtraction.* / SiftMatching.* options to FeatureExtraction.* /
        # FeatureMatching.*. Check each option in its own command's --help instead of guessing.
        _, out, err = run_quiet([t.colmap, "feature_extractor", "--help"], timeout=30)
        fx = out + err
        _, out, err = run_quiet([t.colmap, "sequential_matcher", "--help"], timeout=30)
        sm = out + err
        if "FeatureExtraction.use_gpu" in fx:
            t.opt_feat_gpu = "FeatureExtraction.use_gpu"
        if "FeatureExtraction.max_image_size" in fx:
            t.opt_feat_size = "FeatureExtraction.max_image_size"
        if "FeatureMatching.use_gpu" in sm:
            t.opt_match_gpu = "FeatureMatching.use_gpu"
    # Standalone GLOMAP is deprecated (merged into COLMAP 4.0); only used if you point at one,
    # or as a last resort when COLMAP has no global_mapper.
    t.glomap = find_tool("glomap", overrides.get("glomap"))
    t.glomap_forced = bool(overrides.get("glomap")) and t.glomap is not None
    return t


def mapper_cmd(t):
    if t.glomap and getattr(t, "glomap_forced", False):
        return [t.glomap, "mapper"]
    if t.colmap and t.global_mapper:
        return [t.colmap, "global_mapper"]
    if t.glomap:
        return [t.glomap, "mapper"]
    return None


def write_build_script(base):
    """Write BUILD_SCRIPT into the AutoTracker folder. A copy you edited yourself is kept, not overwritten."""
    base = Path(base)
    path = base / "build_colmap.sh"
    stamp = base / ".build_colmap.sha256"  # hash of the copy this app last wrote
    new_hash = hashlib.sha256(BUILD_SCRIPT.encode()).hexdigest()
    edited = False
    if path.exists():
        cur_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        last = stamp.read_text().strip() if stamp.exists() else ""
        edited = cur_hash not in (new_hash, last)
    if not edited:
        base.mkdir(parents=True, exist_ok=True)
        path.write_text(BUILD_SCRIPT)
        stamp.write_text(new_hash)
    return path, edited


def setup_plan(t):
    """What "Set up…" should offer. Returns dict(kind, reason, cmd).
    kind: none | build (run the COLMAP build script) | pacman/apt (one command) | manual (text only)"""
    fam = distro_family()
    has_mapper = bool(t.colmap and t.global_mapper) or bool(t.glomap)
    if not t.colmap:
        reason = "COLMAP isn't installed."
    elif not has_mapper:
        reason = (f"Your COLMAP ({t.version or 'unknown version'}) is older than 4.0, so it doesn't have "
                  f"the built-in GLOMAP solver (global_mapper).")
    elif t.cuda is False and CUDA_NVCC.exists():
        reason = ("Optional: your COLMAP was built without CUDA, so feature extraction runs on the CPU "
                  "(slow). You have the CUDA toolkit, so rebuilding with CUDA is recommended.")
    elif (getattr(t, "managed", False) and t.cuda and not getattr(t, "ceres_gpu", False)
          and CUDA_NVCC.exists()):
        reason = ("Recommended: bundle adjustment (the final solve step) runs on the CPU, because the "
                  "ceres-solver from the repos has no GPU support. Set up rebuilds Ceres with CUDA + cuDSS "
                  "and relinks COLMAP, so that step runs on your GPU too.")
    else:
        reason = ""
    if reason:
        if fam == "arch":
            return dict(kind="build", reason=reason, cmd=None)
        return dict(kind="manual", reason=reason, cmd=None)
    if not t.ffmpeg:
        if fam == "arch":
            return dict(kind="pacman", reason="ffmpeg isn't installed.", cmd="sudo pacman -S --needed ffmpeg")
        if fam == "debian":
            return dict(kind="apt", reason="ffmpeg isn't installed.", cmd="sudo apt install -y ffmpeg")
        return dict(kind="manual", reason="ffmpeg isn't installed.", cmd=None)
    return dict(kind="none", reason="", cmd=None)


TERMINALS = [
    ("konsole", ["--nofork", "-e"]), ("kitty", []), ("alacritty", ["-e"]), ("foot", []),
    ("wezterm", ["start", "--"]), ("gnome-terminal", ["--"]), ("xfce4-terminal", ["-x"]),
    ("xterm", ["-e"]),
]


def launch_terminal(script):
    candidates = []
    env_term = os.environ.get("TERMINAL")
    if env_term and shutil.which(env_term):
        candidates.append((env_term, ["-e"]))
    candidates += TERMINALS
    for term, pre in candidates:
        exe = shutil.which(term)
        if exe:
            try:
                return subprocess.Popen([exe, *pre, "sh", "-c", script])
            except OSError:
                continue
    return None


def probe_video(ffprobe, path):
    code, out, _ = run_quiet([
        ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames,duration:format=duration",
        "-of", "json", path])
    if code != 0:
        return None
    try:
        j = json.loads(out)
    except ValueError:
        return None
    s = (j.get("streams") or [{}])[0]

    def num(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    fps = 0.0
    try:
        a, b = str(s.get("avg_frame_rate", "0/0")).split("/")
        fps = float(a) / float(b) if float(b) else 0.0
    except ValueError:
        pass
    nb = s.get("nb_frames")
    return dict(dur=num(s.get("duration")) or num((j.get("format") or {}).get("duration")),
                fps=fps, nb=int(nb) if str(nb).isdigit() else None,
                w=s.get("width"), h=s.get("height"))


def frames_for(meta, fps_limit):
    if not meta or not meta["dur"]:
        return None
    if fps_limit > 0:
        eff = min(fps_limit, meta["fps"]) if meta["fps"] else fps_limit
        return max(1, round(meta["dur"] * eff))
    if meta["nb"]:
        return meta["nb"]
    return round(meta["dur"] * meta["fps"]) if meta["fps"] else None


# --------------------------------------------------------------------------- widgets
class VideoList(QListWidget):
    """List of video paths; accepts drag-and-drop of files and folders."""

    def __init__(self, on_change):
        super().__init__()
        self.on_change = on_change
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setAcceptDrops(True)

    def paths(self):
        return [self.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.count())]

    def add_paths(self, paths):
        have = set(self.paths())
        for raw in paths:
            p = Path(raw)
            try:
                files = sorted(f for f in p.iterdir() if f.is_file()) if p.is_dir() else [p]
            except OSError:  # unreadable folder
                continue
            for f in files:
                if f.suffix.lower() in VIDEO_EXTS and str(f) not in have:
                    it = QListWidgetItem(f"{f.name}    —    {f.parent}")
                    it.setData(Qt.ItemDataRole.UserRole, str(f))
                    it.setToolTip(str(f))
                    self.addItem(it)
                    have.add(str(f))
        self.on_change()

    def remove_selected(self):
        for it in self.selectedItems():
            self.takeItem(self.row(it))
        self.on_change()

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dragMoveEvent(self, e):
        e.acceptProposedAction()

    def dropEvent(self, e):
        self.add_paths([u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()])
        e.acceptProposedAction()

    def paintEvent(self, e):
        super().paintEvent(e)
        if self.count() == 0:  # explain what goes here while the list is empty
            p = QPainter(self.viewport())
            p.setPen(self.palette().placeholderText().color())
            p.drawText(self.viewport().rect().adjusted(12, 12, -12, -12),
                       Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                       "No videos yet.\n\nDrag video files (or a folder of videos) here,\n"
                       "or use the buttons below.")
            p.end()


# --------------------------------------------------------------------------- main window
class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AutoTracker — video → camera solve (COLMAP + GLOMAP)")
        self.resize(1200, 820)
        self.settings = QSettings("AutoTrackerGUI", "settings")
        self.tools = None
        self.meta = {}
        self.running = False
        self.cancelled = False
        self.steps, self.jobs, self.idx, self.dead = [], [], 0, set()
        self.cur_kind, self._buf = "", ""
        self._dec = codecs.getincrementaldecoder("utf-8")("replace")
        self.run_id = 0
        self.install_proc = None
        self._gpu_saved, self._gpu_init_done = None, False

        self.proc = QProcess(self)
        self.proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.proc.readyRead.connect(self.on_output)
        self.proc.finished.connect(self.on_finished)
        self.proc.errorOccurred.connect(self.on_proc_error)

        self.poll = QTimer(self)
        self.poll.setInterval(1500)
        self.poll.timeout.connect(self.check_install_done)

        self._build_ui()
        self._load_settings()
        self.refresh_tools()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        # ---- 1. setup
        setup = QGroupBox("1 · Setup — required tools")
        grid = QGridLayout(setup)
        self.dep = {}
        for r, (key, label) in enumerate([("ffmpeg", "ffmpeg"), ("colmap", "COLMAP"), ("glomap", "Solver")]):
            grid.addWidget(QLabel(f"<b>{label}</b>"), r, 0)
            st = QLabel()
            st.setTextFormat(Qt.TextFormat.RichText)
            st.setWordWrap(True)
            st.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            b, a = QPushButton("Browse…"), QPushButton("Auto")
            b.setToolTip("Use an old standalone glomap binary instead (normally not needed)" if key == "glomap"
                         else "Use a binary from a custom location (e.g. your own build)")
            a.setToolTip("Forget the custom path and search PATH again")
            b.clicked.connect(lambda _=False, k=key: self.pick_tool(k))
            a.clicked.connect(lambda _=False, k=key: self.clear_tool(k))
            grid.addWidget(st, r, 1)
            grid.addWidget(b, r, 2)
            grid.addWidget(a, r, 3)
            self.dep[key] = st
        grid.setColumnStretch(1, 1)
        row = QHBoxLayout()
        self.btn_recheck = QPushButton("Re-check")
        self.btn_install = QPushButton("Set up…")
        self.btn_install.setToolTip("Install ffmpeg / build COLMAP from source if something is missing, "
                                    "or add GPU bundle adjustment. Shows what it will do before anything runs.")
        self.btn_recheck.setToolTip("Look for ffmpeg and COLMAP again (e.g. after installing something)")
        self.btn_recheck.clicked.connect(self.refresh_tools)
        self.btn_install.clicked.connect(self.show_install)
        row.addWidget(self.btn_recheck)
        row.addWidget(self.btn_install)
        row.addStretch(1)
        grid.addLayout(row, 3, 0, 1, 4)

        # ---- 2. project + videos
        vids = QGroupBox("2 · Project and videos")
        vl = QVBoxLayout(vids)
        orow = QHBoxLayout()
        orow.addWidget(QLabel("<b>Project folder</b>"))
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("Choose a folder for this project…")
        self.out_edit.setToolTip("Everything this project creates goes here: one folder per video, "
                                 "with its image sequence and camera solve")
        self.out_edit.textChanged.connect(self.update_summary)
        ob = QPushButton("Choose…")
        ob.setToolTip("Pick the project folder. Image sequences and camera solves are saved inside it.")
        ob.clicked.connect(self.pick_output)
        orow.addWidget(self.out_edit, 1)
        orow.addWidget(ob)
        vl.addLayout(orow)
        phint = QLabel("Each video gets its own folder in the project:  <i>video name</i>/images "
                       "(image sequence)  and  <i>video name</i>/sparse (camera solve for Blender).")
        phint.setWordWrap(True)
        vl.addWidget(phint)
        self.vlist = VideoList(self.update_summary)
        self.vlist.setMinimumHeight(130)
        self.vlist.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.vlist.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        vl.addWidget(self.vlist)
        vr = QGridLayout()   # 2 x 2 so the buttons never squeeze the panel
        b1, b2, b3, b4 = (QPushButton(t) for t in
                          ("Add video files…", "Add all videos in a folder…", "Remove selected", "Clear list"))
        b1.setToolTip("Pick one or more video files")
        b2.setToolTip("Pick a folder: every video file directly inside it is added (subfolders aren't searched)")
        b3.setToolTip("Remove the highlighted videos from the list (the files themselves are not touched)")
        b4.setToolTip("Empty the list (the files themselves are not touched)")
        b1.clicked.connect(self.add_videos)
        b2.clicked.connect(self.add_folder)
        b3.clicked.connect(self.vlist.remove_selected)
        b4.clicked.connect(lambda: (self.vlist.clear(), self.update_summary()))
        vr.addWidget(b1, 0, 0)
        vr.addWidget(b2, 0, 1)
        vr.addWidget(b3, 1, 0)
        vr.addWidget(b4, 1, 1)
        vl.addLayout(vr)
        tip = QLabel("Your videos are only read: nothing is copied or moved.")
        tip.setWordWrap(True)
        vl.addWidget(tip)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        vl.addWidget(self.summary)

        # ---- 3. settings
        sett = QGroupBox("3 · Settings")
        form = QFormLayout(sett)
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(0, 120)
        self.fps_spin.setSpecialValueText("Every frame")
        self.fps_spin.setSuffix(" fps")
        self.fps_spin.setToolTip("Frames per second to extract. Fewer frames = much less disk and time. "
                                 "Never goes above the video's own frame rate.")
        self.width_spin = QSpinBox()
        self.width_spin.setRange(0, 8192)
        self.width_spin.setSingleStep(160)
        self.width_spin.setSpecialValueText("Original size")
        self.width_spin.setSuffix(" px wide")
        self.width_spin.setToolTip("Downscale extracted frames (never upscales). 1920 is plenty for most solves.")
        self.quality = QComboBox()
        for label, q in (("High (q2)", 2), ("Medium (q5)", 5), ("Small (q8)", 8)):
            self.quality.addItem(label, q)
        self.quality.setToolTip("JPEG quality of extracted frames. Smaller files cost a little feature accuracy.")
        self.sift_spin = QSpinBox()
        self.sift_spin.setRange(512, 16384)
        self.sift_spin.setValue(4096)
        self.sift_spin.setSingleStep(512)
        self.sift_spin.setToolTip("Max image size used for feature detection (original script: 4096).")
        self.overlap_spin = QSpinBox()
        self.overlap_spin.setRange(1, 100)
        self.overlap_spin.setValue(15)
        self.overlap_spin.setToolTip("How many neighbouring frames each frame is matched against (original: 15).")
        self.chk_gpu = QCheckBox("Use GPU (CUDA) for features and matching")
        self.chk_gpu.setToolTip("Run feature detection and matching on your NVIDIA GPU (much faster). "
                                "Untick only if it fails.")
        form.addRow("Extract at:", self.fps_spin)
        form.addRow("Frame width:", self.width_spin)
        form.addRow("Frame quality:", self.quality)
        form.addRow("Feature image size:", self.sift_spin)
        form.addRow("Matcher overlap:", self.overlap_spin)
        form.addRow("", self.chk_gpu)

        # ---- 4. storage
        stor = QGroupBox("4 · Save disk space")
        sl = QVBoxLayout(stor)
        self.chk_skip = QCheckBox("Skip videos that are already solved")
        self.chk_del_frames = QCheckBox("Delete extracted frames when a video is finished (saves the most space)")
        self.chk_del_frames.setToolTip("The solved camera/points (sparse/) are kept. If your Blender import "
                                       "uses the frames as a background, use the original video there instead.")
        self.chk_del_db = QCheckBox("Delete the COLMAP database when a video is finished")
        self.chk_skip.setToolTip("Videos that already have a camera solve in the project folder are not redone")
        self.chk_del_db.setToolTip("database.db holds COLMAP's features and matches (often hundreds of MB). "
                                   "Not needed for Blender; only for re-running COLMAP steps by hand.")
        sl.addWidget(self.chk_skip)
        sl.addWidget(self.chk_del_frames)
        sl.addWidget(self.chk_del_db)
        self.btn_estimate = QPushButton("Estimate disk use")
        self.btn_estimate.setToolTip("Work out how much space the listed videos will need with these settings "
                                     "(samples one frame per video)")
        self.btn_estimate.clicked.connect(lambda: self.estimate(show=True))
        sl.addWidget(self.btn_estimate)
        self.est_label = QLabel()
        self.est_label.setWordWrap(True)
        sl.addWidget(self.est_label)

        self.controls = QWidget()
        cl = QVBoxLayout(self.controls)
        cl.setContentsMargins(0, 0, 0, 0)
        for g in (setup, vids, sett, stor):
            cl.addWidget(g)
        cl.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(self.controls)
        # never narrower than its contents, so nothing on the left gets cut off
        scroll.setMinimumWidth(self.controls.minimumSizeHint().width()
                               + scroll.verticalScrollBar().sizeHint().width() + 2 * scroll.frameWidth() + 4)

        # ---- right: run + log
        right = QWidget()
        rl = QVBoxLayout(right)
        rr = QHBoxLayout()
        self.btn_start = QPushButton("▶  Start")
        self.btn_cancel = QPushButton("■  Cancel")
        self.btn_open = QPushButton("Open project folder")
        self.btn_start.clicked.connect(self.start)
        self.btn_cancel.clicked.connect(self.cancel)
        self.btn_open.clicked.connect(self.open_output)
        self.btn_cancel.setEnabled(False)
        self.btn_start.setToolTip("Process every video in the list: extract frames, find and match features, "
                                  "solve the camera, export for Blender")
        self.btn_cancel.setToolTip("Stop the current run. The video being processed is redone next time.")
        self.btn_open.setToolTip("Open the project folder in your file manager")
        rr.addWidget(self.btn_start)
        rr.addWidget(self.btn_cancel)
        rr.addStretch(1)
        rr.addWidget(self.btn_open)
        rl.addLayout(rr)
        self.progress = QProgressBar()
        self.status = QLabel("Idle.")
        self.status.setWordWrap(True)
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumBlockCount(6000)
        self.log_box.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        rl.addWidget(self.progress)
        rl.addWidget(self.status)
        rl.addWidget(self.log_box, 1)

        # draggable divider between the settings (left) and the log (right)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(scroll)
        self.splitter.addWidget(right)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 1)
        right.setMinimumWidth(320)
        self.setCentralWidget(self.splitter)
        left_w = max(scroll.minimumWidth(), 640)
        self.resize(max(1200, left_w + 560), 860)
        self.splitter.setSizes([left_w, self.width() - left_w])

    # ------------------------------------------------------------------ settings
    def _load_settings(self):
        s = self.settings
        self.out_edit.setText(s.value("project", "") or "")
        self.fps_spin.setValue(safe_int(s.value("fps"), 0, 0, 120))
        self.width_spin.setValue(safe_int(s.value("width"), 0, 0, 8192))
        self.quality.setCurrentIndex(safe_int(s.value("quality"), 0, 0, self.quality.count() - 1))
        self.sift_spin.setValue(safe_int(s.value("sift"), 4096, 512, 16384))
        self.overlap_spin.setValue(safe_int(s.value("overlap"), 15, 1, 100))
        self.chk_skip.setChecked(truthy(s.value("skip", True)))
        self.chk_del_frames.setChecked(truthy(s.value("del_frames", False)))
        self.chk_del_db.setChecked(truthy(s.value("del_db", False)))
        self._gpu_saved = s.value("gpu", None)
        st = s.value("splitter")
        if st is not None:
            try:
                self.splitter.restoreState(st)   # where you last dragged the divider
            except (TypeError, ValueError):
                pass
        self.fps_spin.valueChanged.connect(self.update_summary)

    def _save_settings(self):
        s = self.settings
        s.setValue("project", self.out_edit.text())
        s.setValue("fps", self.fps_spin.value())
        s.setValue("width", self.width_spin.value())
        s.setValue("quality", self.quality.currentIndex())
        s.setValue("sift", self.sift_spin.value())
        s.setValue("overlap", self.overlap_spin.value())
        s.setValue("gpu", self.chk_gpu.isChecked())
        s.setValue("skip", self.chk_skip.isChecked())
        s.setValue("del_frames", self.chk_del_frames.isChecked())
        s.setValue("del_db", self.chk_del_db.isChecked())
        s.setValue("splitter", self.splitter.saveState())

    def closeEvent(self, e):
        if self.running:
            if QMessageBox.question(self, "AutoTracker", "A run is in progress. Cancel it and quit?") \
                    != QMessageBox.StandardButton.Yes:
                e.ignore()
                return
            self.cancelled = True
            self.proc.kill()
            self.proc.waitForFinished(2000)
        self._save_settings()
        e.accept()

    # ------------------------------------------------------------------ tools / setup
    def base_dir(self):
        """Where Set up keeps COLMAP (source, build, install): the folder picked in the earlier
        version if there was one, otherwise ~/AutoTracker — unchanged from before."""
        v = self.settings.value("base_dir", "") or ""
        return Path(v).expanduser() if v else DEFAULT_BASE

    def colmap_locations(self):
        """Every place a COLMAP built by any version of this app's Set up can be."""
        return [self_built_colmap(self.base_dir()),
                Path.home() / ".local" / "opt" / "colmap" / "bin" / "colmap"]   # first version

    def free_bytes(self, path):
        try:
            return shutil.disk_usage(nearest_existing(Path(path))).free
        except OSError:
            return None

    def refresh_tools(self):
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            ov = {k: self.settings.value(f"path/{k}", "") for k in ("ffmpeg", "colmap", "glomap")}
            self.tools = probe_tools(ov, extra=self.colmap_locations())
        finally:
            QApplication.restoreOverrideCursor()
        t = self.tools
        # built by Set up? and with GPU bundle adjustment (Ceres + cuDSS)?
        t.managed = bool(t.colmap) and Path(t.colmap) in self.colmap_locations()
        t.ceres_gpu = t.managed and (Path(t.colmap).parent.parent / ".ceres-gpu").exists()
        ok, bad = '<span style="color:#3aa35f">✓</span>', '<span style="color:#d9534f">✗</span>'
        warn = '<span style="color:#d9a13a">⚠</span>'

        def custom(k):
            return " <i>(custom path)</i>" if ov.get(k) else ""

        self.dep["ffmpeg"].setText(f"{ok} {t.ffmpeg}{custom('ffmpeg')}" if t.ffmpeg
                                   else f"{bad} not found")
        if t.colmap:
            cu = {True: f"{ok} {getattr(t, 'gpu_kind', '') or 'CUDA'}",
                  False: f"{warn} no GPU support (slow CPU features)",
                  None: "GPU support unknown"}[t.cuda]
            where = " <i>(built by Set up…)</i>" if t.managed else custom("colmap")
            ba = ""
            if t.cuda and t.managed:
                ba = (f" · {ok} GPU bundle adjustment" if t.ceres_gpu else
                      f" · {warn} bundle adjustment on CPU — <b>Set up…</b> adds GPU")
            self.dep["colmap"].setText(f"{ok} {t.version or 'found'} · {cu}{ba}<br><small>{t.colmap}"
                                       f"{where}</small>")
        else:
            self.dep["colmap"].setText(f"{bad} not found — click <b>Set up…</b>")
        m = mapper_cmd(t)
        if m and m[-1] == "global_mapper":
            self.dep["glomap"].setText(f"{ok} COLMAP <tt>global_mapper</tt> (GLOMAP, built in)")
        elif m:
            self.dep["glomap"].setText(f"{ok} standalone glomap (older){custom('glomap')}<br><small>{m[0]}</small>")
        elif t.colmap:
            self.dep["glomap"].setText(f"{bad} needs COLMAP 4.0 or newer — click <b>Set up…</b>")
        else:
            self.dep["glomap"].setText(f"{bad} comes with COLMAP 4.0+")

        if not self._gpu_init_done:  # only pick a default on first check; never override the user afterwards
            self._gpu_init_done = True
            self.chk_gpu.setChecked(t.cuda is not False if self._gpu_saved is None else truthy(self._gpu_saved))
        self.chk_gpu.setToolTip("Run feature detection and matching on your NVIDIA GPU (much faster). "
                                "Untick only if it fails." if t.cuda is not False else
                                "Your COLMAP reports no CUDA. GPU SIFT will probably fail — uncheck if it does.")
        for p in self.vlist.paths():
            self.meta.pop(p, None)
        self.update_summary()

    def pick_tool(self, key):
        path, _ = QFileDialog.getOpenFileName(self, f"Select the {key} executable", "/usr/bin")
        if path:
            self.settings.setValue(f"path/{key}", path)
            self.refresh_tools()

    def clear_tool(self, key):
        self.settings.remove(f"path/{key}")
        self.refresh_tools()

    def setup_text(self, plan, script_path=None, edited=False):
        """The explanation shown in the Set up dialog (plain text)."""
        if plan["kind"] == "none":
            return "Everything needed is installed. ✓"
        if plan["kind"] == "build":
            base = self.base_dir()
            free = self.free_bytes(base)
            cuda = ("with CUDA for your NVIDIA GPU" if CUDA_NVCC.exists()
                    else "with CUDA (the CUDA toolkit will be installed first)")
            lines = [
                plan["reason"], "",
                f"Set up will build COLMAP {COLMAP_TAG} from the official source, {cuda}.",
                "COLMAP 4 already includes the GLOMAP solver, so there is nothing else to build.",
                "For GPU bundle adjustment it also builds Ceres (the solver) with CUDA + cuDSS.",
                "cuDSS is NVIDIA's closed-source library: downloaded from NVIDIA and checked against",
                "NVIDIA's published SHA256 before use. If COLMAP is already built, it's only relinked.",
                "", f"COLMAP is built and installed in {base}  (about {fmt_bytes(MIN_FREE_FOR_BUILD)}; "
                "delete that folder to uninstall).",
            ]
            if free is not None and free < MIN_FREE_FOR_BUILD:
                lines += [f"  ⚠ Only {fmt_bytes(free)} free on that drive — the build may run out of space."]
            lines += [
                "", "What happens in the terminal:",
                "  1. pacman installs any missing build dependencies from the official repos",
                "     (it lists them and asks you to confirm). Nothing comes from the AUR:",
                "     metis (AUR-only on Arch) is built from its official source instead.",
                "     These are normal system packages, so they go on your system drive like any package.",
                "  2. The source code is downloaded and the script pauses, so you can read it",
                "     before anything is compiled.",
                "  3. It compiles (this takes a while) and installs. No sudo for this part.",
            ]
            if script_path:
                lines += ["", f"Script: {script_path}" + ("   (your edited copy — kept as is)" if edited else ""),
                          'Use "View script" to read all of it first.']
            return "\n".join(lines)
        if plan["kind"] in ("pacman", "apt"):
            return f"{plan['reason']}\n\nThis runs in a terminal (type your sudo password there):\n\n  {plan['cmd']}"
        return (f"{plan['reason']}\n\nAutomatic setup only supports Arch-based systems. Install ffmpeg with your "
                "package manager, and COLMAP 4.0 or newer from source:\nhttps://colmap.github.io/install.html\n"
                "Then press Re-check (or use Browse… next to COLMAP).")

    def show_install(self):
        plan = setup_plan(self.tools)
        script_path, edited = (None, False)
        if plan["kind"] == "build":
            try:
                script_path, edited = write_build_script(self.base_dir())
            except OSError as e:
                QMessageBox.warning(self, "AutoTracker", f"Couldn't write the setup script into "
                                    f"{self.base_dir()}:\n{e}")
                return
            plan["cmd"] = (f"env AUTOTRACKER_DIR={shlex.quote(str(self.base_dir()))} "
                           f"bash {shlex.quote(str(script_path))}")
        dlg = QDialog(self)
        dlg.setWindowTitle("Set up")
        dlg.resize(720, 400)
        lay = QVBoxLayout(dlg)
        box = QPlainTextEdit()
        box.setReadOnly(True)
        box.setPlainText(self.setup_text(plan, script_path, edited))
        lay.addWidget(box, 1)
        row = QHBoxLayout()
        if script_path:
            b_view = QPushButton("View script")
            b_view.setToolTip("Open the setup script in your text editor so you can read all of it first")
            b_view.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(script_path))))
            row.addWidget(b_view)
        row.addStretch(1)
        if plan["cmd"]:
            b_copy, b_run = QPushButton("Copy command"), QPushButton("Run in terminal")
            b_copy.clicked.connect(lambda: QApplication.clipboard().setText(plan["cmd"]))
            b_run.clicked.connect(lambda: (self.run_in_terminal(plan["cmd"]), dlg.accept()))
            b_run.setDefault(True)
            b_copy.setToolTip("Copy the command, to run it in your own terminal instead")
            b_run.setToolTip("Open a terminal and run it there (you type your sudo password in that window)")
            row.addWidget(b_copy)
            row.addWidget(b_run)
        b_close = QPushButton("Close")
        b_close.setToolTip("Close without running anything")
        b_close.clicked.connect(dlg.reject)
        row.addWidget(b_close)
        lay.addLayout(row)
        dlg.exec()

    def run_in_terminal(self, cmd):
        script = cmd + "; echo; printf 'Finished. Press Enter to close... '; read _x"
        self.install_proc = launch_terminal(script)
        if not self.install_proc:
            QMessageBox.warning(self, "AutoTracker",
                                "No terminal emulator found (tried konsole, kitty, alacritty, foot, wezterm, "
                                "gnome-terminal, xfce4-terminal, xterm).\nUse 'Copy command' and run it "
                                "yourself, then press Re-check.")
            return
        self.poll.start()

    def check_install_done(self):
        if self.install_proc and self.install_proc.poll() is not None:
            self.poll.stop()
            self.install_proc = None
            self.refresh_tools()

    # ------------------------------------------------------------------ inputs
    def add_videos(self):
        start = self.settings.value("last_dir", str(Path.home()))
        files, _ = QFileDialog.getOpenFileNames(self, "Add videos", start, VIDEO_FILTER)
        if files:
            self.settings.setValue("last_dir", str(Path(files[0]).parent))
            self.vlist.add_paths(files)

    def add_folder(self):
        start = self.settings.value("last_dir", str(Path.home()))
        d = QFileDialog.getExistingDirectory(self, "Add every video in a folder", start)
        if d:
            self.settings.setValue("last_dir", d)
            self.vlist.add_paths([d])

    def pick_output(self):
        d = QFileDialog.getExistingDirectory(self, "Choose the project folder",
                                             str(nearest_existing(Path(self.out_edit.text() or "~"))))
        if d:
            self.out_edit.setText(d)

    def open_output(self):
        p = nearest_existing(Path(self.out_edit.text() or "~"))
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))

    def meta_for(self, path):
        if path not in self.meta:
            t = self.tools
            self.meta[path] = probe_video(t.ffprobe, path) if t and t.ffprobe else None
        return self.meta[path]

    def update_summary(self):
        paths = self.vlist.paths()
        total = 0
        unknown = 0
        for p in paths:
            n = frames_for(self.meta_for(p), self.fps_spin.value())
            if n:
                total += n
            else:
                unknown += 1
        txt = f"{len(paths)} video(s)"
        if paths:
            txt += f" · ≈ {total:,} frames" + (f" (+{unknown} unknown)" if unknown else "")
        try:
            free = shutil.disk_usage(nearest_existing(Path(self.out_edit.text() or "~"))).free
            txt += f" · free space for the project: {fmt_bytes(free)}"
        except OSError:
            pass
        self.summary.setText(txt)

    # ------------------------------------------------------------------ estimate
    def vf_string(self):
        parts = []
        if self.fps_spin.value() > 0:
            parts.append(f"fps={self.fps_spin.value()}")
        if self.width_spin.value() > 0:
            # never upscale; keep width even (JPEG chroma subsampling dislikes odd sizes)
            parts.append(f"scale='trunc(min({self.width_spin.value()},iw)/2)*2':-2")
        return ",".join(parts)

    def output_dir(self):
        text = self.out_edit.text().strip()
        return Path(text).expanduser().absolute() if text else None

    def plan_jobs(self, out):
        """One job per video with a unique scene folder name (shared by the estimate and the run)."""
        jobs, used = [], set()
        for v in self.vlist.paths():
            base = name = Path(v).stem
            k = 2
            while name in used:
                name = f"{base}_{k}"
                k += 1
            used.add(name)
            jobs.append(SimpleNamespace(video=v, name=Path(v).name, scene=out / name, status="pending",
                                        est=frames_for(self.meta_for(v), self.fps_spin.value())))
        return jobs

    def is_solved(self, job):
        model = job.scene / "sparse" / "0"
        return model.is_dir() and any(model.iterdir())

    def estimate(self, show=False):
        """Return estimated *peak* disk use in bytes (respecting cleanup/skip settings), or None.
        Samples one real frame per video at the chosen settings."""
        t = self.tools
        out = self.output_dir()
        if not (t and t.ffmpeg and t.ffprobe and self.vlist.paths() and out):
            if show:
                self.est_label.setText("Choose a project folder, add videos, and make sure ffmpeg is installed.")
            return None
        q = self.quality.currentData()
        vf = self.vf_string()
        per, n_frames, skipped = [], 0, 0
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            for job in self.plan_jobs(out):
                if self.chk_skip.isChecked() and self.is_solved(job):
                    skipped += 1
                    continue
                meta = self.meta_for(job.video)
                n = job.est
                if not n:
                    continue
                with tempfile.TemporaryDirectory() as td:
                    sample = Path(td) / "s.jpg"
                    cmd = [t.ffmpeg, "-y", "-loglevel", "error", "-ss", str((meta["dur"] or 0) / 2),
                           "-i", job.video]
                    if vf:
                        cmd += ["-vf", vf]
                    cmd += ["-frames:v", "1", "-qscale:v", str(q), str(sample)]
                    run_quiet(cmd, timeout=60)
                    if sample.exists():
                        per.append((n * sample.stat().st_size, n * DB_BYTES_PER_FRAME))
                        n_frames += n
        finally:
            QApplication.restoreOverrideCursor()
        if not per:
            if show:
                self.est_label.setText("Nothing to estimate (all videos already solved?)." if skipped
                                       else "Couldn't sample the videos.")
            return None
        fr, dbs = [a for a, _ in per], [b for _, b in per]
        # Deleted-after-each-video data only needs room for the biggest single video at a time.
        peak = (max(fr) if self.chk_del_frames.isChecked() else sum(fr)) + \
               (max(dbs) if self.chk_del_db.isChecked() else sum(dbs))
        if show:
            try:
                free = fmt_bytes(shutil.disk_usage(nearest_existing(out)).free)
            except OSError:
                free = "?"
            self.est_label.setText(
                f"{len(per)} video(s) to process" + (f" ({skipped} already solved, skipped)" if skipped else "")
                + f", ≈ {n_frames:,} frames. Totals: frames ≈ {fmt_bytes(sum(fr))}, database up to ≈ "
                f"{fmt_bytes(sum(dbs))}. Peak disk use with your cleanup settings: ≈ {fmt_bytes(peak)}. "
                f"Free on output drive: {free}. (Sampled one frame per video.)")
        return peak

    # ------------------------------------------------------------------ run
    def log(self, text):
        self.log_box.appendPlainText(text)

    def set_running(self, on):
        self.running = on
        self.controls.setEnabled(not on)
        self.btn_start.setEnabled(not on)
        self.btn_cancel.setEnabled(on)

    def start(self):
        t = self.tools
        mapper = mapper_cmd(t) if t else None
        if not (t and t.ffmpeg and t.colmap and mapper):
            QMessageBox.warning(self, "AutoTracker",
                                "Setup is incomplete — ffmpeg and COLMAP 4.0 or newer are required.\n"
                                "Click 'Set up…' in the Setup box.")
            return
        videos = self.vlist.paths()
        if not videos:
            QMessageBox.information(self, "AutoTracker", "Add at least one video first.")
            return
        out = self.output_dir()
        if not out:
            QMessageBox.warning(self, "AutoTracker", "Choose a project folder first.")
            return
        try:
            out.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            QMessageBox.warning(self, "AutoTracker", f"Can't use that project folder:\n{e}")
            return

        need = self.estimate()
        if need:
            free = shutil.disk_usage(out).free
            if need > free * 0.95:
                if QMessageBox.question(
                        self, "AutoTracker",
                        f"Estimated peak space needed: {fmt_bytes(need)}\nFree on that drive: "
                        f"{fmt_bytes(free)}\n\nThis may not fit. Lower the fps/width, tick the delete "
                        f"options, or pick another drive.\n\nContinue anyway?") != QMessageBox.StandardButton.Yes:
                    return

        self._save_settings()
        self.jobs = self.plan_jobs(out)
        self.steps = self.build_steps(mapper)
        self.run_id += 1
        self.idx, self.dead, self.cancelled = 0, set(), False
        self.progress.setMaximum(len(self.steps))
        self.progress.setValue(0)
        self.log_box.clear()
        self.log(f"Project: {out}")
        self.log(f"Mapper: {' '.join(mapper)}   GPU: {'on' if self.chk_gpu.isChecked() else 'off'}")
        if self.chk_gpu.isChecked() and t.cuda is False:
            self.log("⚠ GPU is on but your COLMAP reports no CUDA — if feature extraction fails, untick GPU.")
        self.set_running(True)
        self.next_step()

    def build_steps(self, mapper):
        t, S = self.tools, []
        gpu = "1" if self.chk_gpu.isChecked() else "0"
        vf = self.vf_string()
        for i, job in enumerate(self.jobs):
            sc = job.scene
            db, img, sp = str(sc / "database.db"), str(sc / "images"), str(sc / "sparse")
            S.append(dict(job=i, label="preparing", fn=self.step_prepare))
            ff = [t.ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-nostats", "-progress", "pipe:1",
                  "-i", job.video]
            if vf:
                ff += ["-vf", vf]
            # ffmpeg treats '%' anywhere in an image-sequence path as a pattern -> escape the folder part
            pattern = str(sc / "images").replace("%", "%%") + "/frame_%06d.jpg"
            ff += ["-qscale:v", str(self.quality.currentData()), pattern]
            S.append(dict(job=i, label="extracting frames", kind="ffmpeg", prog=ff[0], args=ff[1:]))
            S.append(dict(job=i, label="checking frames", fn=self.step_check_frames))
            S.append(dict(job=i, label="detecting features", kind="colmap", prog=t.colmap, args=[
                "feature_extractor", "--database_path", db, "--image_path", img,
                "--ImageReader.single_camera", "1",
                f"--{t.opt_feat_gpu}", gpu,
                f"--{t.opt_feat_size}", str(self.sift_spin.value())]))
            S.append(dict(job=i, label="matching frames", kind="colmap", prog=t.colmap, args=[
                "sequential_matcher", "--database_path", db,
                "--SequentialMatching.overlap", str(self.overlap_spin.value()),
                f"--{t.opt_match_gpu}", gpu]))
            S.append(dict(job=i, label="solving cameras", kind="colmap", prog=mapper[0], args=mapper[1:] + [
                "--database_path", db, "--image_path", img, "--output_path", sp]))
            S.append(dict(job=i, label="checking result", fn=self.step_check_model))
            for tgt in (str(sc / "sparse" / "0"), sp):
                S.append(dict(job=i, label="exporting TXT", kind="colmap", prog=t.colmap, args=[
                    "model_converter", "--input_path", str(sc / "sparse" / "0"),
                    "--output_path", tgt, "--output_type", "TXT"]))
            S.append(dict(job=i, label="cleaning up", fn=self.step_cleanup))
        return S

    # ---- python-side steps: return "ok" | "skip" | "fail"
    def step_prepare(self, job):
        sc = job.scene
        if sc.exists() and not sc.is_dir():
            self.log(f"✗ {job.name}: '{sc}' exists and is a file, not a folder.")
            return "fail"
        if self.chk_skip.isChecked() and self.is_solved(job):
            self.log(f"• {job.name}: already solved — skipping")
            return "skip"
        marker = sc / ".autotracker"
        if sc.exists() and not marker.exists() and any(sc.iterdir()):
            self.log(f"✗ {job.name}: '{sc}' already exists and wasn't created by this app. "
                     f"Pick another project folder or rename that folder.")
            return "fail"
        try:
            sc.mkdir(parents=True, exist_ok=True)
            marker.touch()
            shutil.rmtree(sc / "images", ignore_errors=True)
            shutil.rmtree(sc / "sparse", ignore_errors=True)
            (sc / "database.db").unlink(missing_ok=True)
            (sc / "images").mkdir()
            (sc / "sparse").mkdir()
        except OSError as e:
            self.log(f"✗ {job.name}: can't prepare folder: {e}")
            return "fail"
        return "ok"

    def step_check_frames(self, job):
        if any((job.scene / "images").glob("*.jpg")):
            return "ok"
        self.log(f"✗ {job.name}: no frames were extracted (is it a valid video?)")
        return "fail"

    def step_check_model(self, job):
        model = job.scene / "sparse" / "0"
        if model.is_dir() and any(model.iterdir()):
            return "ok"
        self.log(f"✗ {job.name}: the solver produced no model. Usually the shot has too little camera "
                 f"movement/parallax, heavy motion blur, or too few distinct features.")
        return "fail"

    def step_cleanup(self, job):
        if self.chk_del_frames.isChecked():
            shutil.rmtree(job.scene / "images", ignore_errors=True)
            self.log(f"  removed extracted frames for {job.name}")
        if self.chk_del_db.isChecked():
            (job.scene / "database.db").unlink(missing_ok=True)
            self.log(f"  removed database for {job.name}")
        job.status = "done"
        self.log(f"✓ {job.name} → {job.scene / 'sparse'}")
        return "ok"

    # ---- state machine
    def next_step(self):
        if not self.running:
            return
        if self.cancelled:
            return self.finish()
        while self.idx < len(self.steps) and self.steps[self.idx]["job"] in self.dead:
            self.idx += 1
        if self.idx >= len(self.steps):
            return self.finish()
        st = self.steps[self.idx]
        job = self.jobs[st["job"]]
        self.progress.setValue(self.idx)
        self.set_status(job, st["label"])
        if "fn" in st:
            try:
                res = st["fn"](job)
            except Exception as e:  # keep the queue alive on unexpected errors
                self.log(f"✗ {job.name}: internal error: {e}")
                res = "fail"
            self.idx += 1
            if res == "skip":
                job.status = "skipped"
                self.dead.add(st["job"])
            elif res == "fail":
                job.status = "failed"
                self.dead.add(st["job"])
            QTimer.singleShot(0, self.next_step)
            return
        self.cur_kind, self._buf = st["kind"], ""
        self._dec.reset()
        self.log(f"\n[{st['job'] + 1}/{len(self.jobs)}] {job.name} — {st['label']}")
        self.log("$ " + " ".join(f'"{a}"' if re.search(r"\s", a) else a for a in [st["prog"]] + st["args"]))
        self.proc.start(st["prog"], st["args"])

    def set_status(self, job, label, extra=""):
        n = self.jobs.index(job) + 1
        self.status.setText(f"[{n}/{len(self.jobs)}] {job.name}: {label}{extra}")

    def on_output(self):
        # incremental decoder: a multi-byte character split across two reads isn't garbled
        data = self._dec.decode(bytes(self.proc.readAll()))
        self._buf += data.replace("\r", "\n")
        *lines, self._buf = self._buf.split("\n")
        for line in lines:
            self.handle_line(line)

    def handle_line(self, line):
        line = ANSI_RE.sub("", line).rstrip()
        if not line:
            return
        if self.cur_kind == "ffmpeg" and FFMPEG_PROGRESS_RE.match(line):
            if line.startswith("frame=") and self.idx < len(self.steps):
                job = self.jobs[self.steps[self.idx]["job"]]
                est = f"/{job.est:,}" if job.est else ""
                self.set_status(job, "extracting frames", f"  {line[6:].strip()}{est}")
            return
        self.log(line)

    def on_finished(self, code, status):
        if self._buf.strip():
            self.handle_line(self._buf)
        self._buf = ""
        if self.cancelled or not self.running:
            return self.finish()
        st = self.steps[self.idx]
        if status != QProcess.ExitStatus.NormalExit or code != 0:
            job = self.jobs[st["job"]]
            hint = " ffmpeg couldn't read this file — is it a valid, complete video?" \
                if st.get("kind") == "ffmpeg" else ""
            self.log(f"✗ {job.name}: '{st['label']}' failed (exit code {code}).{hint} Intermediate files "
                     f"are kept in {job.scene} for inspection.")
            job.status = "failed"
            self.dead.add(st["job"])
        self.idx += 1
        QTimer.singleShot(0, self.next_step)

    def on_proc_error(self, err):
        if err == QProcess.ProcessError.FailedToStart and self.running and self.idx < len(self.steps):
            st = self.steps[self.idx]
            job = self.jobs[st["job"]]
            self.log(f"✗ could not start {st['prog']}: {self.proc.errorString()}")
            job.status = "failed"
            self.dead.add(st["job"])
            self.idx += 1
            QTimer.singleShot(0, self.next_step)

    def cancel(self):
        if not self.running:
            return
        self.cancelled = True
        self.log("\nCancelling…")
        if self.idx < len(self.steps):
            job = self.jobs[self.steps[self.idx]["job"]]
            if job.status == "pending":
                job.status = "cancelled"
        if self.proc.state() != QProcess.ProcessState.NotRunning:
            self.proc.terminate()
            rid = self.run_id  # only force-kill the process of *this* run, never a newly started one
            QTimer.singleShot(3000, lambda: rid == self.run_id
                              and self.proc.state() != QProcess.ProcessState.NotRunning
                              and self.proc.kill())
        else:
            self.finish()

    def finish(self):
        if not self.running:
            return
        self.set_running(False)
        c = {k: sum(1 for j in self.jobs if j.status == k) for k in ("done", "skipped", "failed", "cancelled")}
        left = sum(1 for j in self.jobs if j.status == "pending")
        self.progress.setValue(self.progress.maximum() if not self.cancelled else self.idx)
        head = "Cancelled" if self.cancelled else "Finished"
        msg = f"{head}: {c['done']} solved, {c['skipped']} skipped, {c['failed']} failed"
        if c["cancelled"]:
            msg += f", {c['cancelled']} stopped part-way (redone on the next run)"
        if left:
            msg += f", {left} not run"
        self.status.setText(msg + ".")
        self.log("\n" + "=" * 60 + f"\n{msg}.")
        if c["failed"]:
            self.log("Failed:  " + ", ".join(j.name for j in self.jobs if j.status == "failed"))
        self._save_settings()


def main():
    signal.signal(signal.SIGINT, signal.SIG_DFL)  # let Ctrl+C in the launching terminal close the app
    app = QApplication(sys.argv)
    app.setApplicationName("AutoTracker GUI")
    w = Main()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
