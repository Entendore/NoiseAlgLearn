"""
Noise Lab — a single-file interactive course on 2-D noise algorithms.
=====================================================================

Run:     pip install PySide6 numpy
         python noise_lab.py

Contents
--------
· A complete educational noise library (bottom half of this file):
  white / value / Perlin / simplex / Worley / Gabor / spectral 1/f^β,
  analytic derivatives, 3-D variants, seamless (periodic) noise,
  hybrid multifractal, octave rotation, domain warping.
· A PySide6 "lab" around it: 34 guided lessons in 5 parts, A/B compare,
  z-slice animation, lattice & feature overlays, dot-product probe,
  cell X-ray, power spectrum, fade-curve plot, histogram, tiling proof,
  octave strip, flow streamlines, relief / normal-map / contour /
  terrace rendering, 7 colormaps.

The course:
  Part I   Foundations .............. lessons 1–6
  Part II  Fractals & motion ........ lessons 7–11
  Part III Inside the machine ....... lessons 12–19
  Part IV  Algorithm engineering .... lessons 20–24
  Part V   Skills & recipes ......... lessons 25–34
"""

from __future__ import annotations

import math
import random
import sys

import numpy as np
from PySide6.QtCore import Qt, QPointF, QRectF, QTimer, Signal
from PySide6.QtGui import (QBrush, QColor, QFont, QImage, QPainter,
                           QPainterPath, QPalette, QPen)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFrame,
                               QGroupBox, QHBoxLayout, QLabel, QMainWindow,
                               QPushButton, QScrollArea, QSlider, QSizePolicy,
                               QSpinBox, QSplitter, QTextBrowser, QVBoxLayout,
                               QWidget)

# =====================================================================
#  Option lists
# =====================================================================

RESOLUTIONS = [128, 192, 256, 384, 512]

ALGORITHMS = [
    ("Value noise", "value"),
    ("Perlin noise", "perlin"),
    ("Simplex noise", "simplex"),
    ("Worley / cellular", "worley"),
    ("Gabor (band-pass)", "gabor"),
    ("Spectral 1/f^β", "spectral"),
    ("White noise", "white"),
]
FRACTALS = [
    ("Single octave", "single"),
    ("fBm  (sum of octaves)", "fbm"),
    ("Turbulence  (folded |n|)", "turbulence"),
    ("Ridged multifractal", "ridged"),
    ("Billow", "billow"),
    ("Hybrid multifractal (Musgrave)", "hybrid"),
]
FADES = [
    ("Linear  —  C⁰, grid artifacts visible", "linear"),
    ("Smoothstep  —  C¹", "smooth"),
    ("Quintic  —  C²  (Perlin's improved fade)", "quintic"),
]
METRICS = [("Euclidean", "euclidean"), ("Manhattan", "manhattan"),
           ("Chebyshev", "chebyshev")]
CELLMODES = [
    ("F1 — distance to nearest", "f1"),
    ("F2 − F1 — cell edges", "f2f1"),
    ("F2 — second-nearest", "f2"),
    ("Regions — Voronoi id (per-cell random)", "id"),
]
GRADSETS = [
    ("Classic 8 — 4 diag + 4 axis", "classic8"),
    ("16 even directions", "dirs16"),
    ("Random angle — isotropic", "circle"),
]
SHADE_MODES = [("Flat", "flat"), ("Relief — hillshade", "relief"),
               ("Normal map — RGB", "normals")]

CHAPTERS = [
    (1,  "Part I · Foundations"),
    (7,  "Part II · Fractals & motion"),
    (12, "Part III · Inside the machine"),
    (20, "Part IV · Algorithm engineering"),
    (25, "Part V · Skills & recipes"),
]

_ALGO_FN3 = {
    "value": lambda: None,   # replaced below after the noise library loads
}
_ALGO_NAMES = dict(ALGORITHMS)

# =====================================================================
#  Colormaps
# =====================================================================

_COLORMAP_STOPS = {
    "Grayscale": [(0, 0, 0), (1, 1, 1)],
    "Viridis": [(0.267, 0.005, 0.329), (0.283, 0.141, 0.458), (0.254, 0.265, 0.530),
                (0.207, 0.372, 0.553), (0.164, 0.471, 0.558), (0.128, 0.567, 0.551),
                (0.135, 0.659, 0.518), (0.267, 0.749, 0.442), (0.478, 0.821, 0.318),
                (0.741, 0.873, 0.150), (0.993, 0.906, 0.144)],
    "Magma": [(0.001, 0.000, 0.014), (0.056, 0.043, 0.180), (0.135, 0.065, 0.275),
              (0.226, 0.077, 0.338), (0.326, 0.081, 0.365), (0.440, 0.084, 0.357),
              (0.550, 0.110, 0.315), (0.660, 0.180, 0.243), (0.770, 0.290, 0.170),
              (0.880, 0.460, 0.090), (0.987, 0.645, 0.038), (0.987, 0.991, 0.750)],
    "Plasma": [(0.050, 0.030, 0.528), (0.253, 0.011, 0.631), (0.390, 0.027, 0.665),
               (0.522, 0.071, 0.664), (0.648, 0.130, 0.628), (0.762, 0.195, 0.567),
               (0.862, 0.271, 0.486), (0.941, 0.361, 0.398), (0.988, 0.471, 0.296),
               (0.986, 0.609, 0.157), (0.940, 0.975, 0.131)],
    "Terrain": [(0.02, 0.04, 0.24), (0.00, 0.38, 0.60), (0.05, 0.62, 0.35),
                (0.45, 0.56, 0.20), (0.48, 0.32, 0.16), (0.93, 0.92, 0.86)],
    "Glacier": [(0.02, 0.04, 0.10), (0.04, 0.16, 0.34), (0.08, 0.36, 0.58),
                (0.30, 0.64, 0.80), (0.70, 0.90, 0.96), (1.0, 1.0, 1.0)],
    "Diverging": [(0.23, 0.30, 0.62), (0.55, 0.73, 0.89), (0.96, 0.97, 0.98),
                  (0.94, 0.62, 0.42), (0.62, 0.16, 0.18)],
}


def _build_lut(stops):
    n = len(stops)
    xs = np.linspace(0.0, 1.0, n)
    t = np.linspace(0.0, 1.0, 256)
    lut = np.zeros((256, 3), dtype=np.uint8)
    for c in range(3):
        vals = np.interp(t, xs, [float(s[c]) for s in stops])
        lut[:, c] = np.clip(vals * 255.0, 0, 255).astype(np.uint8)
    return lut


_COLORMAPS = {name: _build_lut(stops) for name, stops in _COLORMAP_STOPS.items()}


# =====================================================================
#  Rendering: hillshade, contours, terrace, normal maps
# =====================================================================

def hillshade(field, azim=315.0, elev=50.0, z_boost=1.0):
    """Lambert shading from the field's surface normals."""
    gy, gx = np.gradient(field)
    nx_, ny_, nz = -gx * z_boost, -gy * z_boost, np.ones_like(field)
    az, el = np.radians(azim), np.radians(elev)
    lx, ly, lz = np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)
    n = np.sqrt(nx_ * nx_ + ny_ * ny_ + nz * nz)
    return np.clip((nx_ * lx + ny_ * ly + nz * lz) / n, 0.0, 1.0)


def render_field(field, lut, shade="flat", contours=0, pixel=1.0, terrace=0):
    """Field -> QImage.
    shade   : 'flat' | 'relief' (Lambert hillshade) | 'normals' (RGB-encoded
              tangent-space normal map — the standard game-asset encoding)
    terrace : >=2 quantises the field into flat steps before shading
    pixel   : noise units per pixel — keeps shading frequency-correct.
    """
    f = field
    if terrace >= 2:
        f = np.floor(f * terrace) / (terrace - 1)
    idx = np.clip(f * 255.0, 0, 255).astype(np.uint8)
    rgb = lut[idx].astype(np.float32)

    if shade == "relief":
        s = hillshade(f, z_boost=1.4 / max(pixel, 1e-9))
        rgb *= np.clip(0.30 + 0.85 * s, 0.0, 1.15)[..., None]
    elif shade == "normals":
        gy, gx = np.gradient(f)
        nx_, ny_, nz = -gx / max(pixel, 1e-9), -gy / max(pixel, 1e-9), np.ones_like(f)
        n = np.sqrt(nx_**2 + ny_**2 + nz**2)
        rgb = np.stack([(nx_ / n * 0.5 + 0.5),
                        (ny_ / n * 0.5 + 0.5),
                        (nz / n * 0.5 + 0.5)], axis=-1) * 255.0

    if contours >= 2:
        d = np.abs(np.mod(f * contours, 1.0))
        d = np.minimum(d, 1.0 - d) / contours
        alpha = np.clip(1.0 - d / 0.010, 0.0, 1.0) ** 1.2
        rgb *= (1.0 - 0.55 * alpha[..., None])

    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    h, w = idx.shape
    img = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return img.copy()


def field_to_image(field, lut):
    return render_field(field, lut)


# =====================================================================
#  NOISE LIBRARY
#  Every generator returns values in [0, 1] and is deterministic for a
#  given (x, y, seed). All functions are vectorised.
# =====================================================================

SIMPLEX_G2 = 0.21132486540518713        # (3 - sqrt(3)) / 6 — 2-D skew factor
_WORLEY_SEED2 = 17                      # second random stream for Worley y

_GRAD2 = np.array([
    [ 1.0,  1.0], [-1.0,  1.0], [ 1.0, -1.0], [-1.0, -1.0],
    [ 1.0,  0.0], [-1.0,  0.0], [ 0.0,  1.0], [ 0.0, -1.0],
])

_GRAD3 = np.array([
    [ 1, 1, 0], [-1, 1, 0], [ 1,-1, 0], [-1,-1, 0],
    [ 1, 0, 1], [-1, 0, 1], [ 1, 0,-1], [-1, 0,-1],
    [ 0, 1, 1], [ 0,-1, 1], [ 0, 1,-1], [ 0,-1,-1],
], dtype=np.float64)                     # cube edge midpoints (Perlin's 3-D set)

_DIRS16 = np.stack([np.cos(np.arange(16) * (np.pi / 8.0)),
                    np.sin(np.arange(16) * (np.pi / 8.0))], axis=-1)


# ------------------------------------------------------------ utilities

def _hash32(ix, iy, seed=0) -> np.ndarray:
    """Deterministic 32-bit integer hash of lattice coordinates -> uint32."""
    ix = np.asarray(ix, dtype=np.int64) & 0xFFFFFFFF
    iy = np.asarray(iy, dtype=np.int64) & 0xFFFFFFFF
    h = (ix.astype(np.uint32) * np.uint32(0x27D4EB2D)) \
      ^ (iy.astype(np.uint32) * np.uint32(0x165667B1)) \
      ^ (np.uint32(seed & 0xFFFFFFFF) * np.uint32(0x9E3779B1))
    h ^= h >> np.uint32(15)
    h *= np.uint32(0x85EBCA6B)
    h ^= h >> np.uint32(13)
    h *= np.uint32(0xC2B2AE35)
    h ^= h >> np.uint32(16)
    return h


def _rand01(ix, iy, seed=0):
    return _hash32(ix, iy, seed).astype(np.float64) / 4294967296.0


def _hash3(ix, iy, iz, seed=0) -> np.ndarray:
    ix = np.asarray(ix, dtype=np.int64) & 0xFFFFFFFF
    iy = np.asarray(iy, dtype=np.int64) & 0xFFFFFFFF
    iz = np.asarray(iz, dtype=np.int64) & 0xFFFFFFFF
    h = (ix.astype(np.uint32) * np.uint32(0x27D4EB2D)) \
      ^ (iy.astype(np.uint32) * np.uint32(0x165667B1)) \
      ^ (iz.astype(np.uint32) * np.uint32(0x9E3779B1)) \
      ^ (np.uint32(seed & 0xFFFFFFFF) * np.uint32(0x85EBCA77))
    h ^= h >> np.uint32(15); h *= np.uint32(0x85EBCA6B)
    h ^= h >> np.uint32(13); h *= np.uint32(0xC2B2AE35)
    h ^= h >> np.uint32(16)
    return h


def _rand3(ix, iy, iz, seed=0):
    return _hash3(ix, iy, iz, seed).astype(np.float64) / 4294967296.0


def _fade(t, mode="quintic"):
    """Easing of the fractional position between lattice points.
    THE key idea behind interpolation noise: this curve controls how sharply
    the field bends as it crosses each cell boundary."""
    if mode == "linear":
        return t                                   # C0 — visible artifacts
    if mode == "smooth":
        return t * t * (3.0 - 2.0 * t)             # C1 — smoothstep
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)  # C2 — Perlin's quintic


def _fade_d(t, mode="quintic"):
    """Derivative of the fade curve w.r.t. fractional position."""
    if mode == "linear":
        return np.ones_like(t)
    if mode == "smooth":
        return 6.0 * t * (1.0 - t)
    return 30.0 * t * t * (t - 1.0) ** 2           # d/dt t³(10−15t+6t²)


def _gradient2(ix, iy, seed=0, gradient_set="classic8"):
    """The gradient vector stored at a lattice corner, per chosen set.
    classic8: Perlin's classic 8 (diagonal bias) · dirs16: 16 even
    directions · circle: continuous random angle (fully isotropic)."""
    if gradient_set == "circle":
        a = _rand01(ix, iy, seed) * 2.0 * np.pi
        return np.stack([np.cos(a), np.sin(a)], axis=-1)
    table = _GRAD2 if gradient_set == "classic8" else _DIRS16
    return table[_hash32(ix, iy, seed) % len(table)]


# ---------------------------------------------------- lattice introspection
# These let the UI *draw* what the algorithms store.

def lattice_value(ix, iy, seed=0):
    """The random value stored at a lattice corner (value noise)."""
    return float(_rand01(int(ix), int(iy), seed))


def lattice_gradient(ix, iy, seed=0, gradient_set="classic8"):
    """The gradient vector stored at a lattice corner (Perlin / Simplex)."""
    return _gradient2(int(ix), int(iy), seed, gradient_set)


def lattice_feature(ix, iy, seed=0):
    """The feature point inside a Worley cell (points=1, jitter=1)."""
    return (int(ix) + float(_rand01(int(ix), int(iy), seed)),
            int(iy) + float(_rand01(int(ix), int(iy), seed + _WORLEY_SEED2)))


# ------------------------------------------------------- 2-D generators

def white_noise(shape, seed=0):
    """Independent uniform random values — pure static, zero correlation."""
    return np.random.default_rng(seed).random(shape)


def value_noise(x, y, seed=0, fade="quintic", period=None):
    """Value noise: a random value on every lattice corner; each sample
    blends its four surrounding corners via the fade curve.
    Pass an integer `period` to make it wrap seamlessly."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xi = np.floor(x); yi = np.floor(y)
    fx = x - xi; fy = y - yi
    u = _fade(fx, fade); v = _fade(fy, fade)
    ix = xi.astype(np.int64); iy = yi.astype(np.int64)
    if period is not None:
        p = int(period)
        i0, i1 = ix % p, (ix + 1) % p
        j0, j1 = iy % p, (iy + 1) % p
    else:
        i0, i1, j0, j1 = ix, ix + 1, iy, iy + 1
    n00 = _rand01(i0, j0, seed); n10 = _rand01(i1, j0, seed)
    n01 = _rand01(i0, j1, seed); n11 = _rand01(i1, j1, seed)
    nx0 = n00 + u * (n10 - n00)
    nx1 = n01 + u * (n11 - n01)
    return nx0 + v * (nx1 - nx0)


def value_noise_d(x, y, seed=0, fade="quintic"):
    """Value noise with analytic derivatives: (n, ∂n/∂x, ∂n/∂y)."""
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    xi = np.floor(x); yi = np.floor(y)
    fx = x - xi; fy = y - yi
    u, v = _fade(fx, fade), _fade(fy, fade)
    du, dv = _fade_d(fx, fade), _fade_d(fy, fade)
    ix = xi.astype(np.int64); iy = yi.astype(np.int64)
    n = np.zeros_like(x); gx = np.zeros_like(x); gy = np.zeros_like(x)
    for a in (0, 1):
        for b in (0, 1):
            c = _rand01(ix + a, iy + b, seed)
            wx = u if a else 1.0 - u
            wy = v if b else 1.0 - v
            dwx = du if a else -du
            dwy = dv if b else -dv
            n += wx * wy * c
            gx += dwx * wy * c
            gy += dwy * wx * c
    return n, gx, gy


def perlin_noise(x, y, seed=0, fade="quintic", period=None, gradient_set="classic8"):
    """Perlin gradient noise: lattice corners store gradient vectors; each
    sample blends the four dot products (gradient . offset) with the fade.
    integer `period` => seamless tiling; `gradient_set` selects the corner-
    vector distribution (classic 8 has a diagonal bias)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xi = np.floor(x); yi = np.floor(y)
    fx = x - xi; fy = y - yi
    u = _fade(fx, fade); v = _fade(fy, fade)
    ix = xi.astype(np.int64); iy = yi.astype(np.int64)
    p = None if period is None else int(period)

    def corner(ci, cj, dx, dy):
        if p is not None:
            ci = ci % p; cj = cj % p
        g = _gradient2(ci, cj, seed, gradient_set)
        return g[..., 0] * dx + g[..., 1] * dy

    n00 = corner(ix,     iy,     fx,       fy)
    n10 = corner(ix + 1, iy,     fx - 1.0, fy)
    n01 = corner(ix,     iy + 1, fx,       fy - 1.0)
    n11 = corner(ix + 1, iy + 1, fx - 1.0, fy - 1.0)
    n = (n00 + u * (n10 - n00)) + v * ((n01 + u * (n11 - n01)) -
                                       (n00 + u * (n10 - n00)))
    return np.clip(0.5 + 0.5 * n, 0.0, 1.0)


def perlin_noise_d(x, y, seed=0, fade="quintic", gradient_set="classic8"):
    """Perlin noise WITH ANALYTIC DERIVATIVES: returns (n, ∂n/∂x, ∂n/∂y).
    Exact to machine precision — the basis of erosion, flow maps and bump
    mapping. Product rule:  d/dx Σ w·(g·d) = Σ (dw/dx)(g·d) + w·g.x"""
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    xi = np.floor(x); yi = np.floor(y)
    fx = x - xi; fy = y - yi
    u, v = _fade(fx, fade), _fade(fy, fade)
    du, dv = _fade_d(fx, fade), _fade_d(fy, fade)
    ix = xi.astype(np.int64); iy = yi.astype(np.int64)
    n = np.zeros_like(x); gx = np.zeros_like(x); gy = np.zeros_like(x)
    for a in (0, 1):
        for b in (0, 1):
            g = _gradient2(ix + a, iy + b, seed, gradient_set)
            ddx, ddy = fx - a, fy - b
            c = g[..., 0] * ddx + g[..., 1] * ddy
            wx = u if a else 1.0 - u
            wy = v if b else 1.0 - v
            dwx = du if a else -du
            dwy = dv if b else -dv
            n += wx * wy * c
            gx += dwx * wy * c + wx * wy * g[..., 0]
            gy += dwy * wx * c + wx * wy * g[..., 1]
    return (np.clip(0.5 + 0.5 * n, 0, 1), 0.5 * gx, 0.5 * gy)


def simplex_noise(x, y, seed=0):
    """2-D simplex noise (Perlin 2001, Gustavson-style). The square lattice
    is skewed into triangles; each sample blends 3 radial kernels
    t⁴·(g·d) with t = 0.5 − r².  -> [0, 1]"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    F2 = 0.36602540378443865        # (sqrt(3) - 1) / 2
    G2 = SIMPLEX_G2
    s = (x + y) * F2
    i = np.floor(x + s); j = np.floor(y + s)
    t = (i + j) * G2
    x0 = x - (i - t); y0 = y - (j - t)          # offsets in unskewed space
    i1 = np.where(x0 > y0, 1.0, 0.0)            # which triangle are we in?
    j1 = 1.0 - i1
    i1i = i1.astype(np.int64); j1i = j1.astype(np.int64)
    ii = i.astype(np.int64); jj = j.astype(np.int64)

    total = np.zeros_like(x)
    for dx, dy, ci, cj in (
        (x0,              y0,              ii,       jj),
        (x0 - i1 + G2,    y0 - j1 + G2,    ii + i1i, jj + j1i),
        (x0 - 1.0 + 2*G2, y0 - 1.0 + 2*G2, ii + 1,   jj + 1),
    ):
        tt = 0.5 - dx*dx - dy*dy                 # radial falloff kernel
        g = _GRAD2[_hash32(ci, cj, seed) & 7]
        contrib = tt**4 * (g[..., 0] * dx + g[..., 1] * dy)
        total += np.where(tt > 0.0, contrib, 0.0)
    return np.clip(0.5 + 35.0 * total, 0.0, 1.0)


def worley_noise(x, y, seed=0, metric="euclidean", cell="f1", period=None,
                 points=1, jitter=1.0):
    """Generalised Worley / cellular noise: `points` feature points per cell,
    `jitter`∈[0,1] slides them from the cell centre (0 = regular grid →
    textbook Voronoi) to fully random (1). integer `period` => seamless on
    the torus. cell='id' returns a per-region random value — a Voronoi
    region map. F1 is rescaled by 1/√points so ranges stay comparable."""
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    xi = np.floor(x); yi = np.floor(y)
    best = np.full(x.shape, np.inf)
    second = np.full(x.shape, np.inf)
    best_id = np.zeros(x.shape)
    p = None if period is None else int(period)
    pts = max(1, int(points))
    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            ci = (xi + ox).astype(np.int64)
            cj = (yi + oy).astype(np.int64)
            hi, hj = (ci, cj) if p is None else (ci % p, cj % p)
            for k in range(pts):
                px = ci + 0.5 + jitter * (_rand01(hi, hj, seed + 31337 * k) - 0.5)
                py = cj + 0.5 + jitter * (_rand01(hi, hj, seed + 51121 * k + 17) - 0.5)
                if p is not None:
                    px = px % p; py = py % p
                dx = np.abs(px - x); dy = np.abs(py - y)
                if p is not None:
                    dx = np.minimum(dx, p - dx)
                    dy = np.minimum(dy, p - dy)
                if metric == "euclidean":
                    d = np.hypot(dx, dy)
                elif metric == "manhattan":
                    d = dx + dy
                else:
                    d = np.maximum(dx, dy)
                closer = d < best
                second = np.where(closer, best, np.minimum(second, d))
                best = np.where(closer, d, best)
                best_id = np.where(closer, _rand01(hi, hj, seed + 7919 + k),
                                   best_id)
    s = 1.0 / np.sqrt(pts)
    if cell == "id":
        out = best_id
    elif cell == "f1":
        out = best * s
    elif cell == "f2":
        out = second * s / 1.6
    else:
        out = (second - best) * s / 0.7
    return np.clip(out, 0.0, 1.0)


def gabor_noise(x, y, seed=0, f0=2.0, a=1.6):
    """Gabor noise — a sum of Gaussian-windowed cosine kernels, each with its
    own random orientation & phase. Band-pass by construction: the spectrum
    is a ring at radius f0 (unlike Perlin, which is broadband)."""
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    xi = np.floor(x); yi = np.floor(y)
    s = np.zeros_like(x); e = np.zeros_like(x)
    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            ci = (xi + ox).astype(np.int64)
            cj = (yi + oy).astype(np.int64)
            th = _rand01(ci, cj, seed) * 2.0 * np.pi            # orientation
            ph = _rand01(ci, cj, seed + 101) * 2.0 * np.pi      # phase
            px = ci + _rand01(ci, cj, seed + 202)
            py = cj + _rand01(ci, cj, seed + 303)
            dx = x - px; dy = y - py
            env = np.exp(-np.pi * a * a * (dx * dx + dy * dy))
            s += env * np.cos(2.0 * np.pi * f0 *
                              (dx * np.cos(th) + dy * np.sin(th)) + ph)
            e += env
    return np.clip(0.5 + 0.8 * s / np.maximum(e, 1e-9), 0.0, 1.0)


def spectral_noise(shape, seed=0, beta=1.0):
    """Spectral synthesis: white noise → FFT → multiply amplitude by f^(−β)
    → IFFT. β is THE knob: 0 = white, 1 = pink (fBm-like), 2 = brown.
    The FFT grid is periodic, so the result tiles seamlessly for free."""
    h, w = shape
    white = np.random.default_rng(seed).random(shape)
    F = np.fft.fft2(white)
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    f = np.hypot(fx, fy)
    f[0, 0] = 1.0
    F *= f ** (-float(beta))
    F[0, 0] = 0.0
    out = np.fft.ifft2(F).real
    z = (out - out.mean()) / max(out.std(), 1e-12)
    return np.clip(z / 2.5 * 0.5 + 0.5, 0.0, 1.0)


# ------------------------------------------------------- 3-D generators

def value_noise3(x, y, z, seed=0, fade="quintic"):
    """Value noise on a cubic lattice — trilinear blend of 8 corner values."""
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    z = np.asarray(z, np.float64) if np.ndim(z) else np.full_like(x, float(z))
    xi, yi, zi = np.floor(x), np.floor(y), np.floor(z)
    fx, fy, fz = x - xi, y - yi, z - zi
    u, v, w = _fade(fx, fade), _fade(fy, fade), _fade(fz, fade)
    xi = xi.astype(np.int64); yi = yi.astype(np.int64); zi = zi.astype(np.int64)
    c = lambda a, b, d: _rand3(xi + a, yi + b, zi + d, seed)
    n0 = c(0, 0, 0) + u * (c(1, 0, 0) - c(0, 0, 0))
    n1 = c(0, 1, 0) + u * (c(1, 1, 0) - c(0, 1, 0))
    n2 = c(0, 0, 1) + u * (c(1, 0, 1) - c(0, 0, 1))
    n3 = c(0, 1, 1) + u * (c(1, 1, 1) - c(0, 1, 1))
    return (n0 + v * (n1 - n0)) + w * ((n2 + v * (n3 - n2)) - (n0 + v * (n1 - n0)))


def perlin_noise3(x, y, z, seed=0, fade="quintic"):
    """Perlin gradient noise on a cubic lattice — 8 corner dot-products."""
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    z = np.asarray(z, np.float64) if np.ndim(z) else np.full_like(x, float(z))
    xi, yi, zi = np.floor(x), np.floor(y), np.floor(z)
    fx, fy, fz = x - xi, y - yi, z - zi
    u, v, w = _fade(fx, fade), _fade(fy, fade), _fade(fz, fade)
    xi = xi.astype(np.int64); yi = yi.astype(np.int64); zi = zi.astype(np.int64)

    def corner(a, b, d, dx, dy, dz):
        g = _GRAD3[_hash3(a, b, d, seed) % 12]
        return g[..., 0] * dx + g[..., 1] * dy + g[..., 2] * dz

    n = np.zeros_like(x)
    for da in (0, 1):
        for db in (0, 1):
            for dd in (0, 1):
                contrib = corner(xi + da, yi + db, zi + dd, fx - da, fy - db, fz - dd)
                n += (u if da else 1 - u) * (v if db else 1 - v) \
                     * (w if dd else 1 - w) * contrib
    return np.clip(0.5 + 0.5 * n, 0.0, 1.0)


def simplex_noise3(x, y, z, seed=0):
    """3-D simplex noise (Gustavson-style). The cubic grid is skewed into a
    tetrahedral mesh; each sample blends 4 radial kernels t⁴·(g·d),
    t = 0.6 − d².  -> [0, 1]"""
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    z = np.asarray(z, np.float64) if np.ndim(z) else np.full_like(x, float(z))
    F3, G3 = 1.0 / 3.0, 1.0 / 6.0
    s = (x + y + z) * F3
    i, j, k = np.floor(x + s), np.floor(y + s), np.floor(z + s)
    t = (i + j + k) * G3
    x0, y0, z0 = x - (i - t), y - (j - t), z - (k - t)

    a = x0 >= y0                       # which tetrahedron are we in?
    ayz, axz = y0 >= z0, x0 >= z0
    blt, bxz = y0 < z0, x0 < z0
    i1 = np.where(a, ayz | axz, 0); j1 = np.where(a, 0, ~blt)
    k1 = np.where(a, ~ayz & ~axz, blt)
    i2 = np.where(a, 1, ~blt & ~bxz); j2 = np.where(a, ayz, 1)
    k2 = np.where(a, ~ayz, 1)
    i1, j1, k1 = (arr.astype(np.int64) for arr in (i1, j1, k1))
    i2, j2, k2 = (arr.astype(np.int64) for arr in (i2, j2, k2))
    ii, jj, kk = i.astype(np.int64), j.astype(np.int64), k.astype(np.int64)

    total = np.zeros_like(x)
    for dx, dy, dz, ci, cj, ck in (
        (x0,          y0,          z0,          ii,      jj,      kk),
        (x0 - i1 + G3, y0 - j1 + G3, z0 - k1 + G3, ii + i1, jj + j1, kk + k1),
        (x0 - i2 + 2*G3, y0 - j2 + 2*G3, z0 - k2 + 2*G3, ii + i2, jj + j2, kk + k2),
        (x0 - 1.0 + 3*G3, y0 - 1.0 + 3*G3, z0 - 1.0 + 3*G3, ii + 1, jj + 1, kk + 1),
    ):
        tt = 0.6 - dx*dx - dy*dy - dz*dz
        g = _GRAD3[_hash3(ci, cj, ck, seed) % 12]
        contrib = (tt * tt) ** 2 * (g[..., 0]*dx + g[..., 1]*dy + g[..., 2]*dz)
        total += np.where(tt > 0.0, contrib, 0.0)
    return np.clip(0.5 + 32.0 * total, 0.0, 1.0)


def worley_noise3(x, y, z, seed=0, metric="euclidean", cell="f1"):
    """Worley on a cubic lattice — searches the 27 neighbouring cells."""
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    z = np.asarray(z, np.float64) if np.ndim(z) else np.full_like(x, float(z))
    xi, yi, zi = np.floor(x), np.floor(y), np.floor(z)
    best = np.full(x.shape, np.inf); second = np.full(x.shape, np.inf)
    best_id = np.zeros(x.shape)
    for oz in (-1, 0, 1):
        for oy in (-1, 0, 1):
            for ox in (-1, 0, 1):
                ci = (xi + ox).astype(np.int64)
                cj = (yi + oy).astype(np.int64)
                ck = (zi + oz).astype(np.int64)
                px = ci + _rand3(ci, cj, ck, seed)
                py = cj + _rand3(ci, cj, ck, seed + 7777)
                pz = ck + _rand3(ci, cj, ck, seed + 15401)
                dx, dy, dz = np.abs(px - x), np.abs(py - y), np.abs(pz - z)
                if metric == "euclidean":
                    d = np.sqrt(dx*dx + dy*dy + dz*dz)
                elif metric == "manhattan":
                    d = dx + dy + dz
                else:
                    d = np.maximum(dx, np.maximum(dy, dz))
                closer = d < best
                second = np.where(closer, best, np.minimum(second, d))
                best = np.where(closer, d, best)
                best_id = np.where(closer, _rand3(ci, cj, ck, seed + 7919), best_id)
    out = {"f1": best, "f2": second / 1.8,
           "f2f1": (second - best) / 0.8, "id": best_id}[cell]
    return np.clip(out, 0.0, 1.0)


# -------------------------------------------------- fractal machinery

def _shape_octave(n, kind):
    """The per-octave shaping shared by all fractal variants."""
    if kind == "fbm":
        return n
    if kind == "turbulence":
        return 2.0 * np.abs(n - 0.5)
    if kind == "ridged":
        return (1.0 - np.abs(2.0 * n - 1.0)) ** 2
    return 1.0 - np.abs(2.0 * n - 1.0)                       # 'billow'


def _rot2(x, y, ang):
    c, s = math.cos(ang), math.sin(ang)
    return x * c - y * s, x * s + y * c


def fractal2d(fn, x, y, *, seed=0, octaves=4, gain=0.5, lacunarity=2.0,
              kind="fbm", period=None, rotate=False, **fn_kwargs):
    """fBm / turbulence / ridged / billow / hybrid. Pro improvements:
    · rotate — turn each octave by the golden angle: all octaves normally
      share one lattice, so their axis artifacts align; rotation
      decorrelates them (standard in production fBm). Ignored when tiling.
    · kind='hybrid' — Musgrave's multifractal: each octave's amplitude is
      multiplied by the running value, so valleys stay smooth while peaks
      accumulate detail."""
    total = np.zeros_like(x)
    amp, freq, norm = 1.0, 1.0, 0.0
    weight = 1.0
    for o in range(int(octaves)):
        xr, yr = x, y
        if rotate and period is None:
            xr, yr = _rot2(x, y, o * 2.399963229728653)   # golden angle
        kw = dict(fn_kwargs)
        if period is not None:
            kw["period"] = max(1, int(round(period * freq)))
        n = fn(xr * freq, yr * freq, seed=seed + o * 1013, **kw)
        if kind == "hybrid":
            sig = np.clip(n, 0.0, 1.0) * weight
            total += amp * sig
            weight = np.clip(sig * 2.0, 0.0, 1.0)
        else:
            total += amp * _shape_octave(n, kind)
        norm += amp
        amp *= gain; freq *= lacunarity
    return np.clip(total / norm, 0.0, 1.0)


def fractal3d(fn, x, y, z, *, seed=0, octaves=4, gain=0.5, lacunarity=2.0,
              kind="fbm", rotate=False, **fn_kwargs):
    total = np.zeros_like(x)
    amp, freq, norm = 1.0, 1.0, 0.0
    weight = 1.0
    for o in range(int(octaves)):
        xr, yr = x, y
        if rotate:
            xr, yr = _rot2(x, y, o * 2.399963229728653)
        n = fn(xr * freq, yr * freq, z * freq, seed=seed + o * 1013, **fn_kwargs)
        if kind == "hybrid":
            sig = np.clip(n, 0.0, 1.0) * weight
            total += amp * sig
            weight = np.clip(sig * 2.0, 0.0, 1.0)
        else:
            total += amp * _shape_octave(n, kind)
        norm += amp
        amp *= gain; freq *= lacunarity
    return np.clip(total / norm, 0.0, 1.0)


def octave_fields(fn, x, y, *, z=None, seed=0, octaves=4, gain=0.5,
                  lacunarity=2.0, kind="fbm", period=None, rotate=False,
                  **fn_kwargs):
    """Per-octave breakdown for the teaching strip:
    (per-octave fields, their weights, the normalised sum)."""
    views, weights = [], []
    total = np.zeros_like(x); amp, freq, norm = 1.0, 1.0, 0.0
    weight = 1.0
    for o in range(int(octaves)):
        xr, yr = x, y
        if rotate and period is None:
            xr, yr = _rot2(x, y, o * 2.399963229728653)
        kw = dict(fn_kwargs)
        if period is not None:
            kw["period"] = max(1, int(round(period * freq)))
        if z is None:
            n = fn(xr * freq, yr * freq, seed=seed + o * 1013, **kw)
        else:
            n = fn(xr * freq, yr * freq, z * freq, seed=seed + o * 1013, **kw)
        if kind == "hybrid":
            sig = np.clip(n, 0.0, 1.0) * weight
            views.append(sig)
            weight = np.clip(sig * 2.0, 0.0, 1.0)
        else:
            views.append(_shape_octave(n, kind))
        weights.append(amp)
        total += amp * views[-1]; norm += amp
        amp *= gain; freq *= lacunarity
    return views, weights, np.clip(total / norm, 0.0, 1.0)


def domain_warp_coords(fn, x, y, *, z=None, strength=1.0, seed=0, **fn_kwargs):
    """Domain warping: displace the sample position by another instance of
    the same noise,  p' = p + k·(noise(p) − ½)·2."""
    if z is None:
        qx = fn(x,       y,       seed=seed + 8101, **fn_kwargs)
        qy = fn(x + 5.2, y + 1.3, seed=seed + 8102, **fn_kwargs)
    else:
        qx = fn(x, y, z,       seed=seed + 8101, **fn_kwargs)
        qy = fn(x, y, z + 7.31, seed=seed + 8102, **fn_kwargs)
    return (x + strength * (qx - 0.5) * 2.0,
            y + strength * (qy - 0.5) * 2.0)


_ALGO_FN3 = {
    "value": value_noise3,
    "perlin": perlin_noise3,
    "simplex": simplex_noise3,
    "worley": worley_noise3,
}

# =====================================================================
#  Educational content
# =====================================================================

ALGO_INFO = {
    "value": (
        "<h3>Value noise</h3>"
        "<p>A random number is stored at every corner of a square lattice. Each "
        "pixel blends the four surrounding corner values, weighted by a "
        "<b>fade curve</b>.</p>"
        "<p><b>Why it matters:</b> it is the simplest &ldquo;structured&rdquo; noise &mdash; "
        "proof that interpolating randomness gives coherent blobs instead of static.</p>"
        "<p><b>Signature look:</b> blobby but with visible axis-aligned creases, "
        "because the randomness lives exactly on grid lines.</p>"
        "<p><b>Try:</b> set the fade to <i>Linear</i> and watch diamond artifacts "
        "appear; switch to <i>Quintic</i> and they vanish. Turn on the lattice "
        "overlay to see the grid it lives on.</p>"
        "<p style='color:#7d8aa3'>Implementation: <code>value_noise()</code></p>"
    ),
    "perlin": (
        "<h3>Perlin noise</h3>"
        "<p>Instead of random <i>values</i>, each lattice corner stores a random "
        "<b>gradient vector</b>. A pixel's value comes from dotting each corner's "
        "gradient with the offset to the pixel, then blending the four results "
        "with the fade curve.</p>"
        "<p><b>Why it matters:</b> invented by Ken Perlin for <i>Tron</i> (1982), "
        "technical Academy Award in 1997. The default terrain/texture noise for decades.</p>"
        "<p><b>Signature look:</b> smoother and more directional than value noise; "
        "features stretch along gradient directions.</p>"
        "<p><b>Try:</b> enable the <i>features</i> overlay to see the gradient vectors "
        "the whole field is built from. Compare Linear vs Quintic fade.</p>"
        "<p style='color:#7d8aa3'>Implementation: <code>perlin_noise()</code></p>"
    ),
    "simplex": (
        "<h3>Simplex noise</h3>"
        "<p>Perlin's 2001 rethink: replace the square grid with a triangular "
        "<b>simplex</b> mesh &mdash; the fewest corners a 2-D cell can have. Each pixel "
        "is influenced by only <b>3</b> corners (not 4), using radial kernels "
        "t&#8308;&middot;(g&middot;d) around each vertex.</p>"
        "<p><b>Why it matters:</b> fewer directional artifacts, no axis-aligned bias, "
        "and scales better to 3-D and above (4 corners vs 8+).</p>"
        "<p><b>Signature look:</b> Perlin-like but slightly sharper, with a subtle "
        "triangular grain.</p>"
        "<p><b>Try:</b> turn on the lattice overlay &mdash; the grid is skewed into "
        "triangles. Note each pixel sits inside exactly one triangle.</p>"
        "<p style='color:#7d8aa3'>Implementation: <code>simplex_noise()</code></p>"
    ),
    "worley": (
        "<h3>Worley / cellular noise</h3>"
        "<p>One random <b>feature point</b> is scattered in each cell. A pixel's value "
        "is the distance to the nearest point (F1) &mdash; or the gap between the two "
        "nearest (F2&thinsp;&minus;&thinsp;F1), which lights up the boundaries where "
        "distances are equal.</p>"
        "<p><b>Why it matters:</b> Steven Worley, 1996 &mdash; the basis of stone tiles, "
        "cracked mud, water caustics and Voronoi stylisation.</p>"
        "<p><b>Signature look:</b> crispy cells; F2&thinsp;&minus;&thinsp;F1 gives bright "
        "web-like edges.</p>"
        "<p><b>Try:</b> switch the distance metric to Manhattan or Chebyshev to reshape "
        "the cells; enable the feature-points overlay; try points/cell and jitter.</p>"
        "<p style='color:#7d8aa3'>Implementation: <code>worley_noise()</code></p>"
    ),
    "white": (
        "<h3>White noise</h3>"
        "<p>Every pixel is an independent uniform random number &mdash; no "
        "interpolation, no structure.</p>"
        "<p><b>Why it matters:</b> it is the raw material. Every other algorithm in "
        "this lab is a different way of <b>correlating</b> randomness across space. "
        "It contains all frequencies equally, hence &ldquo;white&rdquo; (like white light).</p>"
        "<p><b>Try:</b> compare its cross-section (pure chaos) with Perlin's "
        "(smooth rolling hills), and its flat power spectrum with Perlin's blob.</p>"
        "<p style='color:#7d8aa3'>Implementation: <code>white_noise()</code></p>"
    ),
}

FRACTAL_INFO = {
    "single": "",
    "fbm": (
        "<hr><h4>fBm — fractional Brownian motion</h4>"
        "<p>Octaves of the base noise are stacked: each octave is sampled "
        "<b>lacunarity&times;</b> finer and weighted <b>persistence&times;</b> weaker. "
        "Big hills + medium bumps + fine grain add up to natural-looking detail.</p>"
    ),
    "turbulence": (
        "<hr><h4>Turbulence</h4>"
        "<p>Like fBm, but each octave is folded with |2n&thinsp;&minus;&thinsp;1|. Sharp "
        "creases appear where the noise crosses 0.5 &mdash; the classic recipe for "
        "marble and fire.</p>"
    ),
    "ridged": (
        "<hr><h4>Ridged multifractal</h4>"
        "<p>Each octave uses (1&thinsp;&minus;&thinsp;|2n&thinsp;&minus;&thinsp;1|)&sup2;, so thin "
        "bright <b>ridges</b> run where the base noise crosses 0.5. Lower persistence "
        "gives smoother chains of mountains.</p>"
    ),
    "billow": (
        "<hr><h4>Billow</h4>"
        "<p>Each octave uses 1&thinsp;&minus;&thinsp;|2n&thinsp;&minus;&thinsp;1|: rounded, puffy "
        "blobs that kiss at cusps. Good for clouds and cotton.</p>"
    ),
}

Z_INFO = (
    "<hr><h4>3-D slice &amp; animation</h4>"
    "<p>The 3-D variants store their randomness on a <b>cubic lattice</b>. "
    "We sample a thin slab at height <b>z</b>; animating z makes the field "
    "<b>morph continuously</b> &mdash; this is exactly how &ldquo;4-D clouds&rdquo; "
    "work in shaders (x, y, time).</p>"
    "<p>Watch features drift and dissolve instead of flickering: spatial "
    "coherence extends into the third axis.</p>"
)
WARP_INFO = (
    "<hr><h4>Domain warping</h4>"
    "<p>Before evaluating the noise, the coordinates themselves are displaced "
    "by another instance of it: <b>p&thinsp;&prime;&thinsp;=&thinsp;p&thinsp;+&thinsp;"
    "k&middot;noise(p)</b>. Stretched, swirling layers appear &mdash; the classic "
    "marble/flame trick. Raise <b>k</b> and the fabric of the field folds.</p>"
)
COMPARE_INFO = (
    "<hr><h4>A / B split view</h4>"
    "<p>Left = current algorithm (A), right = comparison (B), with the "
    "<b>same seed, frequency and fractal settings</b>. Hover either half &mdash; "
    "the cross-section follows your cursor.</p>"
)
ANATOMY_INFO = (
    "<hr><h4>X-ray: anatomy of a cell</h4>"
    "<p>The four panels show each corner's <b>contribution field</b> — for Perlin a "
    "tilted plane <i>g&thinsp;&middot;&thinsp;d</i> (blue negative, red positive), for Value a flat "
    "corner value. The blend panel shows them combined with fade weights "
    "<i>u, v</i>. At every corner, d&thinsp;=&thinsp;0 so the contribution is zero — "
    "that's why Perlin's field is 0.5 on the lattice.</p>"
)
SEAMLESS_INFO = (
    "<hr><h4>Seamless tiling</h4>"
    "<p>With <b>period = frequency</b>, lattice indices wrap modulo the period: "
    "cell (p, j) <i>is</i> cell (0, j). The field repeats exactly — check the "
    "3&times;3 proof under the canvas. For fBm, each octave needs an integer "
    "period, so <b>lacunarity must be integer</b> (2.0): try 1.7 and watch seams "
    "appear.</p>"
)
TERRACE_INFO = (
    "<hr><h4>Terracing</h4>"
    "<p>Quantising the field into N flat steps (<code>floor(n&middot;N)/(N&minus;1)</code>) turns smooth "
    "hills into plateaus with cliffs — a classic stylisation (patterned ground, "
    "paddy fields, game terrain). Each terrace is one contour band, filled.</p>"
)
NORMALMAP_INFO = (
    "<hr><h4>Normal maps</h4>"
    "<p>RGB now encodes the <b>surface direction</b> at each pixel "
    "(R&thinsp;=&thinsp;x, G&thinsp;=&thinsp;y, B&thinsp;=&thinsp;up; flat = lavender "
    "(128,&thinsp;128,&thinsp;255)). Game engines light this per-pixel for near-zero "
    "cost — the noise becomes a bump map without extra geometry.</p>"
)
DOTPROBE_INFO = (
    "<hr><h4>Dot-product probe</h4>"
    "<p>Each Perlin corner stores a gradient <b>g</b>; its influence on a sample at "
    "offset <b>d</b> is <b>g&thinsp;&middot;&thinsp;d&thinsp;=&thinsp;|g||d|&thinsp;cos&thinsp;&theta;</b>. "
    "The probe draws g (hue = direction), d (cyan), the angle &theta;, and the "
    "<b>projection of d onto g</b> &mdash; the only component of d that g responds "
    "to. Moving <i>perpendicular</i> to g changes nothing: that insensitivity to "
    "sideways motion is exactly why gradient noise looks smoother than value "
    "noise.</p>"
)
GABOR_INFO = (
    "<hr><h4>Gabor noise — band-pass by design</h4>"
    "<p>Each cell hides a tiny <b>warped cosine</b>: a Gaussian window times a "
    "plane wave with random orientation &amp; phase. Perlin spreads energy over "
    "many frequencies; Gabor puts it all in a <b>ring at radius f&sub0;</b>. "
    "Check the spectrum panel — one crisp annulus.</p>"
    "<p>That control is the point: film grain, fur, scratches — anything that "
    "should live at <i>exactly one</i> scale.</p>"
)
SPECTRAL_INFO = (
    "<hr><h4>Spectral synthesis — sculpting 1/f<sup>&beta;</sup></h4>"
    "<p>Instead of building structure in space, this builds it in <b>frequency "
    "space</b>: FFT white noise &rarr; multiply amplitude by f<sup>&minus;&beta;</sup> "
    "&rarr; inverse FFT. &beta;=0 white, 1 pink (fBm-like), 2 brown. The "
    "spectrum panel <i>is</i> the construction &mdash; watch its slope track "
    "&beta; exactly. Bonus: the FFT grid is periodic, so this noise "
    "<b>tiles seamlessly</b> with zero extra code.</p>"
)
MULTIFRACTAL_INFO = (
    "<hr><h4>Hybrid multifractal (Musgrave)</h4>"
    "<p>Ordinary fBm adds octaves with fixed weights &mdash; every altitude is "
    "equally rough. Musgrave's hybrid makes each octave's amplitude "
    "<b>depend on the value so far</b>: where the field is already low, later "
    "detail is suppressed (smooth valleys); where it is high, detail "
    "accumulates (rough peaks). One line of state, dramatically more "
    "mountain-like.</p>"
)
ROTATION_INFO = (
    "<hr><h4>Octave rotation — breaking the grid</h4>"
    "<p>All octaves normally share one lattice, so their axis artifacts "
    "<b>align and amplify</b>. Rotating each octave by the golden angle "
    "(137.5&deg;) decorrelates them &mdash; a standard production fix. Watch "
    "the contour lines: with rotation, no direction is special anymore.</p>"
)
DERIV_FLOW_INFO = (
    "<hr><h4>Flow — the noise knows its own slope</h4>"
    "<p>Small arrows show the <b>analytic gradient</b> of the field (hue = "
    "direction, length = steepness). The polyline from your cursor traces "
    "<b>gradient ascent</b> (warm, to a peak) and <b>descent</b> (cool, to a "
    "valley) using exact derivatives &mdash; the same trick real terrain "
    "erosion, flow maps and bump-mapping use. Perlin and Value compute it "
    "exactly; other algorithms fall back to finite differences.</p>"
)
GRADSET_INFO = (
    "<hr><h4>Gradient sets — Perlin's hidden bias</h4>"
    "<p>The classic 8 gradients include the four <b>diagonals</b>, which "
    "length-&#8730;2 &mdash; corner-to-corner features get extra energy "
    "(visible as X-shaped streaks). 16 even directions reduce it; a "
    "<b>continuous random angle</b> removes it entirely. Compare all three "
    "with contours on.</p>"
)

TASK = ("<div style='background:#212a3d; border-left:3px solid #6fd3ff; "
        "padding:6px 8px; margin:8px 0 2px 0;'><b>Try this:</b> {}</div>")

LESSONS = [
    # ---------------- Part I · Foundations ----------------
    dict(title="Raw material — white noise",
         html=("<p>Every pixel is an <b>independent</b> random number. It contains all "
               "frequencies equally — like white light — and your eye can find no pattern "
               "at all.</p><p>Every other algorithm in this course is just a different way "
               "of <b>correlating</b> these random values across space.</p>"
               + TASK.format("Stare — it's pure static. Switch the colormap: same chaos, "
                             "different clothes. Then move to lesson 1 to add structure.")),
         preset=dict(algo="white", kind="single", cmap="Grayscale", freq=4,
                     grid=False, feat=False)),
    dict(title="First structure — value noise",
         html=("<p>Put one random value on each <b>lattice corner</b> (blue circles, sized "
               "by their value) and let every pixel blend its four neighbours. Randomness "
               "becomes <b>coherent</b>: hills roughly one cell wide.</p>"
               "<p>The coherence length <i>is</i> the frequency slider.</p>"
               + TASK.format("Hover — the cross-section rolls smoothly with wavelength ≈ 1 "
                             "cell. Drag frequency 1 → 32. Toggle the grid to see the lattice "
                             "the values live on.")),
         preset=dict(algo="value", kind="single", fade="quintic", freq=4, cmap="Viridis",
                     grid=True, feat=True)),
    dict(title="The fade curve decides everything",
         html=("<p>Between corners we interpolate — but <i>how</i>? The <b>fade curve</b> "
               "eases the blend. <b>Linear</b> is only C⁰: its slope jumps at every cell "
               "edge, leaving diamond-shaped artifacts. <b>Smoothstep</b> fixes the slope "
               "(C¹). <b>Quintic</b> also fixes the curvature (C²) — Perlin's improved fade.</p>"
               "<p>Contour lines (on, at 12 levels) make the kinks impossible to miss.</p>"
               + TASK.format("Cycle fade: Linear → Smoothstep → Quintic, watching the "
                             "contour kinks melt away. The fade-curve plot on the right "
                             "highlights the active curve.")),
         preset=dict(algo="value", kind="single", fade="linear", freq=6, cmap="Terrain",
                     contours=12, grid=True, feat=False)),
    dict(title="Perlin's idea — gradients, not values",
         html=("<p>Perlin's trick: store a random <b>gradient vector</b> at each corner "
               "(arrows — hue encodes direction). A pixel's value is the dot product of "
               "each corner's gradient with the offset to that pixel, blended by the fade.</p>"
               "<p>Because dot(g, 0) = 0, the field is exactly 0.5 <b>at every lattice "
               "point</b> — the 0.5 contour threads through all the arrow bases.</p>"
               + TASK.format("Contours are at 16 levels, so 0.5 is one of them: watch the "
                             "lines converge on the arrows. Hover across an arrow tip vs. "
                             "its base.")),
         preset=dict(algo="perlin", kind="single", fade="quintic", freq=4, cmap="Viridis",
                     contours=16, grid=True, feat=True)),
    dict(title="The tyranny of the square — simplex",
         html=("<p>The square lattice quietly favours horizontal &amp; vertical directions. "
               "Simplex noise (Perlin 2001) <b>skews the grid into triangles</b>: each "
               "sample touches 3 corners instead of 4, using radial kernels "
               "t⁴·(g·d) that fade to zero mid-edge.</p>"
               "<p>Result: fewer directional artifacts, better scaling to 3-D+.</p>"
               + TASK.format("A = Perlin, B = Simplex, same seed. Hunt for axis-aligned "
                             "streaks in A that are absent in B. Hover both halves — the "
                             "cross-section follows you.")),
         preset=dict(algo="simplex", cmp="perlin", kind="single", freq=6, cmap="Viridis",
                     grid=False, feat=False)),
    dict(title="Distance fields — Worley / cellular",
         html=("<p>Scatter one <b>feature point</b> per cell (blue dots). F1 = distance to "
               "the nearest point: contour lines become <b>rings around each dot</b>. "
               "F2&thinsp;−&thinsp;F1 is the gap between the two nearest — near zero exactly "
               "on the <b>Voronoi boundary</b>, so edges flare bright.</p>"
               + TASK.format("With F1 + contours, read the distance map around each dot. "
                             "Switch Cell value to F2−F1 → glowing cell walls. Try the "
                             "Manhattan metric — rings become diamonds.")),
         preset=dict(algo="worley", kind="single", cell="f1", freq=4, cmap="Viridis",
                     contours=10, feat=True, grid=False)),
    # ---------------- Part II · Fractals & motion ----------------
    dict(title="Fractal stacking — fBm",
         html=("<p>One octave = one scale of bumps. Nature has <b>all</b> scales. fBm sums "
               "octaves: each is <i>lacunarity×</i> finer and <i>persistence×</i> weaker "
               "(classic: 2× and ½×). The strip below the canvas shows the sum being built, "
               "octave by octave.</p>"
               + TASK.format("Read the strip: big hills + medium + fine = fBm. Drag "
                             "persistence 0.30 ↔ 0.80 and watch the roughness — and the "
                             "value histogram on the right — change.")),
         preset=dict(algo="perlin", kind="fbm", freq=3, oct=5, gain=0.5, lac=2.0,
                     cmap="Viridis", grid=False, feat=False)),
    dict(title="Shaping octaves — turbulence · ridged · billow",
         html=("<p>Same octaves, different <b>shaping</b> before summing. Turbulence folds "
               "with |2n−1| → sharp creases (marble, fire). Ridged uses "
               "(1−|2n−1|)² → bright mountain ridges where n = 0.5. Billow uses "
               "1−|2n−1| → puffy cusps (clouds).</p>"
               + TASK.format("Cycle the Fractal combo while watching the octave strip — each "
                             "thumbnail is shaped <i>before</i> summing. The histogram "
                             "reshapes dramatically (ridged is bimodal!).")),
         preset=dict(algo="perlin", kind="ridged", freq=3, oct=5, gain=0.5, lac=2.0,
                     cmap="Glacier", grid=False, feat=False)),
    dict(title="Domain warping — instant marble",
         html=("<p>Before evaluating, displace the coordinates with another instance of the "
               "noise itself: <b>p&thinsp;′&thinsp;=&thinsp;p&thinsp;+&thinsp;k·noise(p)</b>. "
               "Features shear into flowing veins — the classic marble/flame recipe. k "
               "controls how violently space folds.</p>"
               + TASK.format("Rise k slowly 0 → 3 and watch the field fold over itself. "
                             "Then switch the Fractal to Turbulence with the Magma "
                             "colormap — genuine marble.")),
         preset=dict(algo="perlin", kind="fbm", freq=3, oct=4, gain=0.5, lac=2.0,
                     warp=True, warp_k=1.5, cmap="Plasma", grid=False, feat=False)),
    dict(title="The third axis — time",
         html=("<p>3-D variants store randomness on a <b>cubic</b> lattice; we render the "
               "z-slice, and z plays <b>time</b>. Animating z makes features drift, bloom "
               "and dissolve coherently — never flickering. This is exactly how shader "
               "&ldquo;4-D clouds&rdquo; work (x, y, time).</p>"
               + TASK.format("Press ▶ Animate. Then switch to Worley F2−F1 — cellular blobs "
                             "are born and die. Pause and scrub z by hand.")),
         preset=dict(algo="perlin", kind="fbm", freq=3, oct=4, gain=0.5, lac=2.0,
                     three_d=True, z=0.0, cmap="Glacier", grid=False, feat=False)),
    dict(title="Putting it together — shaded terrain",
         html=("<p>Two viewing tricks, one pipeline. <b>Relief</b> computes surface normals "
               "from the field's gradient and applies Lambert lighting — your brain reads it "
               "as 3-D instantly. <b>Contours</b> add iso-lines every 1/N, like a topo map. "
               "Real terrain tools combine these with ridged + warped fBm.</p>"
               + TASK.format("Toggle Relief on/off. Then switch Fractal to <b>Ridged</b> and "
                             "re-enable the domain warp (k ≈ 1.5) — convincing mountain "
                             "ranges in three clicks.")),
         preset=dict(algo="perlin", kind="fbm", freq=3, oct=6, gain=0.5, lac=2.0,
                     shade="relief", contours=12, cmap="Terrain", grid=False, feat=False)),
    # ---------------- Part III · Inside the machine ----------------
    dict(title="X-ray — anatomy of a Perlin cell",
         html=("<p>Time to open the hood. The <b>X-ray panel</b> decomposes the cell "
               "under your cursor: four corner contributions <i>g·d</i> (tilted planes, "
               "blue = negative, red = positive) and the fade-weighted blend that "
               "produces the field.</p>"
               "<p>Frequency is 2, so cells are huge. Weights <i>u, v</i> are the fade "
               "curves applied to your exact position.</p>"
               + TASK.format("Hover a corner: its panel turns green at that point "
                             "(g·d = 0) and the blend shows 0.5. Hover mid-edge: "
                             "only two corners have weight. Slide toward a corner and watch "
                             "u → 0, v → 0.")),
         preset=dict(algo="perlin", kind="single", fade="quintic", freq=2, cmap="Viridis",
                     anatomy=True, grid=True, feat=True)),
    dict(title="The spectrum — why 'white', why 1/f",
         html=("<p>The <b>spectrum panel</b> (right) is the field's Fourier power. "
               "White noise fills it uniformly — every frequency equally present, "
               "like white light. Perlin concentrates energy near the centre "
               "(low frequencies) — that's what &lsquo;coherent&rsquo; means "
               "mathematically.</p>"
               "<p>fBm's spectrum falls off like <b>1/f</b> — the statistical "
               "signature of coastlines, clouds, music and stock charts.</p>"
               + TASK.format("Set A = White, B = Perlin: A's spectrum is uniform "
                             "speckle, B's is a central blob. Now set A's Fractal to fBm, "
                             "6 octaves — the blob grows a smooth halo: the 1/f tail. "
                             "Compare Ridged: energy creases along the diagonals!")),
         preset=dict(algo="perlin", cmp="white", kind="single", freq=4, cmap="Viridis")),
    dict(title="Seamless tiling — noise on a torus",
         html=("<p>Textures must repeat invisibly. Trick: make the lattice "
               "<b>periodic</b> — cell indices wrap modulo the period, so cell "
               "(p, j) <i>is</i> cell (0, j) and the plane quietly becomes a "
               "<b>torus</b>. The 3×3 strip under the canvas is the proof.</p>"
               + TASK.format("Drag the tiling: edges continue perfectly. Switch "
                             "Value and Worley — all tile. Enable fBm with lacunarity "
                             "2.0 — still seamless (integer periods per octave). Set "
                             "lacunarity 1.7 — seams! Non-integer frequency ratios break "
                             "the wrap.")),
         preset=dict(algo="perlin", kind="single", freq=4, cmap="Viridis",
                     periodic=True)),
    dict(title="Voronoi regions — colouring the cells",
         html=("<p>So far Worley gave <i>distances</i>. A different reading: colour each "
               "pixel by <b>which point is nearest</b> — instant Voronoi diagram, the "
               "workhorse of stylised stone, camouflage and territory maps.</p>"
               "<p>Regions mode returns a random value per nearest-cell; the patchwork "
               "is the diagram.</p>"
               + TASK.format("Cycle Cell value: Regions → F1 → F2−F1. The "
                             "patchwork borders in Regions sit exactly where F2−F1 is "
                             "dark. Turn on feature points — every blue dot owns one patch.")),
         preset=dict(algo="worley", kind="single", cell="id", freq=5, cmap="Viridis",
                     feat=True)),
    dict(title="Terracing — stylised stepped terrain",
         html=("<p><b>Terracing</b> quantises the field into flat plateaus before "
               "shading, so slopes become cliffs and the relief light rakes across "
               "them. It's literally &lsquo;contour bands, filled&rsquo; — compare with the "
               "contour-line overlay.</p>"
               + TASK.format("Steps 2 → 12: two extremes, poster-flat and nearly "
                             "smooth. Then match it: set contour lines = steps − 1 and "
                             "see each line trace one terrace edge. Switch shading to "
                             "Relief vs Flat — cliffs only exist when lit.")),
         preset=dict(algo="perlin", kind="fbm", freq=3, oct=5, gain=0.5, lac=2.0,
                     shade="relief", terrace=6, contours=0, cmap="Terrain")),
    dict(title="Normal maps — noise in the graphics pipeline",
         html=("<p>Rendering now outputs the field as a <b>normal map</b>: each "
               "pixel stores which way its surface faces, RGB = (x, y, up). Flat "
               "areas are lavender (128, 128, 255); slopes tint toward red/green. "
               "This is how games get detailed surfaces from flat geometry.</p>"
               + TASK.format("Compare Normal map vs Relief — same data, two encodings: "
                             "one for humans, one for the GPU's lighting equation. Note the "
                             "directional colour: west-facing slopes redden, north-facing "
                             "green, by convention.")),
         preset=dict(algo="perlin", kind="fbm", freq=4, oct=5, gain=0.5, lac=2.0,
                     shade="normals", cmap="Grayscale")),
    dict(title="Seeds — same recipe, new world",
         html=("<p>Every field you've made is fully determined by its <b>seed</b> and "
               "parameters — nothing is stored. Same seed ⇒ bit-identical field, "
               "forever. That's the contract of procedural generation: huge worlds "
               "from tiny descriptions.</p>"
               "<p>The seed only reshuffles the lattice hashes — the "
               "<i>character</i> (frequency content, structure) is set by your "
               "sliders, and the spectrum proves it.</p>"
               + TASK.format("Hit <b>Random</b> five times — new worlds, same character "
                             "(watch the spectrum barely change). Now type seed "
                             "<b>42</b>: you get exactly this field back any time. "
                             "Congratulations — you've completed the course. Free play awaits.")),
         preset=dict(algo="simplex", kind="fbm", freq=3, oct=5, gain=0.5, lac=2.0,
                     seed=42, cmap="Glacier")),
    dict(title="The dot product — g·d up close",
         html=("<p>Perlin's whole contribution formula is one operation: "
               "<b>g&thinsp;&middot;&thinsp;d&thinsp;=&thinsp;|g||d|&thinsp;cos&thinsp;&theta;</b>. "
               "The probe panel (right) locks onto the corner <i>nearest your cursor</i>: "
               "<b>g</b> (hue-coded) is the stored gradient; <b>d</b> (cyan) is the offset "
               "to your sample; the white arc is &theta;; the thick bar is the projection "
               "of d onto g &mdash; orange when the vectors agree, blue when they "
               "disagree.</p>"
               "<p>Hover <i>exactly on</i> a corner: d&thinsp;=&thinsp;0, so this corner "
               "contributes 0 whatever g is &mdash; which is why Perlin fields equal 0.5 "
               "at every lattice point.</p>"
               + TASK.format("Slide outward along g: the bar grows orange. Slide the "
                             "opposite way: it flips blue/negative. Move in a circle "
                             "around the corner at fixed radius: the bar is longest "
                             "parallel to g, zero perpendicular — dot products only "
                             "see the aligned component.")),
         preset=dict(algo="perlin", kind="single", fade="quintic", freq=3, cmap="Viridis",
                     grid=True, feat=True, dotprobe=True)),
    # ---------------- Part IV · Algorithm engineering ----------------
    dict(title="Gradient sets — Perlin's diagonal bias",
         html=("<p>Perlin's classic 8 gradients include the four <b>diagonals</b> "
               "&mdash; length &#8730;2, so corner-to-corner features get extra "
               "energy. You'll see it as faint X-shaped streaks. The fix: more "
               "even directions, or a <b>continuous random angle</b> per corner "
               "(perfectly isotropic).</p>"
               + TASK.format("Contours on. Cycle the Gradient set combo: "
                             "Classic 8 → 16 directions → Random angle, and "
                             "watch diagonal streaks dissolve. Then set B = A and flip "
                             "only the gradient set for a direct A/B.")),
         preset=dict(algo="perlin", kind="single", fade="quintic", freq=6,
                     cmap="Viridis", contours=14, gset="classic8", grid=True)),
    dict(title="Octave rotation — decorrelating the lattice",
         html=("<p>Every fBm octave reuses the same lattice axes, so their "
               "artifacts <b>line up and reinforce</b> — a subtle but real "
               "flaw in production use. Rotating each octave by the golden "
               "angle (137.5&deg;) makes no direction special.</p>"
               + TASK.format("Toggle 'Rotate octaves' and compare the contour "
                             "patterns: aligned → decorrelated. For the "
                             "loudest demo, use Billow at high frequency and watch "
                             "the axis-aligned cusps scatter. Then rotate in 3-D mode "
                             "too — the churn becomes non-directional.")),
         preset=dict(algo="perlin", kind="billow", freq=8, oct=4, gain=0.5,
                     lac=2.0, rotate=True, cmap="Viridis")),
    dict(title="Hybrid multifractal — rough peaks, smooth valleys",
         html=("<p>Plain fBm spends detail everywhere equally. Musgrave's "
               "<b>hybrid multifractal</b> gates each octave by the value so "
               "far: low regions suppress later octaves (silky valleys), high "
               "regions accumulate them (craggy summits). One extra line of "
               "state &mdash; a whole new landform.</p>"
               + TASK.format("Switch Fractal between fBm and Hybrid at the same "
                             "seed: same randomness, different geography. Add Relief "
                             "shading and ridged-flavoured comparison. Drag persistence "
                             "&mdash; hybrid is much more sensitive to it.")),
         preset=dict(algo="perlin", kind="hybrid", freq=3, oct=6, gain=0.55,
                     lac=2.0, shade="relief", cmap="Terrain")),
    dict(title="Gabor noise — one frequency, on purpose",
         html=("<p>Perlin noise is <i>broadband</i>: all frequencies at once. "
               "<b>Gabor noise</b> sums tiny warped cosines with random "
               "orientations, so its energy lives in a <b>ring at exactly "
               "f&thinsp;=&thinsp;f&sub0;</b> — see the spectrum panel. This is "
               "spectral engineering, not an accident.</p>"
               + TASK.format("Drag f&sub0; up and down: the texture's stroke "
                             "scale AND the spectrum ring radius move together. "
                             "Compare with Perlin at the same frequency — one ring vs "
                             "a filled blob. That's what 'band-pass' means.")),
         preset=dict(algo="gabor", kind="single", freq=4, f0=2.0, cmap="Viridis")),
    dict(title="Spectral synthesis — noise from the FFT side",
         html=("<p>The deepest cut: skip spatial construction entirely. FFT "
               "white noise &rarr; scale each frequency by "
               "f<sup>&minus;&beta;</sup> &rarr; inverse FFT. &beta; is the "
               "single most honest parameter in the course: <b>0</b> white, "
               "<b>1</b> pink (what fBm approximates), <b>2</b> brown. And "
               "because the FFT grid is periodic, it <b>tiles perfectly</b> "
               "for free.</p>"
               + TASK.format("Drag &beta; 0 → 3 and watch the spectrum "
                             "panel's slope follow exactly — the render is literally "
                             "the inverse of that spectrum. Enable Seamless tiles: no "
                             "period arithmetic, no lacunarity constraint — it just "
                             "tiles. Turn on Flow: how do the streamlines differ from "
                             "Perlin's?")),
         preset=dict(algo="spectral", kind="single", beta=20, cmap="Glacier",
                     flow=True, periodic=True)),
    # ---------------- Part V · Skills & recipes ----------------
    dict(title="Reading the field — every noise is a heightfield",
         html=("<p>Core skill: every 2-D field is a <b>landscape</b>, and the "
               "cross-section is your microscope. Value noise profiles are flat-ish "
               "shelves joined by eased steps; Perlin profiles are smooth S-curves "
               "that pass through 0.5 at every lattice point; Worley F1 is a "
               "conescape — V-shaped valleys plunging to zero <b>exactly at each "
               "feature point</b>.</p>"
               + TASK.format("Switch to Worley F1 with feature points on. Slide "
                             "along a row in the cross-section and count the V-valleys — "
                             "one per blue dot. Now Perlin: find the flat crossings where "
                             "the profile pierces 0.5 — each one sits above an arrow base. "
                             "This reading skill unlocks every lesson after this.")),
         preset=dict(algo="perlin", kind="fbm", freq=4, oct=4, gain=0.5,
                     lac=2.0, cmap="Viridis")),
    dict(title="Persistence — the spectrum's tail",
         html=("<p>Persistence is not just 'roughness' — it is a <b>spectral "
               "slope</b> in disguise. Octave powers fall as "
               "gain<sup>2o</sup>, so an fBm's power spectrum obeys "
               "P&thinsp;&prop;&thinsp;f<sup>&minus;&beta;</sup> with "
               "<b>&beta;&thinsp;=&thinsp;&minus;2&thinsp;ln(gain)&thinsp;/&thinsp;ln(lacunarity)</b>. "
               "Classic (0.5,&thinsp;2.0) ⇒ &beta;&thinsp;=&thinsp;2. Gain 0.7 "
               "⇒ &beta;&thinsp;&asymp;&thinsp;1. Gain 0.35 ⇒ "
               "&beta;&thinsp;&asymp;&thinsp;3.</p>"
               "<p>B is live <b>Spectral</b> noise built directly from f<sup>&minus;&beta;</sup> "
               "— your calibration target.</p>"
               + TASK.format("Set B's &beta; slider to 2.0 and compare: the fBm blob "
                             "and the spectral blob should have near-identical falloff. "
                             "Drag gain to 0.70 — predict &beta; — then check by matching "
                             "B. You've just proven persistence and &beta; are the same knob.")),
         preset=dict(algo="perlin", cmp="spectral", kind="fbm", freq=3, oct=6,
                     gain=0.5, lac=2.0, cmap="Viridis")),
    dict(title="Lacunarity — the gaps between octaves",
         html=("<p>Lacunarity sets how far apart octave frequencies sit. "
               "<b>Integer 2.0</b> puts them on a harmonic ladder — the spectrum "
               "shows distinct clumps at 1×, 2×, 4×. Drag it "
               "off-integer and the clumps smear together: octave bands <b>overlap "
               "and beat</b>, producing subtly different (some say richer, some "
               "say muddier) texture.</p>"
               + TASK.format("Watch the spectrum while dragging lacunarity "
                             "1.5 → 2.0 → 3.0: clumps merge, lock into a ladder, "
                             "spread apart. Now enable Seamless tiles at lacunarity 1.7 — "
                             "seams return (lesson 14's warning, seen from the spectral "
                             "side).")),
         preset=dict(algo="perlin", kind="fbm", freq=3, oct=5, gain=0.5,
                     lac=2.0, cmap="Viridis")),
    dict(title="Distance metrics — circles, diamonds, squares",
         html=("<p>'Distance' has a family: the p-norms. <b>Euclidean</b> draws "
               "round equipotentials; <b>Manhattan</b> (|dx|+|dy|) draws diamonds; "
               "<b>Chebyshev</b> (max) draws squares. Same points, same "
               "algorithm — three different geometries, just by changing how "
               "distance is measured.</p>"
               + TASK.format("With F1 + contours, cycle the metric: rings "
                             "→ diamonds → squares around every dot. Switch to "
                             "F2&thinsp;&minus;&thinsp;F1 and repeat — Manhattan cell walls "
                             "become strikingly straight and architectural. Which metric "
                             "would you pick for cracked mud? For city blocks?")),
         preset=dict(algo="worley", kind="single", cell="f1", freq=4,
                     contours=12, feat=True, cmap="Viridis")),
    dict(title="Jitter — from crystal grid to Poisson field",
         html=("<p>With <b>jitter&nbsp;0</b>, feature points sit at perfect grid "
               "positions: textbook Voronoi — and the field is almost periodic. "
               "The spectrum betrays it: sharp <b>Bragg peaks</b> where the point "
               "lattice glares back. Raise jitter and the peaks melt into a smooth "
               "cloud: the point set becomes Poisson-like, the most 'natural' "
               "distribution there is.</p>"
               + TASK.format("Jitter 0 with the spectrum panel visible: find the "
                             "periodic spikes. Sweep jitter to 1.0 and watch them dissolve. "
                             "Then set points/cell = 3 — denser cells, same character "
                             "(F1 is rescaled by 1/&radic;points so brightness stays put). "
                             "Jitter 0 + points 1 is the only fully regular noise in the lab.")),
         preset=dict(algo="worley", kind="single", cell="f1", freq=4, jitter=0.0,
                     points=1, feat=True, cmap="Viridis")),
    dict(title="Inside simplex — triangles and t⁴",
         html=("<p>Turn on the lattice overlay: simplex's square grid has been "
               "<b>skewed</b> into triangles. Each sample finds its triangle "
               "(3 corners, never 4) and blends three kernels "
               "t&#8308;&middot;(g&middot;d) with t&thinsp;=&thinsp;0.5&thinsp;&minus;&thinsp;r&sup2;. "
               "The key trick: t hits <b>zero</b> partway along each edge, so each "
               "kernel dies before it could bleed across the triangle.</p>"
               + TASK.format("Hover a triangle's centre — you're equidistant from "
                             "three hue-coded arrows. Slide across a shared edge and watch "
                             "the far corner's influence die to nothing exactly on the line: "
                             "that hard cutoff is what keeps simplex free of the square "
                             "grid's directional fingerprint.")),
         preset=dict(algo="simplex", kind="single", freq=3, grid=True,
                     feat=True, cmap="Viridis")),
    dict(title="Follow the gradient — flow lines",
         html=("<p>Perlin and Value noise can report their <b>exact derivative</b> "
               "— no finite-difference guessing. The flow overlay draws &nabla;field "
               "arrows (hue&thinsp;=&thinsp;direction), and from your cursor two "
               "streamlines: warm climbs to a <b>peak</b>, cool dives to a "
               "<b>valley</b>. These are the same derivatives that drive erosion "
               "sims, fluid advection and cheap bump lighting.</p>"
               + TASK.format("Drop the cursor on a hillside and follow the warm "
                             "line to a summit. Now hunt for a <b>saddle</b>: a point where "
                             "the warm line forks toward two peaks — you've found the mountain "
                             "pass. Note the flow arrows are densest on steep flanks and "
                             "vanish at extrema, exactly like real terrain analysis.")),
         preset=dict(algo="perlin", kind="single", freq=4, flow=True,
                     grid=True, cmap="Viridis")),
    dict(title="Recipe: living clouds",
         html=("<p>Time to cook. Billow's puffy cusps are cloud cauliflowers; "
               "sampling 3-D noise at z&thinsp;=&thinsp;time makes them "
               "<b>evolve</b> instead of scroll; the Glacier colormap plus relief-"
               "free lighting sells the sky. Five controls, one weather system.</p>"
               + TASK.format("Press ▶ Animate. Raise persistence toward 0.7 "
                             "— the sky turns stormy. Drop to 3 octaves for stylised flat "
                             "clouds. Swap the algorithm to Worley F2&thinsp;&minus;&thinsp;F1 "
                             "and re-animate: cellular 'boiling' — a completely different "
                             "weather model from the same pipeline.")),
         preset=dict(algo="perlin", kind="billow", freq=3, oct=5, gain=0.55,
                     lac=2.0, three_d=True, z=0.0, speed=8, cmap="Glacier")),
    dict(title="Recipe: marble, properly",
         html=("<p>The classic Veins-of-Carrara recipe is two ingredients "
               "stacked: <b>turbulence</b> (sharp creases at every n&thinsp;=&thinsp;0.5 "
               "crossing) + <b>domain warp</b> (shear those creases into flowing "
               "bands). Lesson 9 introduced warp on gentle fBm; now the creases "
               "give the warp something to fold.</p>"
               + TASK.format("Sweep k 0.5 → 2.5: veins go from rippled "
                             "corrugation to full turbulent swirl. Then cycle the Fractal "
                             "combo — the same warp turns fbm into silk and ridged into "
                             "strata. Finish with the Magma colormap and a touch of Relief "
                             "shading: stone you could texture a lobby floor with.")),
         preset=dict(algo="perlin", kind="turbulence", freq=3, oct=5, gain=0.5,
                     lac=2.0, warp=True, warp_k=1.8, cmap="Plasma")),
    dict(title="Capstone — the mountain pipeline",
         html=("<p>Everything you've learned, in one render: <b>warp</b> folds "
               "space into believable geology, <b>ridged</b> octaves raise sharp "
               "ranges along the creases, <b>octave rotation</b> kills the last "
               "grid ghosts, and <b>relief + contours</b> present it like a "
               "survey map. This is genuinely close to how film-game terrain "
               "tools build backdrop mountains.</p>"
               + TASK.format("Hit Random a few times — every seed is a new "
                             "range. Push warp k to 2.2 for canyon country. Swap ridged "
                             "for Hybrid multifractal: rougher summits, smoother basins. "
                             "You now understand every slider on this panel — the lab is "
                             "yours. Build something.")),
         preset=dict(algo="perlin", kind="ridged", freq=3, oct=6, gain=0.5,
                     lac=2.0, warp=True, warp_k=1.2, shade="relief",
                     contours=8, rotate=True, cmap="Terrain")),
]

TIP_HTML = (
    "<hr><p style='color:#8fa3c8; font-size:11px;'>"
    "<b>Hover</b> the field — the cross-section plots the noise along your cursor's row.<br>"
    "<b>Arrows</b> = Perlin/Simplex gradients, <b>hue = direction</b> (a colour wheel).<br>"
    "<b>Blue dots</b> = stored lattice values (Value) or Worley feature points.<br>"
    "<b>X-ray</b> decomposes the hovered cell; the <b>Dot probe</b> magnifies the nearest "
    "corner (g, d, θ, g·d).<br>"
    "<b>Flow</b> shows ∇field arrows and a streamline to the local peak/valley; the "
    "<b>spectrum</b> shows frequency content.<br>"
    "<b>Relief/Normals</b> render it as terrain; <b>Contours</b> draw iso-lines.<br>"
    "All maths lives in the <code>NOISE LIBRARY</code> section of this file — read along."
    "</p>"
)

GLOBAL_QSS = """
QMainWindow, QSplitter::handle { background: #171a21; }
QSplitter::handle { background: #262b36; width: 3px; }
QGroupBox { font-weight: 600; border: 1px solid #262b36; border-radius: 8px;
            margin-top: 12px; padding: 6px; background: transparent; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px;
                   color: #93a4c3; }
QSlider::groove:horizontal { height: 4px; border-radius: 2px; background: #2a3040; }
QSlider::sub-page:horizontal { background: #3f6fb5; border-radius: 2px; }
QSlider::handle:horizontal { width: 13px; height: 13px; margin: -5px 0;
                             border-radius: 6px; background: #7fb1ff; }
QSlider::handle:horizontal:hover { background: #a8c9ff; }
QComboBox QAbstractItemView { background: #20242e; color: #d7dbe4;
                              selection-background-color: #3f7fd4; }
QPushButton { padding: 4px 10px; }
QTextBrowser { background: transparent; border: none; color: #cfd6e4;
               font-size: 12.5px; }
QScrollArea { background: transparent; }
QWidget#controlsPanel { background: #14171e; border-right: 1px solid #262b36; }
"""


# =====================================================================
#  Canvas — field image, A/B split, lattice overlays, flow, crosshair
# =====================================================================

class NoiseCanvas(QWidget):
    """Paints the field(s), lattice overlays, flow arrows/streamline and a
    hover crosshair. In compare mode two squares share one frequency:
    A (left) and B (right)."""

    hoverMoved = Signal(float, float, str)          # nx, ny, 'A' | 'B'

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(360, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self._field: np.ndarray | None = None
        self._img: QImage | None = None
        self._img2: QImage | None = None
        self._algo = "perlin"
        self._algo2: str | None = None
        self._freq = 4.0
        self._seed = 0
        self._gset = "classic8"
        self._show_grid = False
        self._show_features = True
        self._hover = (2.0, 2.0)
        self._hover_side = "A"
        self._flow = None
        self._stream = ([], [])

    # ------------------------------------------------------------ public API

    def set_field(self, field, freq, lut, algo, seed, show_grid, show_features,
                  shade="flat", contours=0, terrace=0, gradient_set="classic8"):
        self._field = field
        pixel = freq / field.shape[0]
        self._img = render_field(field, lut, shade, contours, pixel, terrace)
        self._freq = float(freq)
        self._algo = algo
        self._seed = int(seed)
        self._gset = gradient_set
        self._show_grid = show_grid
        self._show_features = show_features
        nx, ny = self._hover
        self._hover = (min(nx, self._freq), min(ny, self._freq))
        self.update()

    def set_compare(self, field, lut, algo, shade="flat", contours=0, terrace=0):
        if field is None:
            self._img2 = None
            self._algo2 = None
        else:
            pixel = self._freq / field.shape[0]
            self._img2 = render_field(field, lut, shade, contours, pixel, terrace)
            self._algo2 = algo
        self.update()

    def set_gradient_overlay(self, gx, gy, step):
        """(gx, gy) sampled on a grid of spacing `step` noise units, or None."""
        self._flow = None if gx is None else (gx, gy, float(step))
        self.update()

    def set_streamline(self, up, down):
        self._stream = (up, down)
        self.update()

    def hover(self):
        return self._hover

    def hover_side(self):
        return self._hover_side

    def image(self):
        return self._img

    # ------------------------------------------------------------- geometry

    def _dest_rects(self):
        m, gap = 10.0, 14.0
        w, h = self.width() - 2 * m, self.height() - 2 * m
        if self._img2 is None:
            s = max(10.0, min(w, h))
            return [(QRectF(m + (w - s) / 2, m + (h - s) / 2, s, s), "A")]
        s = max(10.0, min(h, (w - gap) / 2.0))
        x = m + (w - (2 * s + gap)) / 2.0
        y = m + (h - s) / 2.0
        return [(QRectF(x, y, s, s), "A"), (QRectF(x + s + gap, y, s, s), "B")]

    def _rect_of(self, side):
        for r, s in self._dest_rects():
            if s == side:
                return r
        return self._dest_rects()[0][0]

    def _to_px(self, r: QRectF, nx: float, ny: float) -> QPointF:
        f = self._freq
        return QPointF(r.left() + nx / f * r.width(), r.top() + ny / f * r.height())

    # ------------------------------------------------------------- painting

    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#101218"))
        if self._img is None:
            p.setPen(QColor("#8a93a6"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "no field yet")
            return
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        rects = self._dest_rects()
        for r, side in rects:
            img = self._img if side == "A" else self._img2
            p.drawImage(r, img, QRectF(0, 0, img.width(), img.height()))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QColor("#3a4152"), 1))
            p.drawRect(r)
        if self._img2 is None:                       # overlays: single view only
            r0 = rects[0][0]
            if self._show_grid:
                self._paint_grid(p, r0)
            if self._show_features:
                self._paint_features(p, r0)
            if self._flow is not None:
                self._paint_flow(p, r0)
            if self._stream[0] or self._stream[1]:
                self._paint_stream(p, r0)
        self._paint_tags(p, rects)
        self._paint_crosshair(p, rects)

    def _paint_tags(self, p: QPainter, rects):
        p.setFont(QFont(self.font().family(), 10, QFont.Weight.Bold))
        for r, side in rects:
            name = _ALGO_NAMES.get(self._algo if side == "A" else self._algo2, "?")
            text = f"{side} · {name}"
            tw = p.fontMetrics().horizontalAdvance(text) + 12
            pill = QRectF(r.left() + 6, r.top() + 6, tw, p.fontMetrics().height() + 4)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(0, 0, 0, 150))
            p.drawRoundedRect(pill, 4, 4)
            p.setPen(QColor(235, 240, 250))
            p.drawText(pill, Qt.AlignmentFlag.AlignCenter, text)

    def _paint_crosshair(self, p: QPainter, rects):
        nx, ny = self._hover
        side = self._hover_side
        r = self._rect_of(side)
        ra = rects[0][0]
        p.setPen(QPen(QColor(255, 255, 255, 80), 1, Qt.PenStyle.DashLine))
        y = self._to_px(r, 0.0, ny).y()
        x_end = rects[-1][0].right()
        p.drawLine(QPointF(ra.left(), y), QPointF(x_end, y))
        x = self._to_px(r, nx, 0.0).x()
        p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
        pt = self._to_px(r, nx, ny)
        p.setBrush(QColor(255, 255, 255, 200))
        p.setPen(QPen(QColor("#0d0f15"), 1))
        p.drawEllipse(pt, 3.5, 3.5)
        if len(rects) > 1:
            other = self._rect_of("B" if side == "A" else "A")
            pt2 = self._to_px(other, nx, ny)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QColor(255, 255, 255, 130), 1))
            p.drawEllipse(pt2, 4.5, 4.5)

    def _paint_grid(self, p: QPainter, r: QRectF):
        p.save(); p.setClipRect(r)
        f = self._freq
        if self._algo == "simplex":
            g2 = SIMPLEX_G2
            hi, lo = int(math.ceil(f)) + 2, -2

            def v(i, j):
                return self._to_px(r, i - (i + j) * g2, j - (i + j) * g2)

            p.setPen(QPen(QColor(255, 255, 255, 42), 1))
            for k in range(lo, hi):
                p.drawLine(v(k, lo), v(k, hi))            # constant i
                p.drawLine(v(lo, k), v(hi, k))            # constant j
                p.drawLine(v(k + lo, lo), v(k + hi, hi))  # main diagonals
        else:
            p.setPen(QPen(QColor(255, 255, 255, 40), 1))
            k = 0
            while k <= f + 1e-9:
                x = r.left() + (k / f) * r.width()
                y = r.top() + (k / f) * r.height()
                p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
                p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
                k += 1
        p.restore()

    def _paint_features(self, p: QPainter, r: QRectF):
        p.save(); p.setClipRect(r)
        seed, n, cell = self._seed, int(math.ceil(self._freq)), r.width() / self._freq
        if self._algo in ("perlin", "simplex"):
            colour = QColor(255, 203, 92, 220)
            p.setPen(QPen(colour, 1.6)); p.setBrush(QBrush(colour))
            L, head = 0.40 * cell, 0.16 * cell
            for i in range(n + 1):
                for j in range(n + 1):
                    if self._algo == "perlin":
                        cx, cy = float(i), float(j)
                    else:
                        g2 = SIMPLEX_G2
                        cx, cy = i - (i + j) * g2, j - (i + j) * g2
                    gv = lattice_gradient(i, j, seed, self._gset)
                    # hue encodes direction — a colour wheel on the lattice
                    ang = (math.degrees(math.atan2(gv[1], gv[0]))) % 360.0
                    col = QColor.fromHsv(int(ang), 190, 255, 235)
                    p.setPen(QPen(col, 1.8))
                    p.setBrush(QBrush(col))
                    c = self._to_px(r, cx, cy)
                    p.drawEllipse(c, 2.2, 2.2)
                    tip = QPointF(float(c.x() + gv[0] * L), float(c.y() + gv[1] * L))
                    p.drawLine(c, tip)
                    for s in (0.5, -0.5):
                        dx = -gv[0] * math.cos(s) + gv[1] * math.sin(s)
                        dy = -gv[0] * math.sin(s) - gv[1] * math.cos(s)
                        p.drawLine(tip, QPointF(float(tip.x() + dx * head),
                                                float(tip.y() + dy * head)))
        elif self._algo == "value":
            colour = QColor(120, 220, 255, 200)
            p.setPen(QPen(QColor(120, 220, 255, 220), 1)); p.setBrush(QBrush(colour))
            for i in range(n + 1):
                for j in range(n + 1):
                    v = lattice_value(i, j, seed)
                    rad = cell * (0.05 + 0.16 * v)
                    p.drawEllipse(self._to_px(r, float(i), float(j)), rad, rad)
        elif self._algo == "worley":
            p.setPen(QPen(QColor(120, 220, 255, 230), 1))
            p.setBrush(QBrush(QColor(120, 220, 255, 200)))
            rad = max(2.0, cell * 0.07)
            for i in range(n + 1):
                for j in range(n + 1):
                    fx, fy = lattice_feature(i, j, seed)
                    p.drawEllipse(self._to_px(r, fx, fy), rad, rad)
        p.restore()

    def _paint_flow(self, p: QPainter, r: QRectF):
        p.save(); p.setClipRect(r)
        gx, gy, step = self._flow
        cell = r.width() / self._freq
        H, W = gx.shape
        for j in range(H):
            for i in range(W):
                a = float(gx[j, i]); b = float(gy[j, i])
                m = math.hypot(a, b)
                if m < 1e-4:
                    continue
                c = self._to_px(r, i * step, j * step)
                L = min(0.42 * cell, max(3.0, 0.30 * cell * m))
                ux, uy = a / m, b / m
                hue = int(math.degrees(math.atan2(b, a)) % 360)
                col = QColor.fromHsv(hue, 160, 255, 150)
                p.setPen(QPen(col, 1.4, Qt.PenStyle.SolidLine,
                              Qt.PenCapStyle.RoundCap))
                tip = QPointF(float(c.x() + ux * L), float(c.y() + uy * L))
                p.drawLine(c, tip)
                ang = math.atan2(uy, ux)
                for s in (2.6, -2.6):
                    p.drawLine(tip,
                               QPointF(float(tip.x() - math.cos(ang - s) * 3.5),
                                       float(tip.y() - math.sin(ang - s) * 3.5)))
        p.restore()

    def _paint_stream(self, p: QPainter, r: QRectF):
        p.save(); p.setClipRect(r)
        for pts, colour in ((self._stream[0], QColor("#ffd166")),
                            (self._stream[1], QColor("#74c0ff"))):
            if len(pts) < 2:
                continue
            path = QPainterPath(self._to_px(r, *pts[0]))
            for x, y in pts[1:]:
                path.lineTo(self._to_px(r, x, y))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(colour, 2.4, Qt.PenStyle.SolidLine,
                          Qt.PenCapStyle.RoundCap))
            p.drawPath(path)
            e = self._to_px(r, *pts[-1])
            p.setBrush(QBrush(colour))
            p.setPen(QPen(QColor("#0d0f15"), 1))
            p.drawEllipse(e, 4.0, 4.0)
        p.restore()

    # --------------------------------------------------------------- events

    def mouseMoveEvent(self, ev):
        self._update_hover(ev.position())

    def _update_hover(self, pos: QPointF):
        for r, side in self._dest_rects():
            if r.left() <= pos.x() <= r.right():
                fy = (pos.y() - r.top()) / r.height()
                nx = (pos.x() - r.left()) / r.width() * self._freq
                ny = float(np.clip(fy, 0.0, 1.0)) * self._freq
                self._hover = (float(np.clip(nx, 0.0, self._freq)), ny)
                self._hover_side = side
                self.hoverMoved.emit(*self._hover, side)
                self.update()
                return


# =====================================================================
#  Teaching panels
# =====================================================================

class CrossSection(QWidget):
    """1-D slice under the cursor; in compare mode draws A and B together."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(170)
        self._values: np.ndarray | None = None
        self._values2: np.ndarray | None = None
        self._primary = "A"
        self._freq = 4.0
        self._hover_x: float | None = None
        self._ticks = False

    def set_data(self, values, freq, hover_x=None, ticks=False,
                 values2=None, primary="A"):
        self._values, self._values2 = values, values2
        self._primary = primary
        self._freq = float(freq)
        self._hover_x = hover_x
        self._ticks = ticks
        self.update()

    def _curve(self, p, values, colour, width, left, top, w, h):
        n = len(values)
        xs = left + (np.arange(n) / (n - 1)) * w
        ys = top + (1.0 - values) * h
        path = QPainterPath(QPointF(float(xs[0]), float(ys[0])))
        for x, y in zip(xs[1:], ys[1:]):
            path.lineTo(QPointF(float(x), float(y)))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(colour, width))
        p.drawPath(path)
        return xs, ys

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.fillRect(self.rect(), QColor("#0d0f15"))
        if self._values is None or len(self._values) < 2:
            return
        ml, mr, mt, mb = 34.0, 12.0, 8.0, 20.0
        w, h = self.width() - ml - mr, self.height() - mt - mb
        if w <= 10 or h <= 10:
            return
        left, top = ml, mt

        p.setFont(QFont(self.font().family(), 8))
        for gy, lab in ((1.0, "1.0"), (0.5, "0.5"), (0.0, "0.0")):
            y = top + (1.0 - gy) * h
            p.setPen(QPen(QColor(255, 255, 255, 26), 1, Qt.PenStyle.DashLine))
            p.drawLine(QPointF(left, y), QPointF(left + w, y))
            p.setPen(QColor("#5b6478"))
            p.drawText(QRectF(4, y - 6, ml - 8, 12),
                       Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, lab)

        if self._values2 is not None:      # the non-primary side behind, thinner
            c2 = QColor("#6fd3ff") if self._primary == "B" else QColor("#ffcb5c")
            self._curve(p, self._values2, c2, 1.4, left, top, w, h)
        c1 = QColor("#6fd3ff") if self._primary == "A" else QColor("#ffcb5c")
        xs, ys = self._curve(p, self._values, c1, 2.0, left, top, w, h)

        if self._ticks:
            p.setPen(QPen(QColor(255, 255, 255, 70), 1))
            k = 0
            while k <= self._freq + 1e-9:
                x = left + (k / self._freq) * w
                p.drawLine(QPointF(x, top + h), QPointF(x, top + h + 4))
                k += 1

        if self._hover_x is not None:
            frac = float(np.clip(self._hover_x / self._freq, 0.0, 1.0))
            x = left + frac * w
            idx = int(np.clip(round(frac * (len(self._values) - 1)), 0,
                              len(self._values) - 1))
            p.setPen(QPen(QColor(255, 255, 255, 120), 1, Qt.PenStyle.DashLine))
            p.drawLine(QPointF(x, top), QPointF(x, top + h))
            p.setBrush(QColor("#ffffff"))
            p.setPen(QPen(QColor("#0d0f15"), 1))
            p.drawEllipse(QPointF(x, float(ys[idx])), 3.2, 3.2)

        if self._values2 is not None:      # legend
            p.setFont(QFont(self.font().family(), 8, QFont.Weight.Bold))
            for i, (lab, col) in enumerate((("A", "#6fd3ff"), ("B", "#ffcb5c"))):
                p.setPen(QColor(col))
                p.drawText(QPointF(left + w - 44 + i * 24, top + 10), lab)

        p.setPen(QColor("#5b6478"))
        p.drawText(QRectF(0, self.height() - 16, self.width(), 14),
                   Qt.AlignmentFlag.AlignHCenter,
                   "value along the hovered row   ·   x \u2192")


class FadeCurves(QWidget):
    """Plots the three interpolation fades; highlights the active one."""

    CURVES = [
        ("linear",  "linear  C⁰",     lambda t: t),
        ("smooth",  "smoothstep  C¹", lambda t: t * t * (3 - 2 * t)),
        ("quintic", "quintic  C²",    lambda t: t**3 * (t * (6*t - 15) + 10)),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(140)
        self._active: str | None = "quintic"

    def set_active(self, key: str | None):
        if key != self._active:
            self._active = key
            self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.fillRect(self.rect(), QColor("#0d0f15"))
        ml, mr, mt, mb = 30.0, 10.0, 8.0, 22.0
        w, h = self.width() - ml - mr, self.height() - mt - mb
        if w < 40 or h < 30:
            return
        p.setFont(QFont(self.font().family(), 8))
        p.setPen(QPen(QColor(255, 255, 255, 26), 1, Qt.PenStyle.DashLine))
        for gy in (0.25, 0.5, 0.75):
            y = mt + (1 - gy) * h
            p.drawLine(QPointF(ml, y), QPointF(ml + w, y))
        ts = np.linspace(0.0, 1.0, 120)
        for key, label, fn in self.CURVES:
            active = key == self._active
            ys = fn(ts)
            pts = [QPointF(ml + t * w, mt + (1 - y) * h) for t, y in zip(ts, ys)]
            path = QPainterPath(pts[0])
            for pt in pts[1:]:
                path.lineTo(pt)
            p.setPen(QPen(QColor("#6fd3ff") if active else QColor("#39415a"),
                          2.4 if active else 1.4))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)
            p.setPen(QColor("#9fd9ff") if active else QColor("#4a5470"))
            p.drawText(QPointF(ml + 6, mt + 11 +
                       (1 - fn(0.035)) * h + (26 if key == "linear" else
                                              12 if key == "smooth" else 0)), label)
        p.setPen(QPen(QColor(255, 255, 255, 40), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(QRectF(ml, mt, w, h))
        p.setPen(QColor("#5b6478"))
        p.drawText(QRectF(0, self.height() - 16, self.width(), 14),
                   Qt.AlignmentFlag.AlignHCenter,
                   "weight given to the corner ahead  ·  t →")


class HistogramStrip(QWidget):
    """Distribution of the current field's values."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(104)
        self._hist: np.ndarray | None = None
        self._mean = 0.5

    def set_field(self, field):
        hist, _ = np.histogram(np.clip(field, 0, 1), bins=56, range=(0.0, 1.0))
        self._hist = hist / max(float(hist.max()), 1.0)
        self._mean = float(field.mean())
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.fillRect(self.rect(), QColor("#0d0f15"))
        if self._hist is None:
            return
        ml, mr, mt, mb = 12.0, 12.0, 8.0, 18.0
        w, h = self.width() - ml - mr, self.height() - mt - mb
        n = len(self._hist)
        bw = w / n
        path = QPainterPath(QPointF(ml, mt + h))
        for i, v in enumerate(self._hist):
            path.lineTo(QPointF(ml + i * bw, mt + (1 - v) * h))
            path.lineTo(QPointF(ml + (i + 1) * bw, mt + (1 - v) * h))
        path.lineTo(QPointF(ml + w, mt + h))
        path.closeSubpath()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(111, 211, 255, 60))
        p.drawPath(path)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor("#6fd3ff"), 1.4))
        p.drawPath(path)
        x = ml + self._mean * w
        p.setPen(QPen(QColor(255, 203, 92, 200), 1, Qt.PenStyle.DashLine))
        p.drawLine(QPointF(x, mt), QPointF(x, mt + h))
        p.setFont(QFont(self.font().family(), 8))
        p.setPen(QColor("#ffcb5c"))
        p.drawText(QPointF(min(x + 4, ml + w - 46), mt + 10), f"\u03bc {self._mean:.2f}")
        p.setPen(QColor("#5b6478"))
        p.drawText(QRectF(0, self.height() - 15, self.width(), 14),
                   Qt.AlignmentFlag.AlignHCenter,
                   "value distribution   0 → 1")


class SpectrumView(QWidget):
    """Log power spectrum (2-D FFT, DC centred). White noise fills the plane
    uniformly — hence 'white'. Structured noise concentrates energy at low
    frequencies; fBm falls off like 1/f — the signature of nature."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(158)
        self._img: QImage | None = None

    def set_field(self, field):
        f = np.fft.fftshift(np.fft.fft2(field - float(field.mean())))
        p = np.log1p(np.abs(f))
        hi = float(np.percentile(p, 99.9))
        p = np.clip((p - p.min()) / max(hi - p.min(), 1e-9), 0.0, 1.0)
        self._img = field_to_image(p, _COLORMAPS["Magma"])
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        p.fillRect(self.rect(), QColor("#0d0f15"))
        if self._img is None:
            return
        side = min(self.width() - 12, self.height() - 24)
        r = QRectF((self.width() - side) / 2.0, 2.0, side, side)
        p.drawImage(r, self._img, QRectF(0, 0, self._img.width(), self._img.height()))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor("#3a4152"), 1))
        p.drawRect(r)
        p.setFont(QFont(self.font().family(), 8))
        p.setPen(QColor("#5b6478"))
        p.drawText(QRectF(0, r.bottom() + 2, self.width(), 14),
                   Qt.AlignmentFlag.AlignHCenter,
                   "log power spectrum · centre = low frequency")


class PerlinAnatomy(QWidget):
    """X-ray of the hovered lattice cell: the four corner contributions and
    the fade-weighted blend that produces the final value there."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(172)
        self._tiles: list[QImage] = []
        self._labels: list[str] = []
        self._weights = ""
        self._div = _COLORMAPS["Diverging"]
        self._blend = _COLORMAPS["Viridis"]

    def set_colormap(self, lut):
        self._blend = lut

    def set_probe(self, algo, nx, ny, seed, fade="quintic"):
        if algo not in ("perlin", "value"):
            return
        N = 40
        t = np.linspace(0.0, 1.0, N)
        T, S = np.meshgrid(t, t)                    # position inside the cell
        ix, iy = int(math.floor(nx)), int(math.floor(ny))
        fx, fy = nx - ix, ny - iy
        u = _fade(fx, fade)
        v = _fade(fy, fade)
        tiles, labels, contribs = [], [], []
        for ci, cj in ((0, 0), (1, 0), (0, 1), (1, 1)):
            if algo == "perlin":
                g = lattice_gradient(ix + ci, iy + cj, seed)
                c = g[0] * (T - ci) + g[1] * (S - cj)   # dot(gradient, offset)
                labels.append(f"g{ci}{cj}\u00b7d")
            else:
                c = np.full_like(T, lattice_value(ix + ci, iy + cj, seed))
                labels.append(f"n{ci}{cj}={c.flat[0]:.2f}")
            contribs.append(c)
            tiles.append(field_to_image(np.clip(0.5 + 0.5 * c, 0, 1), self._div))
        nx0 = contribs[0] + u * (contribs[1] - contribs[0])   # blend bottom edge
        nx1 = contribs[2] + u * (contribs[3] - contribs[2])   # blend top edge
        final = nx0 + v * (nx1 - nx0)                          # blend edges
        tiles.append(field_to_image(np.clip(0.5 + 0.5 * final, 0, 1), self._blend))
        labels.append("blend")
        self._tiles, self._labels = tiles, labels
        self._weights = (f"u = fade({fx:.2f}) = {u:.2f}      "
                         f"v = fade({fy:.2f}) = {v:.2f}      "
                         f"cell ({ix},{iy})")
        self.update()

    def clear(self):
        self._tiles = []; self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        p.fillRect(self.rect(), QColor("#0d0f15"))
        if not self._tiles:
            p.setPen(QColor("#5b6478"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "hover a cell…")
            return
        n = len(self._tiles)
        gap, lab_h = 6, 14
        s = min((self.width() - gap * (n + 1)) / n, self.height() - lab_h - 26)
        if s < 20:
            return
        x = (self.width() - (n * s + (n - 1) * gap)) / 2.0
        y = 4.0
        p.setFont(QFont(self.font().family(), 8))
        for img, label in zip(self._tiles, self._labels):
            r = QRectF(x, y, s, s)
            p.drawImage(r, img, QRectF(0, 0, img.width(), img.height()))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QColor("#3a4152"), 1))
            p.drawRect(r)
            p.setPen(QColor("#8fa3c8"))
            p.drawText(QRectF(x - 4, y + s + 1, s + 8, lab_h),
                       Qt.AlignmentFlag.AlignCenter, label)
            x += s + gap
        p.setPen(QColor("#aeb8cc"))
        p.drawText(QRectF(0, self.height() - 20, self.width(), 18),
                   Qt.AlignmentFlag.AlignCenter, self._weights)


class TilePreview(QWidget):
    """3×3 wrap-around: if every tile edge continues into its neighbour,
    the noise is seamless. Amber box marks the 'real' image."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(128)
        self._img: QImage | None = None

    def set_image(self, img):
        self._img = img
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        p.fillRect(self.rect(), QColor("#0d0f15"))
        if self._img is None:
            return
        side = min(self.width() - 10, self.height() - 22) / 3.0
        ox = (self.width() - side * 3) / 2.0
        oy = (self.height() - 14 - side * 3) / 2.0
        src = QRectF(0, 0, self._img.width(), self._img.height())
        for j in range(3):
            for i in range(3):
                p.drawImage(QRectF(ox + i * side, oy + j * side, side, side),
                            self._img, src)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor(255, 255, 255, 26), 1))
        for k in (1, 2):
            p.drawLine(QPointF(ox + k * side, oy),
                       QPointF(ox + k * side, oy + 3 * side))
            p.drawLine(QPointF(ox, oy + k * side),
                       QPointF(ox + 3 * side, oy + k * side))
        p.setPen(QPen(QColor(255, 203, 92, 220), 1.4, Qt.PenStyle.DashLine))
        p.drawRect(QRectF(ox + side, oy + side, side, side))
        p.setFont(QFont(self.font().family(), 8))
        p.setPen(QColor("#8a6a1e"))
        p.drawText(QRectF(0, self.height() - 14, self.width(), 12),
                   Qt.AlignmentFlag.AlignHCenter,
                   "seamless? follow the pattern across tile borders")


class DotProductDiagram(QWidget):
    """Zooms on the lattice corner nearest the cursor and draws the Perlin
    dot product live:  g (stored gradient), d (offset to the sample), the
    angle θ between them, and the projection of d onto g — which IS the dot:
        g·d = |g| · |d| · cos θ = |g| × (projection length)
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(250)
        self._has = False
        self._g = (1.0, 0.0)
        self._d = (0.5, 0.5)
        self._cell = None            # hovered cell origin, relative to corner
        self._note = ""

    def set_probe(self, algo, nx, ny, seed, gradient_set="classic8"):
        if algo == "perlin":
            bx, by = math.floor(nx), math.floor(ny)
            ci, cj = min(((bx + a, by + b) for a in (0, 1) for b in (0, 1)),
                         key=lambda c: (nx - c[0])**2 + (ny - c[1])**2)
            dx, dy = nx - ci, ny - cj
            g = lattice_gradient(ci, cj, seed, gradient_set)
            self._cell = (bx - ci, by - cj)
            self._note = f"lattice corner ({ci}, {cj})"
        elif algo == "simplex":                     # faithful to the algorithm:
            F2, G2 = 0.36602540378443865, SIMPLEX_G2           # first vertex,
            s = (nx + ny) * F2                                 # unskewed space
            ci, cj = int(math.floor(nx + s)), int(math.floor(ny + s))
            t = (ci + cj) * G2
            dx, dy = nx - (ci - t), ny - (cj - t)
            g = lattice_gradient(ci, cj, seed, gradient_set)
            self._cell = None
            self._note = f"simplex vertex ({ci}, {cj}) · unskewed space"
        else:
            return
        self._g = (float(g[0]), float(g[1]))
        self._d = (float(dx), float(dy))
        self._has = True
        self.update()

    def clear(self):
        if self._has:
            self._has = False
            self.update()

    @staticmethod
    def _arrow(p, a, b, colour, width):
        p.setBrush(QBrush(colour))
        p.setPen(QPen(colour, width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(a, b)
        ang = math.atan2(b.y() - a.y(), b.x() - a.x())
        L = max(8.0, width * 3.2)
        for s in (2.55, -2.55):
            p.drawLine(b, QPointF(b.x() - math.cos(ang - s) * L,
                                  b.y() - math.sin(ang - s) * L))

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.fillRect(self.rect(), QColor("#0d0f15"))
        if not self._has:
            p.setPen(QColor("#5b6478"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "hover the field to probe its nearest corner…")
            return

        gx, gy = self._g
        dx, dy = self._d
        gL = math.hypot(gx, gy)
        dL = math.hypot(dx, dy)
        dot = gx * dx + gy * dy

        band = 64
        cx = self.width() / 2.0
        cy = (self.height() - band) / 2.0
        R = max(min(self.width(), self.height() - band) / 2.0 - 26.0, 40.0)

        def P(x, y):
            return QPointF(cx + x * R, cy - y * R)

        p.fillRect(self.rect(),
                   QColor(255, 120, 90, 16) if dot >= 0 else QColor(110, 150, 255, 18))

        p.setPen(QPen(QColor("#39415a"), 1))
        p.drawLine(QPointF(cx - 1.6 * R, cy), QPointF(cx + 1.6 * R, cy))
        p.drawLine(QPointF(cx, cy - 1.6 * R), QPointF(cx, cy + 1.6 * R))
        p.setPen(QPen(QColor("#4a5470"), 1))
        for u in (-1.0, 1.0):
            p.drawLine(P(u, -0.06), P(u, 0.06))
            p.drawLine(P(-0.06, u), P(0.06, u))
        p.setPen(QPen(QColor("#3a4152"), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                if i or j:
                    p.drawEllipse(P(float(i), float(j)), 2.0, 2.0)
        if self._cell is not None:
            ox, oy = self._cell
            p.setPen(QPen(QColor(255, 255, 255, 34), 1, Qt.PenStyle.DashLine))
            p.drawRect(QRectF(P(float(ox), float(oy)), P(float(ox + 1), float(oy + 1))))

        a_g = math.degrees(math.atan2(gy, gx))
        a_d = math.degrees(math.atan2(dy, dx)) if dL > 1e-9 else a_g
        delta = ((a_d - a_g + 180.0) % 360.0) - 180.0
        if dL > 1e-9:
            r = 0.36 * R
            p.setPen(QPen(QColor(255, 255, 255, 110), 1.4))
            p.drawArc(QRectF(cx - r, cy - r, 2 * r, 2 * r),
                      round(a_g * 16), round(delta * 16))
            mid = math.radians(a_g + delta / 2.0)
            p.setPen(QColor("#cfd6e4"))
            p.setFont(QFont(self.font().family(), 9, QFont.Weight.Bold))
            p.drawText(P(math.cos(mid) * 0.52 * R, math.sin(mid) * 0.52 * R)
                       + QPointF(-16.0, 4.0), f"\u03b8={abs(delta):.0f}\u00b0")

        pos = dot >= 0.0
        sign_col = QColor("#ff8a5c") if pos else QColor("#5c8aff")
        if dL > 1e-9:
            t = dot / (gL * gL)
            foot = P(t * gx, t * gy)
            tip = P(dx, dy)
            p.setPen(QPen(QColor("#8fa3c8"), 1.2, Qt.PenStyle.DashLine))
            p.drawLine(tip, foot)                   # perpendicular drop
            p.setPen(QPen(sign_col, 4.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawLine(P(0.0, 0.0), foot)           # the projection itself

        k = min(1.0, 1.25 / gL)
        gtip = P(gx * k, gy * k)
        ghue = int(math.degrees(math.atan2(gy, gx)) % 360)
        self._arrow(p, P(0.0, 0.0), gtip, QColor.fromHsv(ghue, 190, 255, 240), 2.8)
        if dL > 1e-9:
            dtip = P(dx, dy)
            self._arrow(p, P(0.0, 0.0), dtip, QColor("#6fd3ff"), 2.8)
            p.setBrush(QColor("#ffffff"))
            p.setPen(QPen(QColor("#0d0f15"), 1))
            p.drawEllipse(dtip, 3.0, 3.0)
        p.setBrush(QColor("#ffffff"))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(P(0.0, 0.0), 3.4, 3.4)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QColor("#e8ecf5"))
        p.setFont(QFont(self.font().family(), 10, QFont.Weight.Bold))
        p.drawText(gtip + QPointF(7.0, -6.0), "g")
        if dL > 1e-9:
            p.drawText(P(dx, dy) + (QPointF(8.0, -6.0) if dx >= 0
                                    else QPointF(-20.0, -6.0)), "d")

        top = self.height() - band
        p.setFont(QFont(self.font().family(), 9))
        p.setPen(QColor("#aeb8cc"))
        p.drawText(QRectF(8, top, self.width() - 16, 16), Qt.AlignmentFlag.AlignLeft,
                   f"g = ({gx:+.2f}, {gy:+.2f})   |g| = {gL:.2f}        "
                   f"d = ({dx:+.2f}, {dy:+.2f})   |d| = {dL:.2f}")
        p.setFont(QFont(self.font().family(), 10, QFont.Weight.Bold))
        if dL > 1e-9:
            cos_t = dot / (gL * dL)
            p.setPen(sign_col)
            p.drawText(QRectF(8, top + 17, self.width() - 16, 17),
                       Qt.AlignmentFlag.AlignLeft,
                       f"g\u00b7d = {dot:+.3f}  =  |g||d|cos\u03b8 = "
                       f"{gL:.2f}\u00b7{dL:.2f}\u00b7{cos_t:+.2f}     "
                       f"\u03b8 = {abs(delta):.1f}\u00b0")
        else:
            p.setPen(QColor("#cfd6e4"))
            p.drawText(QRectF(8, top + 17, self.width() - 16, 17),
                       Qt.AlignmentFlag.AlignLeft,
                       "d = 0  \u2192  this corner's contribution is exactly 0")
        p.setFont(QFont(self.font().family(), 8))
        p.setPen(QColor("#5b6478"))
        p.drawText(QRectF(8, top + 36, self.width() - 16, 14),
                   Qt.AlignmentFlag.AlignLeft,
                   "thick bar = projection of d onto g (\u2248 |d|cos\u03b8) \u2014 "
                   "the only part of d that g can \u201csee\u201d")
        p.setPen(QColor("#66718a"))
        p.drawText(QRectF(8, top + 49, self.width() - 16, 13),
                   Qt.AlignmentFlag.AlignLeft, self._note)


class OctaveStrip(QWidget):
    """Renders each fractal octave as a thumbnail, then the weighted sum."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(118)
        self._thumbs: list[QImage] = []
        self._labels: list[str] = []

    def set_octaves(self, views, weights, combined, lut):
        self._thumbs = [field_to_image(v, lut) for v in views]
        self._thumbs.append(field_to_image(combined, lut))
        self._labels = [f"oct {o}  ×{w:.2f}" for o, w in enumerate(weights)]
        self._labels.append("sum  Σ/Σw")
        self.update()

    def clear(self):
        self._thumbs = []; self._labels = []
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        p.fillRect(self.rect(), QColor("#0d0f15"))
        n = len(self._thumbs)
        if n < 2:
            return
        gap, lab_h = 22, 20
        s = min(self.height() - lab_h - 8,
                (self.width() - gap * (n + 1)) / n)
        if s < 24:
            return
        x = (self.width() - (n * s + (n + 1) * gap)) / 2.0
        y = 4.0
        p.setFont(QFont(self.font().family(), 9))
        for i, (img, lab) in enumerate(zip(self._thumbs, self._labels)):
            r = QRectF(x, y, s, s)
            p.drawImage(r, img, QRectF(0, 0, img.width(), img.height()))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QColor("#3a4152"), 1))
            p.drawRect(r)
            p.setPen(QColor("#8fa3c8"))
            p.drawText(QRectF(x - 12, y + s + 2, s + 24, lab_h),
                       Qt.AlignmentFlag.AlignCenter, lab)
            if i < n - 1:
                p.setPen(QColor("#cfd6e4"))
                f = QFont(self.font().family(), 13)
                f.setBold(True); p.setFont(f)
                p.drawText(QRectF(x + s, y, gap, s),
                           Qt.AlignmentFlag.AlignCenter,
                           "=" if i == n - 2 else "+")
                p.setFont(QFont(self.font().family(), 9))
            x += s + gap


# =====================================================================
#  Main window
# =====================================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Noise Lab · learn 2-D noise algorithms")
        self.resize(1400, 900)
        self.setMinimumSize(1020, 680)
        self._lesson_idx = 0
        self._fieldA: np.ndarray | None = None
        self._fieldB: np.ndarray | None = None
        self._freq = 4.0
        self._playing = False
        self._flowgrad = None
        self._lut = _COLORMAPS["Viridis"]

        self._debounce = QTimer(self, singleShot=True, interval=35)
        self._debounce.timeout.connect(self._recompute)
        self._anim = QTimer(self, interval=30)
        self._anim.timeout.connect(self._tick)

        root = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(root)
        root.addWidget(self._build_controls())
        root.addWidget(self._build_center())
        root.addWidget(self._build_right())
        root.setStretchFactor(1, 1)
        root.setSizes([320, 660, 420])

        self._update_control_states()
        self._recompute()

    # ------------------------------------------------------------- left panel

    def _build_controls(self) -> QWidget:
        panel = QWidget(objectName="controlsPanel")
        v = QVBoxLayout(panel)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(8)

        title = QLabel("NOISE LAB")
        title.setStyleSheet("font-size:18px; font-weight:700; letter-spacing:3px; color:#e8ecf5;")
        sub = QLabel("a guided tour of noise algorithms")
        sub.setStyleSheet("color:#8fa3c8;")
        v.addWidget(title); v.addWidget(sub); v.addSpacing(6)

        gl, glv = self._group("Lessons")
        self.lesson_combo = QComboBox()
        self._rows: list[int | None] = []        # combo row -> lesson number

        def add_row(text, lesson):
            self.lesson_combo.addItem(text)
            self._rows.append(lesson)
            if lesson is None:                   # grey out chapter headers
                self.lesson_combo.setItemData(
                    self.lesson_combo.count() - 1, 0, Qt.ItemDataRole.UserRole - 1)

        add_row("\u2014  free play  \u2014", 0)
        starts = dict(CHAPTERS)
        for i, les in enumerate(LESSONS, start=1):
            if i in starts:
                add_row(f"\u2500\u2500\u2500  {starts[i]}  \u2500\u2500\u2500", None)
            add_row(f"{i}  \u00b7  {les['title']}", i)
        glv.addWidget(self.lesson_combo)
        nav = QHBoxLayout()
        b_prev = QPushButton("\u25c0  Prev")
        b_next = QPushButton("Next  \u25b6")
        b_prev.clicked.connect(lambda: self._step_lesson(-1))
        b_next.clicked.connect(lambda: self._step_lesson(+1))
        nav.addWidget(b_prev); nav.addWidget(b_next)
        glv.addLayout(nav)
        v.addWidget(gl)

        g, gv = self._group("Algorithm")
        self.algo_combo = self._combo(gv, ALGORITHMS)
        self.compare_combo = self._combo(gv, [("\u2014 no comparison \u2014", None)] + ALGORITHMS,
                                         "A / B split view (same seed & freq)")
        self.fade_combo = self._combo(gv, FADES, "Fade / interpolation")
        self.gset_combo = self._combo(gv, GRADSETS, "Gradient set (Perlin)")
        self.metric_combo = self._combo(gv, METRICS, "Distance metric")
        self.cell_combo = self._combo(gv, CELLMODES, "Cell value")
        wrow = QHBoxLayout()
        wrow.addWidget(QLabel("Worley points/cell"))
        self.points_spin = QSpinBox()
        self.points_spin.setRange(1, 4)
        wrow.addWidget(self.points_spin)
        self.jitter_slider = self._slider(gv, "Jitter", 0, 100, 100,
                                          lambda v: f"{v/100:.2f}")
        gv.addLayout(wrow)
        v.addWidget(g)

        g2, gv2 = self._group("Fractal stacking")
        self.fractal_combo = self._combo(gv2, FRACTALS)
        self.freq_slider = self._slider(gv2, "Frequency", 1, 32, 4, lambda v: f"{v} cells")
        self.oct_slider = self._slider(gv2, "Octaves", 1, 8, 4)
        self.gain_slider = self._slider(gv2, "Persistence", 5, 100, 50, lambda v: f"{v/100:.2f}")
        self.lac_slider = self._slider(gv2, "Lacunarity", 15, 40, 20, lambda v: f"{v/10:.2f}")
        self.rotate_cb = QCheckBox("Rotate octaves (golden angle)")
        gv2.addWidget(self.rotate_cb)
        v.addWidget(g2)

        ga, gva = self._group("Generator parameters")
        self.f0_slider = self._slider(gva, "Gabor f\u2080", 2, 16, 4, lambda v: f"{v/2:.1f}")
        self.beta_slider = self._slider(gva, "Spectral \u03b2", 0, 30, 10, lambda v: f"{v/10:.1f}")
        v.addWidget(ga)

        g3d, gv3d = self._group("3-D slice · time")
        self.three_d_cb = QCheckBox("Use 3-D noise (z = time)")
        gv3d.addWidget(self.three_d_cb)
        self.z_slider = self._slider(gv3d, "z", 0, 100, 0, lambda v: f"{v/10:.1f}")
        self.speed_slider = self._slider(gv3d, "Speed", 1, 40, 10)
        self.play_btn = QPushButton("\u25b6  Animate")
        self.play_btn.setCheckable(True)
        self.play_btn.toggled.connect(self._toggle_play)
        gv3d.addWidget(self.play_btn)
        v.addWidget(g3d)

        gw, gvw = self._group("Domain warp")
        self.warp_cb = QCheckBox("Warp space:  p \u2192 p + k\u00b7noise(p)")
        gvw.addWidget(self.warp_cb)
        self.warp_slider = self._slider(gvw, "Strength k", 0, 30, 10, lambda v: f"{v/10:.1f}")
        v.addWidget(gw)

        gv_, gvv = self._group("Visual style & probes")
        self.shade_combo = self._combo(gvv, SHADE_MODES, "Shading")
        crow = QHBoxLayout()
        crow.addWidget(QLabel("Contours"))
        self.contour_spin = QSpinBox(); self.contour_spin.setRange(0, 40)
        crow.addWidget(self.contour_spin, 1)
        crow.addWidget(QLabel("Terrace"))
        self.terrace_spin = QSpinBox(); self.terrace_spin.setRange(0, 24)
        self.terrace_spin.setSpecialValueText("off")
        crow.addWidget(self.terrace_spin, 1)
        gvv.addLayout(crow)
        self.anatomy_cb = QCheckBox("X-ray: anatomy of hovered cell")
        self.dot_cb = QCheckBox("Dot-product probe (g \u00b7 d)")
        self.flow_cb = QCheckBox("Flow: \u2207field arrows + streamline")
        for cb in (self.anatomy_cb, self.dot_cb, self.flow_cb):
            gvv.addWidget(cb)
        v.addWidget(gv_)

        g3, gv3 = self._group("Output")
        seed_row = QHBoxLayout()
        seed_row.addWidget(QLabel("Seed"))
        self.seed_spin = QSpinBox(); self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(1337)
        rnd = QPushButton("Random")
        rnd.clicked.connect(lambda: self.seed_spin.setValue(random.randrange(1_000_000)))
        seed_row.addWidget(self.seed_spin, 1); seed_row.addWidget(rnd)
        gv3.addLayout(seed_row)
        self.cmap_combo = self._combo(gv3, [(n, n) for n in _COLORMAPS])
        self.res_combo = self._combo(gv3, [(f"{r} × {r}", r) for r in RESOLUTIONS],
                                     "Resolution")
        self.periodic_cb = QCheckBox("Seamless tiles (period = frequency)")
        gv3.addWidget(self.periodic_cb)
        v.addWidget(g3)

        g4, gv4 = self._group("Overlays")
        self.grid_cb = QCheckBox("Show lattice grid")
        self.feat_cb = QCheckBox("Show lattice features (points / gradients)")
        self.feat_cb.setChecked(True)
        gv4.addWidget(self.grid_cb); gv4.addWidget(self.feat_cb)
        v.addWidget(g4)
        v.addStretch(1)

        self.algo_combo.setCurrentIndex(1)
        self.cmap_combo.setCurrentIndex(1)
        self.res_combo.setCurrentIndex(2)

        self.lesson_combo.currentIndexChanged.connect(self._on_lesson_selected)
        for w in (self.algo_combo, self.compare_combo, self.fade_combo,
                  self.gset_combo, self.metric_combo, self.cell_combo,
                  self.fractal_combo, self.cmap_combo, self.res_combo,
                  self.shade_combo):
            w.currentIndexChanged.connect(self._schedule)
        for w in (self.freq_slider, self.oct_slider, self.gain_slider,
                  self.lac_slider, self.jitter_slider, self.f0_slider,
                  self.beta_slider):
            w.valueChanged.connect(self._schedule)
        for w in (self.three_d_cb, self.warp_cb, self.grid_cb, self.feat_cb,
                  self.periodic_cb, self.anatomy_cb, self.dot_cb, self.flow_cb,
                  self.rotate_cb):
            w.toggled.connect(self._schedule)
        for w in (self.warp_slider, self.z_slider, self.terrace_spin,
                  self.seed_spin, self.points_spin):
            w.valueChanged.connect(self._schedule)
        self.z_slider.sliderPressed.connect(self._stop_play)

        scroll = QScrollArea(widgetResizable=True, frameShape=QFrame.Shape.NoFrame)
        scroll.setWidget(panel)
        scroll.setMinimumWidth(300)
        return scroll

    # ---------------------------------------------------- center / right panels

    def _build_center(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(4)
        self.canvas = NoiseCanvas()
        self.canvas.hoverMoved.connect(self._on_hover)
        v.addWidget(self.canvas, 1)
        self.strip = OctaveStrip()
        v.addWidget(self.strip)
        self.tile_preview = TilePreview()
        self.tile_preview.setVisible(False)
        v.addWidget(self.tile_preview)
        self.readout = QLabel("hover the field…  ·  pick a lesson on the left")
        self.readout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.readout.setStyleSheet(
            "color:#aeb8cc; font-family:'Consolas','Menlo',monospace; font-size:12px;")
        v.addWidget(self.readout)
        return w

    def _build_right(self) -> QWidget:
        scroll = QScrollArea(widgetResizable=True, frameShape=QFrame.Shape.NoFrame)
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(6, 12, 12, 12)
        v.setSpacing(8)

        self.lesson_group, lv = self._group("Lesson")
        self.lesson_title = QLabel(); self.lesson_title.setWordWrap(True)
        self.lesson_title.setStyleSheet("font-size:14px; font-weight:600; color:#e8ecf5;")
        lv.addWidget(self.lesson_title)
        self.lesson_text = QTextBrowser(openExternalLinks=False)
        self.lesson_text.setMinimumHeight(200)
        lv.addWidget(self.lesson_text)
        v.addWidget(self.lesson_group)
        self.lesson_group.hide()

        g1, gv1 = self._group("Cross-section · row under cursor")
        self.cross = CrossSection()
        gv1.addWidget(self.cross)
        v.addWidget(g1)

        self.fade_group, fgv = self._group("Interpolation fade curves")
        self.fadec = FadeCurves()
        fgv.addWidget(self.fadec)
        v.addWidget(self.fade_group)

        gspec, gsv = self._group("Power spectrum")
        self.spectrum = SpectrumView()
        gsv.addWidget(self.spectrum)
        v.addWidget(gspec)

        self.dot_group, dgv = self._group("Dot product \u00b7 g\u00b7d at the nearest corner")
        self.dotdiag = DotProductDiagram()
        dgv.addWidget(self.dotdiag)
        v.addWidget(self.dot_group)
        self.dot_group.hide()

        self.anatomy_group, agv = self._group("X-ray · hovered cell")
        self.anatomy = PerlinAnatomy()
        agv.addWidget(self.anatomy)
        v.addWidget(self.anatomy_group)
        self.anatomy_group.hide()

        g3, gv3 = self._group("Value distribution")
        self.hist = HistogramStrip()
        gv3.addWidget(self.hist)
        v.addWidget(g3)

        g2, gv2 = self._group("About this algorithm")
        self.info = QTextBrowser(openExternalLinks=False)
        gv2.addWidget(self.info)
        v.addWidget(g2)
        v.addStretch(1)
        scroll.setWidget(w)
        return scroll

    # ------------------------------------------------------------ UI helpers

    def _group(self, title):
        g = QGroupBox(title)
        gv = QVBoxLayout(g)
        gv.setContentsMargins(8, 4, 8, 8)
        gv.setSpacing(6)
        return g, gv

    def _combo(self, layout, items, caption=None):
        if caption:
            cap = QLabel(caption)
            cap.setStyleSheet("color:#8fa3c8; font-size:11px;")
            layout.addWidget(cap)
        combo = QComboBox()
        for label, _ in items:
            combo.addItem(label)
        layout.addWidget(combo)
        return combo

    def _slider(self, layout, name, lo, hi, val, fmt=None):
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0); h.setSpacing(8)
        h.addWidget(QLabel(name))
        sl = QSlider(Qt.Orientation.Horizontal)
        sl.setRange(lo, hi); sl.setValue(val)
        val_label = QLabel(fmt(val) if fmt else str(val))
        val_label.setMinimumWidth(60)
        val_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        h.addWidget(sl, 1); h.addWidget(val_label)
        sl.valueChanged.connect(lambda v: val_label.setText(fmt(v) if fmt else str(v)))
        layout.addWidget(row)
        return sl

    @staticmethod
    def _pick(combo, items, value):
        for i, (_, v) in enumerate(items):
            if v == value:
                combo.setCurrentIndex(i)
                return

    # --------------------------------------------------------- param access

    def _algo(self):
        return ALGORITHMS[self.algo_combo.currentIndex()][1]

    def _compare_algo(self):
        i = self.compare_combo.currentIndex()
        return None if i <= 0 else ALGORITHMS[i - 1][1]

    def _fractal(self):
        return FRACTALS[self.fractal_combo.currentIndex()][1]

    def _fade(self):
        return FADES[self.fade_combo.currentIndex()][1]

    def _gset(self):
        return GRADSETS[self.gset_combo.currentIndex()][1]

    def _shade(self):
        return SHADE_MODES[self.shade_combo.currentIndex()][1]

    def _algo_kwargs(self, algo, three_d=False):
        if algo == "value":
            return {"fade": self._fade()}
        if algo == "perlin":
            kw = {"fade": self._fade()}
            if not three_d:
                kw["gradient_set"] = self._gset()
            return kw
        if algo == "worley":
            kw = {"metric": METRICS[self.metric_combo.currentIndex()][1],
                  "cell": CELLMODES[self.cell_combo.currentIndex()][1]}
            if not three_d:
                kw["points"] = self.points_spin.value()
                kw["jitter"] = self.jitter_slider.value() / 100.0
            return kw
        if algo == "gabor":
            return {"f0": self.f0_slider.value() / 2.0}
        return {}

    # -------------------------------------------------------------- lessons

    def _step_lesson(self, d):
        row, n = self.lesson_combo.currentIndex(), self.lesson_combo.count()
        for _ in range(n):
            row = (row + d) % n
            if self._rows[row] is not None:
                self.lesson_combo.setCurrentIndex(row)
                return

    def _on_lesson_selected(self, row):
        lesson = self._rows[row] if row < len(self._rows) else 0
        if lesson is None:                        # header row -> jump to part start
            target = next((r for r in range(row + 1, len(self._rows))
                           if self._rows[r] is not None), 0)
            self.lesson_combo.setCurrentIndex(target)
            return
        self._lesson_idx = lesson
        self.lesson_group.setVisible(lesson > 0)
        if lesson == 0:
            self._update_control_states()
            self._update_info()
            return
        les = LESSONS[lesson - 1]
        p = les["preset"]
        widgets = [self.algo_combo, self.compare_combo, self.fractal_combo,
                   self.fade_combo, self.gset_combo, self.metric_combo,
                   self.cell_combo, self.cmap_combo, self.shade_combo,
                   self.freq_slider, self.oct_slider, self.gain_slider,
                   self.lac_slider, self.jitter_slider, self.f0_slider,
                   self.beta_slider, self.z_slider, self.speed_slider,
                   self.warp_slider, self.seed_spin, self.points_spin,
                   self.three_d_cb, self.warp_cb, self.grid_cb, self.feat_cb,
                   self.contour_spin, self.terrace_spin, self.periodic_cb,
                   self.anatomy_cb, self.dot_cb, self.flow_cb, self.rotate_cb]
        for w in widgets:
            w.blockSignals(True)
        self._pick(self.algo_combo, ALGORITHMS, p.get("algo", "perlin"))
        cmpv = p.get("cmp")
        self.compare_combo.setCurrentIndex(
            0 if cmpv is None else 1 + [a for _, a in ALGORITHMS].index(cmpv))
        self._pick(self.fractal_combo, FRACTALS, p.get("kind", "single"))
        self._pick(self.fade_combo, FADES, p.get("fade", "quintic"))
        self._pick(self.gset_combo, GRADSETS, p.get("gset", "classic8"))
        self._pick(self.metric_combo, METRICS, p.get("metric", "euclidean"))
        self._pick(self.cell_combo, CELLMODES, p.get("cell", "f1"))
        self._pick(self.cmap_combo, [(n, n) for n in _COLORMAPS], p.get("cmap", "Viridis"))
        self._pick(self.shade_combo, SHADE_MODES, p.get("shade", "flat"))
        self.freq_slider.setValue(p.get("freq", 4))
        self.oct_slider.setValue(p.get("oct", 4))
        self.gain_slider.setValue(int(p.get("gain", 0.5) * 100))
        self.lac_slider.setValue(int(p.get("lac", 2.0) * 10))
        self.jitter_slider.setValue(int(p.get("jitter", 1.0) * 100))
        self.f0_slider.setValue(int(p.get("f0", 2.0) * 2))
        self.beta_slider.setValue(int(p.get("beta", 1.0) * 10))
        self.points_spin.setValue(p.get("points", 1))
        self.speed_slider.setValue(int(p.get("speed", 10)))
        self.three_d_cb.setChecked(p.get("three_d", False))
        self.z_slider.setValue(int(round(p.get("z", 0.0) * 10)))
        self.warp_cb.setChecked(p.get("warp", False))
        self.warp_slider.setValue(int(p.get("warp_k", 1.0) * 10))
        self.grid_cb.setChecked(p.get("grid", False))
        self.feat_cb.setChecked(p.get("feat", False))
        self.contour_spin.setValue(p.get("contours", 0))
        self.terrace_spin.setValue(p.get("terrace", 0))
        self.periodic_cb.setChecked(p.get("periodic", False))
        self.anatomy_cb.setChecked(p.get("anatomy", False))
        self.dot_cb.setChecked(p.get("dotprobe", False))
        self.flow_cb.setChecked(p.get("flow", False))
        self.rotate_cb.setChecked(p.get("rotate", False))
        if "seed" in p:
            self.seed_spin.setValue(p["seed"])
        for w in widgets:
            w.blockSignals(False)
        self.lesson_title.setText(f"Lesson {lesson} — {les['title']}")
        self.lesson_text.setHtml(les["html"])
        if self._playing:
            self.play_btn.setChecked(False)
        self._update_control_states()
        self._update_info()
        self._recompute()

    def _schedule(self, *_):
        if self._lesson_idx > 0:                     # manual tweak -> exit lesson
            self._lesson_idx = 0
            self.lesson_combo.blockSignals(True)
            self.lesson_combo.setCurrentIndex(0)
            self.lesson_combo.blockSignals(False)
            self.lesson_group.hide()
        self._debounce.start()

    # ------------------------------------------------------------ recompute

    def _compute_field(self, algo, n, freq, seed, z=None, want_octaves=False):
        if algo == "white":
            return white_noise((n, n), seed), None
        if algo == "spectral":
            return spectral_noise((n, n), seed,
                                  beta=self.beta_slider.value() / 10.0), None
        if z is not None and algo not in _ALGO_FN3:
            z = None            # gabor has no 3-D form; render a static slice
        xs = (np.arange(n) + 0.5) * (freq / n)
        X, Y = np.meshgrid(xs, xs)
        fn = globals()[f"{algo}_noise"] if z is None else _ALGO_FN3[algo]
        kw = self._algo_kwargs(algo, three_d=(z is not None))
        periodic = (z is None and self.periodic_cb.isChecked()
                    and algo in ("value", "perlin", "worley"))
        if periodic:
            kw = {**kw, "period": int(round(freq))}
        rotate = (self.rotate_cb.isChecked()
                  and self._fractal() != "single" and not periodic)
        if self.warp_cb.isChecked() and self.warp_slider.value() > 0:
            X, Y = domain_warp_coords(fn, X, Y, z=z,
                                      strength=self.warp_slider.value() / 10.0,
                                      seed=seed + 555, **kw)
        oct_data = None
        if self._fractal() == "single":
            field = fn(X, Y, seed=seed, **kw) if z is None else fn(X, Y, z, seed=seed, **kw)
        else:
            params = dict(seed=seed, octaves=self.oct_slider.value(),
                          gain=self.gain_slider.value() / 100.0,
                          lacunarity=self.lac_slider.value() / 10.0,
                          kind=self._fractal(), rotate=rotate,
                          period=int(round(freq)) if periodic else None, **kw)
            if z is None:
                field = fractal2d(fn, X, Y, **params)
                if want_octaves:
                    oct_data = octave_fields(fn, X, Y, **params)
            else:
                field = fractal3d(fn, X, Y, z,
                                  **{k: v for k, v in params.items() if k != "period"})
                if want_octaves:
                    oct_data = octave_fields(fn, X, Y, z=z, **params)
        return field, oct_data

    # ---------------------------------------------------------- flow overlay

    def _update_flow(self, algo, freq, seed, z, cmp_on):
        ok = (self.flow_cb.isChecked() and z is None and not cmp_on
              and algo not in ("white",))
        if not ok:
            self.canvas.set_gradient_overlay(None)
            self.canvas.set_streamline([], [])
            self._flowgrad = None
            return
        m = 16
        if algo in ("perlin", "value"):               # exact analytic derivatives
            xs = np.linspace(0.0, freq, m)
            X, Y = np.meshgrid(xs, xs)
            if algo == "perlin":
                _, gx, gy = perlin_noise_d(X, Y, seed=seed, gradient_set=self._gset())
            else:
                _, gx, gy = value_noise_d(X, Y, seed=seed, fade=self._fade())
            step = freq / (m - 1)
            self._flowgrad = (gx, gy, step)
        else:                                         # finite differences
            f = self._fieldA
            gyy, gxx = np.gradient(f)
            step_full = freq / (f.shape[0] - 1)
            k = max(1, f.shape[0] // m)
            gx, gy = gxx[::k, ::k], gyy[::k, ::k]
            step = step_full * k
            self._flowgrad = (gxx, gyy, step_full)
        self.canvas.set_gradient_overlay(gx, gy, step)

    def _grad_at(self, x, y):
        if self._flowgrad is None:
            return 0.0, 0.0
        gx, gy, step = self._flowgrad
        H, W = gx.shape
        fx = np.clip(x / step, 0.0, W - 1.001)
        fy = np.clip(y / step, 0.0, H - 1.001)
        i, j = int(fx), int(fy)
        tx, ty = fx - i, fy - j
        a = gx[j, i] * (1 - tx) + gx[j, i + 1] * tx
        b = gx[j + 1, i] * (1 - tx) + gx[j + 1, i + 1] * tx
        c = gy[j, i] * (1 - tx) + gy[j, i + 1] * tx
        d = gy[j + 1, i] * (1 - tx) + gy[j + 1, i + 1] * tx
        return float(a * (1 - ty) + b * ty), float(c * (1 - ty) + d * ty)

    def _update_streamline(self, nx, ny):
        if self._flowgrad is None:
            self.canvas.set_streamline([], [])
            return
        freq = self._freq
        h = 0.02

        def walk(sign):
            pts = [(nx, ny)]
            x, y = nx, ny
            for _ in range(400):
                gx, gy = self._grad_at(x, y)
                m = math.hypot(gx, gy)
                if m < 1e-5:
                    break
                x += sign * h * gx / m
                y += sign * h * gy / m
                if not (-0.05 <= x <= freq + 0.05 and -0.05 <= y <= freq + 0.05):
                    break
                pts.append((min(max(x, 0.0), freq), min(max(y, 0.0), freq)))
            return pts

        self.canvas.set_streamline(walk(+1.0), walk(-1.0))

    # ------------------------------------------------------------ recompute

    def _recompute(self):
        n = RESOLUTIONS[self.res_combo.currentIndex()]
        freq = float(self.freq_slider.value())
        seed = self.seed_spin.value()
        algo, cmp_algo = self._algo(), self._compare_algo()
        z = self.z_slider.value() / 10.0 if self.three_d_cb.isChecked() else None
        lut = _COLORMAPS[self.cmap_combo.currentText()]
        self._lut = lut
        shade, contours = self._shade(), self.contour_spin.value()
        terrace = self.terrace_spin.value()

        want_oct = (not self._playing and self._fractal() != "single"
                    and algo not in ("white", "spectral"))
        self._fieldA, oct_data = self._compute_field(algo, n, freq, seed, z, want_oct)
        self._fieldB = (self._compute_field(cmp_algo, n, freq, seed, z)[0]
                        if cmp_algo is not None else None)

        overlays_ok = (z is None) and (cmp_algo is None)
        # feature dots are only faithful for Worley's default point layout
        feat_ok = (self.feat_cb.isChecked() and overlays_ok
                   and not (algo == "worley"
                            and (self.points_spin.value() > 1
                                 or self.jitter_slider.value() < 100)))
        self.canvas.set_field(self._fieldA, freq, lut, algo, seed,
                              self.grid_cb.isChecked() and overlays_ok,
                              feat_ok,
                              shade, contours, terrace,
                              self._gset() if algo == "perlin" else "classic8")
        self.canvas.set_compare(self._fieldB, lut, cmp_algo, shade, contours, terrace)
        if want_oct and oct_data is not None:
            self.strip.set_octaves(*oct_data, lut)
        else:
            self.strip.clear()
        self.spectrum.set_field(self._fieldA)
        self.hist.set_field(self._fieldA)
        self.anatomy.set_colormap(lut)
        self._update_flow(algo, freq, seed, z, cmp_algo is not None)
        self._update_control_states()
        self._update_info()
        self._refresh_cross()

    # ------------------------------------------------------------- playback

    def _toggle_play(self, on):
        self._playing = on
        self.play_btn.setText("\u23f8  Pause" if on else "\u25b6  Animate")
        if on:
            if not self.three_d_cb.isChecked():
                self.three_d_cb.setChecked(True)
            self._anim.start()
        else:
            self._anim.stop()

    def _stop_play(self):
        if self._playing:
            self.play_btn.setChecked(False)

    def _tick(self):
        v = self.z_slider.value() + self.speed_slider.value()
        if v > 100:
            v = 0
        self.z_slider.blockSignals(True)
        self.z_slider.setValue(v)
        self.z_slider.blockSignals(False)
        self._recompute()

    # ----------------------------------------------- cross-section + probes

    def _refresh_cross(self, nx=None, ny=None, side=None):
        if self._fieldA is None:
            return
        if nx is None:
            nx, ny = self.canvas.hover()
            side = self.canvas.hover_side()
        freq = self._freq
        nx = float(np.clip(nx, 0.0, freq)); ny = float(np.clip(ny, 0.0, freq))
        n = self._fieldA.shape[0]
        row = int(np.clip(round(ny / freq * n - 0.5), 0, n - 1))
        col = int(np.clip(round(nx / freq * n - 0.5), 0, n - 1))
        fa, fb = self._fieldA, self._fieldB
        primary_a = not (side == "B" and fb is not None)
        main, other = (fa, fb) if primary_a else (fb, fa)
        algo_now = self._algo() if primary_a else (self._compare_algo() or self._algo())
        square = algo_now in ("value", "perlin", "worley")
        self.cross.set_data(main[row, :], freq, nx,
                            ticks=(square and self.grid_cb.isChecked() and fb is None),
                            values2=None if other is None else other[row, :],
                            primary="A" if primary_a else "B")
        self.readout.setText(
            f"{'A' if primary_a else 'B'} · x = {nx:5.2f}   y = {ny:5.2f}   "
            f"noise = {main[row, col]:.3f}")
        three_d = self.three_d_cb.isChecked()
        if (self.dot_cb.isChecked() and self.dot_cb.isEnabled() and not three_d
                and self._algo() in ("perlin", "simplex")):
            self.dotdiag.set_probe(self._algo(), nx, ny, self.seed_spin.value(),
                                   self._gset() if self._algo() == "perlin"
                                   else "classic8")
        elif self.dot_cb.isChecked():
            self.dotdiag.clear()
        if (self.anatomy_cb.isChecked() and self.anatomy_cb.isEnabled()
                and not three_d and self._algo() in ("perlin", "value")):
            self.anatomy.set_probe(self._algo(), nx, ny, self.seed_spin.value(),
                                   self._fade())
        elif self.anatomy_cb.isChecked():
            self.anatomy.clear()
        if self._flowgrad is not None:
            self._update_streamline(nx, ny)

    def _on_hover(self, nx, ny, side):
        self._refresh_cross(nx, ny, side)

    # ------------------------------------------------------------- UI state

    def _update_control_states(self):
        algo, cmp_on = self._algo(), self._compare_algo() is not None
        white, three_d = algo == "white", self.three_d_cb.isChecked()
        spectral = algo == "spectral"
        plain = white or spectral
        single = self._fractal() == "single"
        lattice3d = algo in ("value", "perlin", "simplex", "worley")
        visual_ok = not white
        self.shade_combo.setEnabled(visual_ok)
        self.contour_spin.setEnabled(visual_ok)
        self.terrace_spin.setEnabled(visual_ok)
        self.fade_group.setVisible(algo in ("value", "perlin"))
        self.fadec.set_active(self._fade() if (algo in ("value", "perlin") and not three_d)
                              else None)
        periodic_ok = (algo in ("value", "perlin", "worley") and not three_d and not plain)
        self.periodic_cb.setEnabled(periodic_ok)
        self.tile_preview.setVisible(self.periodic_cb.isChecked() and periodic_ok)
        if self.tile_preview.isVisible():
            self.tile_preview.set_image(self.canvas.image())
        anatomy_ok = algo in ("perlin", "value") and not three_d and not white
        self.anatomy_cb.setEnabled(anatomy_ok)
        self.anatomy_group.setVisible(self.anatomy_cb.isChecked() and anatomy_ok)
        dot_ok = algo in ("perlin", "simplex") and not three_d and not white
        self.dot_cb.setEnabled(dot_ok)
        self.dot_group.setVisible(self.dot_cb.isChecked() and dot_ok)
        flow_ok = not white and not three_d and not cmp_on
        self.flow_cb.setEnabled(flow_ok)
        self.three_d_cb.setEnabled(lattice3d)
        self.z_slider.setEnabled(three_d and lattice3d)
        self.speed_slider.setEnabled(three_d and lattice3d)
        self.play_btn.setEnabled(three_d and lattice3d)
        self.freq_slider.setEnabled(not plain)
        self.fractal_combo.setEnabled(not plain)
        self.rotate_cb.setEnabled(not plain and not single and algo != "gabor")
        self.fade_combo.setEnabled(not white and not three_d and algo in ("value", "perlin"))
        self.gset_combo.setEnabled(algo == "perlin" and not three_d)
        self.metric_combo.setEnabled(algo == "worley")
        self.cell_combo.setEnabled(algo == "worley")
        self.points_spin.setEnabled(algo == "worley" and not three_d)
        self.jitter_slider.setEnabled(algo == "worley" and not three_d)
        self.f0_slider.setEnabled(algo == "gabor")
        self.beta_slider.setEnabled(spectral)
        self.oct_slider.setEnabled(not (plain or single))
        self.gain_slider.setEnabled(not (plain or single))
        self.lac_slider.setEnabled(not (plain or single))
        self.warp_cb.setEnabled(not plain and algo != "spectral")
        self.warp_slider.setEnabled(not plain and self.warp_cb.isChecked())
        self.grid_cb.setEnabled(not white and not three_d and not cmp_on)
        self.feat_cb.setEnabled(algo in ("value", "perlin", "simplex", "worley")
                                and not three_d and not cmp_on)
        self.strip.setVisible(not plain and not single)

    def _update_info(self):
        algo = self._algo()
        extra = ""
        if algo == "gabor":
            extra += GABOR_INFO
        if algo == "spectral":
            extra += SPECTRAL_INFO
        if self._fractal() == "hybrid":
            extra += MULTIFRACTAL_INFO
        if self.rotate_cb.isChecked() and self.rotate_cb.isEnabled():
            extra += ROTATION_INFO
        if algo == "perlin" and self.gset_combo.isEnabled():
            extra += GRADSET_INFO
        if self.flow_cb.isChecked() and self.flow_cb.isEnabled():
            extra += DERIV_FLOW_INFO
        if self.three_d_cb.isChecked():
            extra += Z_INFO
        if self.warp_cb.isChecked() and self.warp_slider.value() > 0:
            extra += WARP_INFO
        if self._compare_algo() is not None:
            extra += COMPARE_INFO
        if self.dot_cb.isChecked() and self.dot_cb.isEnabled():
            extra += DOTPROBE_INFO
        if self.anatomy_cb.isChecked() and self.anatomy_cb.isEnabled():
            extra += ANATOMY_INFO
        if self.periodic_cb.isChecked() and self.periodic_cb.isEnabled():
            extra += SEAMLESS_INFO
        if self.terrace_spin.value() >= 2:
            extra += TERRACE_INFO
        if self._shade() == "normals":
            extra += NORMALMAP_INFO
        if self.shade_combo.currentIndex() == 1 or self.contour_spin.value() >= 2:
            extra += ("<hr><h4>Reading the render</h4>"
                      "<p><b>Relief</b> estimates each pixel's surface normal and applies "
                      "Lambert lighting; <b>contours</b> mark every 1/N elevation step.</p>")
        self.info.setHtml(ALGO_INFO.get(algo, "")
                          + FRACTAL_INFO.get(self._fractal(), "") + extra + TIP_HTML)


# =====================================================================
#  Theme & entry point
# =====================================================================

def _apply_dark_palette(app: QApplication):
    p = QPalette()
    p.setColor(QPalette.ColorRole.Window, QColor("#171a21"))
    p.setColor(QPalette.ColorRole.WindowText, QColor("#d7dbe4"))
    p.setColor(QPalette.ColorRole.Base, QColor("#12151c"))
    p.setColor(QPalette.ColorRole.AlternateBase, QColor("#1a1e27"))
    p.setColor(QPalette.ColorRole.Text, QColor("#d7dbe4"))
    p.setColor(QPalette.ColorRole.Button, QColor("#20242e"))
    p.setColor(QPalette.ColorRole.ButtonText, QColor("#d7dbe4"))
    p.setColor(QPalette.ColorRole.Highlight, QColor("#3f7fd4"))
    p.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    p.setColor(QPalette.ColorRole.ToolTipBase, QColor("#20242e"))
    p.setColor(QPalette.ColorRole.ToolTipText, QColor("#d7dbe4"))
    for role in (QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText,
                 QPalette.ColorRole.WindowText):
        p.setColor(QPalette.ColorGroup.Disabled, role, QColor("#6b7280"))
    app.setPalette(p)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    _apply_dark_palette(app)
    app.setStyleSheet(GLOBAL_QSS)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()