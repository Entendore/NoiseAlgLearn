#!/usr/bin/env python3
"""
Noise Lab — UNIFIED EDITION. The complete 3-D noise learning app in one file.

    pip install PySide6 numpy            (required)
    pip install pyqtgraph PyOpenGL       (optional: 3-D volume tab)
    pip install PyOpenGL                 (optional: GPU GLSL preview tab)

    python noise_lab.py

CONTENTS
  * 25 algorithms across 12 families (white/value/perlin '85+improved/simplex/
    worley+metrics+cells/gabor/craters/fourier/midpoint/wavelet/colored/hyper-ND/
    tiled/torus/dual-lattice/curl x2/flow/rotated fBm/musgrave x2/marble/
    strata/reaction-diffusion)
  * 43 guided lessons with one-click setups and progress tracking
  * 9 tabs: 2-D slice · 3-D volume · GPU GLSL preview · Compare A/B ·
    Octave lab · Cell anatomy · Spectrum & stats · Terrain lab · Final exam
  * View modes: field, gradient, iso-lines, shaded relief; planes XY/XZ/YZ;
    fade-curve explorer; domain warp; terracing; analytic-gradient arrows;
    Worley feature points; Perlin corner-gradient arrows; hover readout;
    colorbar; 1-D profile; PNG slice + volume-stack export; "view source".
"""

import inspect
import math
import os
import sys
import time

import numpy as np
from PySide6.QtCore import (QPointF, QObject, QRunnable, QRect, Qt, QTimer,
                            QThreadPool, Signal)
from PySide6.QtGui import (QColor, QFont, QImage, QPainter, QPalette, QPen,
                           QPolygonF, QSurfaceFormat)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox,
                               QDoubleSpinBox, QDialog, QFileDialog, QFormLayout,
                               QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                               QMainWindow, QMessageBox, QPlainTextEdit,
                               QPushButton, QScrollArea, QSlider, QSizePolicy,
                               QSpinBox, QSplitter, QTabWidget, QTextBrowser,
                               QVBoxLayout, QWidget)

try:                                                     # optional 3-D volume
    import pyqtgraph.opengl as _gl
    _GL_OK = True
except Exception:
    _gl = None
    _GL_OK = False
try:                                                     # optional GPU preview
    from OpenGL import GL as gl
    from PySide6.QtOpenGLWidgets import QOpenGLWidget
    _GLW_OK = True
except Exception:
    gl = None
    QOpenGLWidget = QWidget
    _GLW_OK = False


# ============================================================================
#  1. CORE MATH
# ============================================================================

_QUINTIC = lambda t: t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
_lerp = lambda a, b, t: a + t * (b - a)

# Active fade curve (swap via the Interpolation panel; algorithms look this
# up at call time, so the override affects Value/Perlin/Tiled live).
FADE_CURVES = {
    "nearest · no blend":     lambda t: np.where(t >= 0.5, 1.0, 0.0),
    "linear (C⁰)":            lambda t: t,
    "smoothstep (C¹)":        lambda t: t * t * (3.0 - 2.0 * t),
    "quintic (C²) · default": _QUINTIC,
}
DEFAULT_FADE = "quintic (C²) · default"
_fade = _QUINTIC


def set_fade(name):
    global _fade
    _fade = FADE_CURVES.get(name, _QUINTIC)


def _permutation(seed):
    rng = np.random.default_rng(seed)
    p = rng.permutation(256).astype(np.int64)
    return np.concatenate([p, p])


def _hash01(ix, iy, iz, k):
    """Deterministic pseudo-random float in [0, 1) from 3 ints + a seed."""
    h = (ix * 374761393 + iy * 668265263 +
         iz * 1274126177 + np.int64(k) * np.int64(2654435761))
    h = h & 0xFFFFFFFF
    h = h ^ (h >> 13)
    h = (h * 1103515245) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return (h & 0xFFFFFF).astype(np.float64) / float(1 << 24)


_GRAD3 = np.array([
    [ 1,  1, 0], [-1,  1, 0], [ 1, -1, 0], [-1, -1, 0],
    [ 1,  0, 1], [-1,  0, 1], [ 1,  0, -1], [-1,  0, -1],
    [ 0,  1, 1], [ 0, -1, 1], [ 0,  1, -1], [ 0, -1, -1]], dtype=np.float64)


# ============================================================================
#  2. ALGORITHMS — LATTICE FAMILY
# ============================================================================

class WhiteNoise3:
    """One independent random number per lattice cell, NO interpolation —
    the flat-spectrum baseline every other algorithm improves upon."""

    def __init__(self, seed=0):
        self.seed = int(seed)

    def __call__(self, x, y, z):
        return _hash01(np.floor(x).astype(np.int64),
                       np.floor(y).astype(np.int64),
                       np.floor(z).astype(np.int64), self.seed) * 2.0 - 1.0


class ValueNoise3:
    """Random value per lattice corner, blended with the (active) fade curve.
    Cheapest smooth noise; features visibly align with the grid."""

    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.perm = _permutation(seed)
        self.values = rng.random(256) * 2.0 - 1.0

    def _lattice(self, X, Y, Z):
        h = self.perm[self.perm[self.perm[X & 255] + (Y & 255)] + (Z & 255)]
        return self.values[h]

    def __call__(self, x, y, z):
        fx, fy, fz = np.floor(x), np.floor(y), np.floor(z)
        X, Y, Z = fx.astype(np.int64), fy.astype(np.int64), fz.astype(np.int64)
        u, v, w = _fade(x - fx), _fade(y - fy), _fade(z - fz)
        v000 = self._lattice(X, Y, Z);         v100 = self._lattice(X + 1, Y, Z)
        v010 = self._lattice(X, Y + 1, Z);     v110 = self._lattice(X + 1, Y + 1, Z)
        v001 = self._lattice(X, Y, Z + 1);     v101 = self._lattice(X + 1, Y, Z + 1)
        v011 = self._lattice(X, Y + 1, Z + 1); v111 = self._lattice(X + 1, Y + 1, Z + 1)
        return _lerp(
            _lerp(_lerp(v000, v100, u), _lerp(v010, v110, u), v),
            _lerp(_lerp(v001, v101, u), _lerp(v011, v111, u), v), w)


class PerlinNoise3:
    """Improved Perlin (2002): quintic fade, 12 edge-midpoint gradients via
    Perlin's bit-trick. Field crosses zero on every cell face."""

    def __init__(self, seed=0):
        self.perm = _permutation(seed)

    @staticmethod
    def _grad(h, x, y, z):
        u = np.where(h < 8, x, y)
        v = np.where(h < 4, y, np.where((h == 12) | (h == 14), x, z))
        return np.where((h & 1) == 0, u, -u) + np.where((h & 2) == 0, v, -v)

    def __call__(self, x, y, z):
        p = self.perm
        fx, fy, fz = np.floor(x), np.floor(y), np.floor(z)
        X, Y, Z = fx.astype(np.int64) & 255, fy.astype(np.int64) & 255, fz.astype(np.int64) & 255
        xf, yf, zf = x - fx, y - fy, z - fz
        u, v, w = _fade(xf), _fade(yf), _fade(zf)
        A = p[X] + Y;     AA = p[A] + Z;     AB = p[A + 1] + Z
        B = p[X + 1] + Y; BA = p[B] + Z;     BB = p[B + 1] + Z
        x00 = _lerp(self._grad(p[AA], xf, yf, zf),
                    self._grad(p[BA], xf - 1, yf, zf), u)
        x10 = _lerp(self._grad(p[AB], xf, yf - 1, zf),
                    self._grad(p[BB], xf - 1, yf - 1, zf), u)
        x01 = _lerp(self._grad(p[AA + 1], xf, yf, zf - 1),
                    self._grad(p[BA + 1], xf - 1, yf, zf - 1), u)
        x11 = _lerp(self._grad(p[AB + 1], xf, yf - 1, zf - 1),
                    self._grad(p[BB + 1], xf - 1, yf - 1, zf - 1), u)
        return _lerp(_lerp(x00, x10, v), _lerp(x01, x11, v), w)


class PerlinClassic3:
    """Perlin's ORIGINAL 1985 formulation: cubic fade 3t²−2t³ (only C¹ —
    second derivative creases on cell faces) and cube-corner gradients
    (±1,±1,±1)/√3 (diagonal bias the 2002 paper removed)."""

    def __init__(self, seed=0):
        self.perm = _permutation(seed)

    @staticmethod
    def _grad(h, x, y, z):
        i = h & 7
        gx = np.where(i & 1, -1.0, 1.0)
        gy = np.where(i & 2, -1.0, 1.0)
        gz = np.where(i & 4, -1.0, 1.0)
        return (gx * x + gy * y + gz * z) * 0.5773502691896258

    def __call__(self, x, y, z):
        p = self.perm
        fx, fy, fz = np.floor(x), np.floor(y), np.floor(z)
        X, Y, Z = fx.astype(np.int64) & 255, fy.astype(np.int64) & 255, fz.astype(np.int64) & 255
        xf, yf, zf = x - fx, y - fy, z - fz
        f = lambda t: t * t * (3.0 - 2.0 * t)
        u, v, w = f(xf), f(yf), f(zf)
        A = p[X] + Y;     AA = p[A] + Z;     AB = p[A + 1] + Z
        B = p[X + 1] + Y; BA = p[B] + Z;     BB = p[B + 1] + Z
        x00 = _lerp(self._grad(p[AA], xf, yf, zf),
                    self._grad(p[BA], xf - 1, yf, zf), u)
        x10 = _lerp(self._grad(p[AB], xf, yf - 1, zf),
                    self._grad(p[BB], xf - 1, yf - 1, zf), u)
        x01 = _lerp(self._grad(p[AA + 1], xf, yf, zf - 1),
                    self._grad(p[BA + 1], xf - 1, yf, zf - 1), u)
        x11 = _lerp(self._grad(p[AB + 1], xf, yf - 1, zf - 1),
                    self._grad(p[BB + 1], xf - 1, yf - 1, zf - 1), u)
        return _lerp(_lerp(x00, x10, v), _lerp(x01, x11, v), w)


class SimplexNoise3:
    """Simplex (Perlin 2001): tetrahedral grid — 4 corners per sample instead
    of 8, radial kernels (0.6−r²)⁴·(g·d)."""

    def __init__(self, seed=0):
        perm = _permutation(seed)
        self.perm, self.perm_mod12 = perm, perm % 12

    def __call__(self, x, y, z):
        F3, G3 = 1.0 / 3.0, 1.0 / 6.0
        s = (x + y + z) * F3
        i = np.floor(x + s).astype(np.int64)
        j = np.floor(y + s).astype(np.int64)
        k = np.floor(z + s).astype(np.int64)
        t = (i + j + k) * G3
        x0, y0, z0 = x - (i - t), y - (j - t), z - (k - t)

        # Which tetrahedron?  Rank (x0,y0,z0) — the 6 possible orderings.
        a, b, c = x0 >= y0, y0 >= z0, x0 >= z0
        i1 = np.where(a, np.where(b | c, 1, 0), 0)
        j1 = np.where(a, 0, np.where(b, 1, 0))            # FIX: z≥y≥x case
        k1 = np.where(a, np.where(b, 0, np.where(c, 0, 1)),
                         np.where(b, 0, 1))
        i2 = np.where(a, 1, np.where(~b, 0, np.where(c, 1, 0)))
        j2 = np.where(a, np.where(b, 1, 0), 1)
        k2 = np.where(a, np.where(b, 0, 1),
                      np.where(b, np.where(c, 0, 1), 1))  # FIX: z≥y≥x case

        x1, y1, z1 = x0 - i1 + G3,     y0 - j1 + G3,     z0 - k1 + G3
        x2, y2, z2 = x0 - i2 + 2 * G3, y0 - j2 + 2 * G3, z0 - k2 + 2 * G3
        x3, y3, z3 = x0 - 1.0 + 3 * G3, y0 - 1.0 + 3 * G3, z0 - 1.0 + 3 * G3

        ii, jj, kk = i & 255, j & 255, k & 255
        pm, p = self.perm_mod12, self.perm
        gi0 = pm[ii + p[jj + p[kk]]]
        gi1 = pm[ii + i1 + p[jj + j1 + p[kk + k1]]]
        gi2 = pm[ii + i2 + p[jj + j2 + p[kk + k2]]]
        gi3 = pm[ii + 1 + p[jj + 1 + p[kk + 1]]]

        def corner(tq, qx, qy, qz, gi):
            tq = np.where(tq < 0.0, 0.0, tq)
            t2 = tq * tq
            g = _GRAD3[gi]
            return t2 * t2 * (g[..., 0] * qx + g[..., 1] * qy + g[..., 2] * qz)

        n0 = corner(0.6 - x0*x0 - y0*y0 - z0*z0, x0, y0, z0, gi0)
        n1 = corner(0.6 - x1*x1 - y1*y1 - z1*z1, x1, y1, z1, gi1)
        n2 = corner(0.6 - x2*x2 - y2*y2 - z2*z2, x2, y2, z2, gi2)
        n3 = corner(0.6 - x3*x3 - y3*y3 - z3*z3, x3, y3, z3, gi3)
        return 32.0 * (n0 + n1 + n2 + n3)


class WorleyNoise3:
    """Cellular noise: one hashed feature point per cell; scan the 3×3×3
    neighbourhood, return F1/F2/(F2−F1) distances — or 'Cells' = hashed id of
    the nearest cell (a Voronoi diagram). METRIC is a design parameter:
    Euclidean (round), Manhattan (diamonds), Chebyshev (squares), Minkowski 3."""

    SCALES = {"Euclidean": 1.1, "Manhattan": 2.4, "Chebyshev": 1.3, "Minkowski 3": 1.7}

    def __init__(self, seed=0, metric="Euclidean", mode="F1"):
        self.seed, self.metric, self.mode = int(seed), metric, mode

    def _dist(self, dx, dy, dz):
        if self.metric == "Manhattan":
            return np.abs(dx) + np.abs(dy) + np.abs(dz)
        if self.metric == "Chebyshev":
            return np.maximum(np.maximum(np.abs(dx), np.abs(dy)), np.abs(dz))
        if self.metric == "Minkowski 3":
            return (np.abs(dx) ** 3 + np.abs(dy) ** 3 + np.abs(dz) ** 3) ** (1.0 / 3.0)
        return np.sqrt(dx * dx + dy * dy + dz * dz)

    def __call__(self, x, y, z):
        xi, yi, zi = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64), np.floor(z).astype(np.int64)
        f1 = np.full(x.shape, np.inf)
        f2 = np.full(x.shape, np.inf)
        bx = np.zeros(x.shape, dtype=np.int64)
        by = np.zeros(x.shape, dtype=np.int64)
        bz = np.zeros(x.shape, dtype=np.int64)
        s = self.seed
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    cx, cy, cz = xi + dx, yi + dy, zi + dz
                    px = cx + _hash01(cx, cy, cz, s)
                    py = cy + _hash01(cx, cy, cz, s + 7919)
                    pz = cz + _hash01(cx, cy, cz, s + 15485)
                    d = self._dist(px - x, py - y, pz - z)
                    better = d < f1
                    f2 = np.where(better, f1, np.minimum(f2, d))
                    bx, by, bz = (np.where(better, cx, bx),
                                  np.where(better, cy, by),
                                  np.where(better, cz, bz))
                    f1 = np.minimum(f1, d)
        if self.mode == "Cells":
            return _hash01(bx, by, bz, s + 777) * 2.0 - 1.0
        sc = self.SCALES[self.metric]
        f1, f2 = f1 / sc, f2 / sc
        r = f1 if self.mode == "F1" else (f2 if self.mode == "F2" else f2 - f1)
        return np.clip(r, 0.0, 1.0) * 2.0 - 1.0

    def slice_feature_points(self, n, freq, k):
        s = self.seed
        return [((i + _hash01(i, j, k, s)) / freq,
                 (j + _hash01(i, j, k, s + 7919)) / freq)
                for i in range(n) for j in range(n)]


class TiledLatticeNoise3:
    """SEAMLESS tiling by hashing lattice indices MODULO a power-of-two period
    P — the field repeats exactly every P units, by construction.
    Worley subtlety: point POSITIONS use unwrapped cells, only the HASH wraps.
    fBm stays seamless only if every octave spans an integer number of
    periods (lacunarity 2, frequency = period)."""

    def __init__(self, seed=0, kind="perlin", period=4):
        period = int(period)
        if period < 2 or (period & (period - 1)) != 0:
            raise ValueError("period must be a power of two (2, 4, 8, 16, ...)")
        self.kind, self.period, self.seed = kind, period, int(seed)
        self.mode = "F1"

    def _h(self, ix, iy, iz, k=0):
        P = self.period - 1
        return _hash01(ix & P, iy & P, iz & P, self.seed + k)

    def __call__(self, x, y, z):
        if self.kind == "worley":
            return self._worley(x, y, z)
        fx, fy, fz = np.floor(x), np.floor(y), np.floor(z)
        X, Y, Z = fx.astype(np.int64), fy.astype(np.int64), fz.astype(np.int64)
        u, v, w = _fade(x - fx), _fade(y - fy), _fade(z - fz)
        if self.kind == "value":
            c = lambda i, j, k: self._h(X + i, Y + j, Z + k) * 2.0 - 1.0
            dots = (c(0, 0, 0), c(1, 0, 0), c(0, 1, 0), c(1, 1, 0),
                    c(0, 0, 1), c(1, 0, 1), c(0, 1, 1), c(1, 1, 1))
        else:
            xf, yf, zf = x - fx, y - fy, z - zf
            corner_idx = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
                          (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1))
            offs = ((xf, yf, zf), (xf-1, yf, zf), (xf, yf-1, zf), (xf-1, yf-1, zf),
                    (xf, yf, zf-1), (xf-1, yf, zf-1), (xf, yf-1, zf-1), (xf-1, yf-1, zf-1))
            dots = []
            for (oi, oj, ok), (dx, dy, dz) in zip(corner_idx, offs):
                gi = np.clip(self._h(X + oi, Y + oj, Z + ok) * 11.999, 0, 11).astype(np.int64)
                g = _GRAD3[gi]
                dots.append(g[..., 0] * dx + g[..., 1] * dy + g[..., 2] * dz)
        n000, n100, n010, n110, n001, n101, n011, n111 = dots
        return _lerp(
            _lerp(_lerp(n000, n100, u), _lerp(n010, n110, u), v),
            _lerp(_lerp(n001, n101, u), _lerp(n011, n111, u), v), w)

    def _worley(self, x, y, z):
        xi, yi, zi = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64), np.floor(z).astype(np.int64)
        f1 = np.full(x.shape, np.inf)
        f2 = np.full(x.shape, np.inf)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    cx, cy, cz = xi + dx, yi + dy, zi + dz
                    px = cx + self._h(cx, cy, cz, 0)       # position: unwrapped
                    py = cy + self._h(cx, cy, cz, 7919)
                    pz = cz + self._h(cx, cy, cz, 15485)
                    d = (px - x) ** 2 + (py - y) ** 2 + (pz - z) ** 2
                    f2 = np.where(d < f1, f1, np.minimum(f2, d))
                    f1 = np.minimum(f1, d)
        f1, f2 = np.sqrt(f1) / 1.1, np.sqrt(f2) / 2.0
        r = f1 if self.mode == "F1" else (f2 if self.mode == "F2" else f2 - f1)
        return np.clip(r, 0.0, 1.0) * 2.0 - 1.0

    def slice_feature_points(self, n, freq, k):
        if self.kind != "worley":
            return []
        return [((i + self._h(i, j, k, 0)) / freq,
                 (j + self._h(i, j, k, 7919)) / freq)
                for i in range(n) for j in range(n)]


# ============================================================================
#  3. ALGORITHMS — DERIVATIVE FAMILY
# ============================================================================

class PerlinDeriv3:
    """Improved Perlin returning (value, ∂N/∂x, ∂N/∂y, ∂N/∂z) in ONE
    evaluation. ∂N/∂x = Σ [ (∂W_c/∂x)(g_c·d_c) + W_c·g_cx ] with
    ∂W/∂x = ±fade'(u)·(other-axis weights), fade'(t) = 30t²(t−1)².
    Fixed in this edition: the corner-gradient decomposition now handles ALL
    12 gradient cases (the earlier version zeroed the z component for
    gradients like (0,±1,±1))."""

    def __init__(self, seed=0):
        self.perm = _permutation(seed)

    @staticmethod
    def _gradc(h, x, y, z):
        """Exact gradient VECTOR of Perlin's bit-trick gradient function."""
        su = np.where((h & 1) == 0, 1.0, -1.0)
        sv = np.where((h & 2) == 0, 1.0, -1.0)
        ux = (h < 8).astype(np.float64)
        uy = 1.0 - ux                                  # u = x if h<8 else y
        vx = ((h == 12) | (h == 14)).astype(np.float64)
        vy = (h < 4).astype(np.float64)                # v = y / x / z
        vz = 1.0 - vx - vy
        return su * ux + sv * vx, su * uy + sv * vy, sv * vz

    def __call__(self, x, y, z):
        p = self.perm
        fx, fy, fz = np.floor(x), np.floor(y), np.floor(z)
        X, Y, Z = fx.astype(np.int64) & 255, fy.astype(np.int64) & 255, fz.astype(np.int64) & 255
        xf, yf, zf = x - fx, y - fy, z - fz
        u, v, w = _QUINTIC(xf), _QUINTIC(yf), _QUINTIC(zf)   # fixed quintic
        du = 30.0 * xf * xf * (xf - 1.0) ** 2
        dv = 30.0 * yf * yf * (yf - 1.0) ** 2
        dw = 30.0 * zf * zf * (zf - 1.0) ** 2

        A = p[X] + Y;     AA = p[A] + Z;     AB = p[A + 1] + Z
        B = p[X + 1] + Y; BA = p[B] + Z;     BB = p[B + 1] + Z
        hh = (p[AA], p[BA], p[AB], p[BB], p[AA + 1], p[BA + 1], p[AB + 1], p[BB + 1])
        offs = ((xf, yf, zf), (xf-1, yf, zf), (xf, yf-1, zf), (xf-1, yf-1, zf),
                (xf, yf, zf-1), (xf-1, yf, zf-1), (xf, yf-1, zf-1), (xf-1, yf-1, zf-1))
        dots = [PerlinNoise3._grad(hh[i], *offs[i]) for i in range(8)]
        comps = [self._gradc(hh[i], *offs[i]) for i in range(8)]
        n000, n100, n010, n110, n001, n101, n011, n111 = dots

        val = _lerp(_lerp(_lerp(n000, n100, u), _lerp(n010, n110, u), v),
                    _lerp(_lerp(n001, n101, u), _lerp(n011, n111, u), v), w)

        dNx = du * ((1-v)*(1-w)*(n100-n000) + (1-v)*w*(n101-n001)
                    + v*(1-w)*(n110-n010) + v*w*(n111-n011))
        dNy = dv * ((1-u)*(1-w)*(n010-n000) + (1-u)*w*(n011-n001)
                    + u*(1-w)*(n110-n100) + u*w*(n111-n101))
        dNz = dw * ((1-u)*(1-v)*(n001-n000) + (1-u)*v*(n011-n010)
                    + u*(1-v)*(n101-n100) + u*v*(n111-n110))

        W = ((1-u)*(1-v)*(1-w), u*(1-v)*(1-w), (1-u)*v*(1-w), u*v*(1-w),
             (1-u)*(1-v)*w, u*(1-v)*w, (1-u)*v*w, u*v*w)
        for i in range(8):
            dNx += W[i] * comps[i][0]
            dNy += W[i] * comps[i][1]
            dNz += W[i] * comps[i][2]
        return val, dNx, dNy, dNz


class CurlNoise3:
    """Classic curl (Bridson 2007): |∇×ψ| for a 3-field vector potential,
    via central differences — 12 noise evaluations per sample."""

    def __init__(self, seed=0, eps=0.01):
        self.eps = eps
        self.P = SimplexNoise3(seed)
        self.Q = SimplexNoise3(seed + 101)
        self.R = SimplexNoise3(seed + 202)

    def __call__(self, x, y, z):
        e = self.eps
        P, Q, R = self.P, self.Q, self.R
        cx = (R(x, y+e, z)-R(x, y-e, z))/(2*e) - (Q(x, y, z+e)-Q(x, y, z-e))/(2*e)
        cy = (P(x, y, z+e)-P(x, y, z-e))/(2*e) - (R(x+e, y, z)-R(x-e, y, z))/(2*e)
        cz = (Q(x+e, y, z)-Q(x-e, y, z))/(2*e) - (P(x, y+e, z)-P(x, y-e, z))/(2*e)
        m = np.sqrt(cx*cx + cy*cy + cz*cz)
        return (1.0 - np.exp(-m * 2.0)) * 2.0 - 1.0


class CurlAnalyticNoise3:
    """Curl with ANALYTIC derivatives: 3 evaluations instead of 12, no
    differencing error, no epsilon."""

    def __init__(self, seed=0):
        self.P = PerlinDeriv3(seed)
        self.Q = PerlinDeriv3(seed + 101)
        self.R = PerlinDeriv3(seed + 202)

    def __call__(self, x, y, z):
        _, dPdx, dPdy, dPdz = self.P(x, y, z)
        _, dQdx, dQdy, dQdz = self.Q(x, y, z)
        _, dRdx, dRdy, dRdz = self.R(x, y, z)
        cx, cy, cz = dRdy - dQdz, dPdz - dRdx, dQdx - dPdy
        m = np.sqrt(cx*cx + cy*cy + cz*cz)
        return (1.0 - np.exp(-m * 2.0)) * 2.0 - 1.0


class FlowNoise3:
    """Flow noise (after Perlin 2001): F = n + k·∂n/∂x — the analytic
    gradient shears features; sweeping z makes them stream."""

    def __init__(self, seed=0, strength=1.0):
        self.k = float(strength)
        self.N = PerlinDeriv3(seed)

    def __call__(self, x, y, z):
        n, gx, _, _ = self.N(x, y, z)
        return np.clip(n + self.k * gx * 1.4, -1.0, 1.0)


# ============================================================================
#  4. ALGORITHMS — KERNEL / SPECTRAL / N-D FAMILY
# ============================================================================

class GaborNoise3:
    """Gabor / sparse-convolution noise (Lagae 2009): one hashed impulse per
    cell — gaussian window × oriented cosine carrier. The BANDWIDTH slider
    designs the spectrum directly (narrow = a ring in the FFT)."""

    def __init__(self, seed=0, bandwidth=4.0, f0=0.7, lam=1.15):
        self.seed, self.bandwidth, self.f0, self.lam = int(seed), float(bandwidth), f0, lam

    def __call__(self, x, y, z):
        xi, yi, zi = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64), np.floor(z).astype(np.int64)
        acc = np.zeros_like(x)
        s = self.seed
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    cx, cy, cz = xi + dx, yi + dy, zi + dz
                    px, py = cx + _hash01(cx, cy, cz, s), cy + _hash01(cx, cy, cz, s + 1)
                    pz = cz + _hash01(cx, cy, cz, s + 2)
                    u4, u5 = _hash01(cx, cy, cz, s + 3), _hash01(cx, cy, cz, s + 4)
                    u6 = _hash01(cx, cy, cz, s + 5)
                    ph = 2.0 * np.pi * _hash01(cx, cy, cz, s + 6)
                    oz = 2.0 * u4 - 1.0
                    r = np.sqrt(np.maximum(1.0 - oz * oz, 0.0))
                    th = 2.0 * np.pi * u5
                    ox, oy = r * np.cos(th), r * np.sin(th)
                    f = self.f0 * (self.bandwidth ** u6)
                    dxo, dyo, dzo = px - x, py - y, pz - z
                    g = np.exp(-np.pi * self.lam * self.lam *
                               (dxo*dxo + dyo*dyo + dzo*dzo))
                    acc += g * np.cos(2.0 * np.pi * f *
                                      (dxo*ox + dyo*oy + dzo*oz) + ph)
        return np.clip(acc * 0.55, -1.0, 1.0)

    def slice_feature_points(self, n, freq, k):
        s = self.seed
        return [((i + _hash01(i, j, k, s)) / freq,
                 (j + _hash01(i, j, k, s + 1)) / freq)
                for i in range(n) for j in range(n)]


class CraterNoise3:
    """Craters — the same sparse convolution as Gabor, kernel = radial
    BOWL + RIM. Overlapping craters erode each other like regolith."""

    def __init__(self, seed=0, size=1.05):
        self.seed, self.size = int(seed), float(size)

    def __call__(self, x, y, z):
        xi, yi, zi = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64), np.floor(z).astype(np.int64)
        acc = np.zeros_like(x)
        s = self.seed
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    cx, cy, cz = xi + dx, yi + dy, zi + dz
                    px, py = cx + _hash01(cx, cy, cz, s), cy + _hash01(cx, cy, cz, s + 1)
                    pz = cz + _hash01(cx, cy, cz, s + 2)
                    r = self.size * (0.55 + 0.9 * _hash01(cx, cy, cz, s + 3))
                    u = np.sqrt((px-x)**2 + (py-y)**2 + (pz-z)**2) / r
                    acc += np.where(u < 1.2,
                                    -((1.0 - u*u) ** 2)
                                    + 0.6 * np.exp(-((u - 0.85) / 0.25) ** 2),
                                    0.0)
        return np.clip(acc * 0.8, -1.0, 1.0)

    slice_feature_points = GaborNoise3.slice_feature_points


class FourierNoise3:
    """Spectral synthesis: N(x) = Σ aᵢ·sin(2π kᵢ·x + φᵢ) over random integer
    wavevectors, aᵢ ∝ 1/|kᵢ|. Exactly periodic (period 1); its FFT IS the
    dot set ±kᵢ. Not compact — every sample touches every wave."""

    internal_fractal = True

    def __init__(self, seed=0, waves=48):
        rng = np.random.default_rng(seed)
        ks = rng.integers(-6, 7, size=(max(int(waves), 8), 3))
        ks = ks[np.linalg.norm(ks, axis=1) > 0]
        self.k = ks.T.astype(np.float64)
        a = 1.0 / np.linalg.norm(ks, axis=1)
        self.a = a / a.sum()
        self.ph = rng.random(len(ks)) * 2.0 * np.pi

    def __call__(self, x, y, z):
        acc = np.zeros_like(x)
        for (kx, ky, kz), a, ph in zip(self.k.T, self.a, self.ph):
            acc += a * np.sin(2.0 * np.pi * (kx*x + ky*y + kz*z) + ph)
        return acc * 1.6


class SpectralNoise3:
    """COLORED noise: random phases × |k|^(−α/2) in the FFT, inverse-
    transformed. α=0 white · 1 pink (1/f) · 2 brown · −1 blue. The spectrum
    you dial is the spectrum you get. Periodic 64³ grid, trilinear lookup."""

    internal_fractal = True

    def __init__(self, seed=0, alpha=1.0, n=64):
        self.n = n = int(n)
        rng = np.random.default_rng(seed)
        k = np.fft.fftfreq(n, d=1.0 / n)
        KX, KY, KZ = np.meshgrid(k, k, k, indexing="ij")
        K2 = KX*KX + KY*KY + KZ*KZ
        K2[0, 0, 0] = 1.0
        F = K2 ** (-float(alpha) / 2.0) * np.exp(2j * np.pi * rng.random((n, n, n)))
        F[0, 0, 0] = 0.0
        f = np.real(np.fft.ifftn(F))
        f = (f - f.mean()) / (f.std() + 1e-12)
        self.grid = f * 0.55

    def __call__(self, x, y, z):
        return _trilinear_periodic(self.grid, x, y, z)


def _trilinear_periodic(grid, X, Y, Z):
    """Periodic trilinear lookup of a cubic grid at float coords."""
    n = grid.shape[0]
    fx, fy, fz = np.floor(X), np.floor(Y), np.floor(Z)
    i0, j0, k0 = fx.astype(np.int64) % n, fy.astype(np.int64) % n, fz.astype(np.int64) % n
    i1, j1, k1 = (i0 + 1) % n, (j0 + 1) % n, (k0 + 1) % n
    tx, ty, tz = X - fx, Y - fy, Z - fz
    g = grid
    c00 = _lerp(g[i0, j0, k0], g[i1, j0, k0], tx)
    c10 = _lerp(g[i0, j1, k0], g[i1, j1, k0], tx)
    c01 = _lerp(g[i0, j0, k1], g[i1, j0, k1], tx)
    c11 = _lerp(g[i0, j1, k1], g[i1, j1, k1], tx)
    return _lerp(_lerp(c00, c10, ty), _lerp(c01, c11, ty), tz)


class GradientNoiseND:
    """Gradient noise in N dimensions: 2ⁿ hypercube corners, hashed to unit
    gradients in Rⁿ. Amplitude is EMPIRICALLY calibrated at construction
    (high-D dot products shrink — measure, then rescale). Missing input axes
    are treated as constant 0, so the app's 3 call sites work at any dim."""

    def __init__(self, seed=0, dim=4):
        self.dim = d = int(dim)
        rng = np.random.default_rng(seed)
        self.perm = _permutation(seed)
        g = rng.normal(size=(256, d))
        g /= np.linalg.norm(g, axis=1, keepdims=True) + 1e-12
        self.G = g
        self.offs = [tuple(int(b) for b in np.binary_repr(c, d)) for c in range(1 << d)]
        pts = rng.random((4096, d)) * 16.0
        s = self._raw(*(pts[:, k] for k in range(d)))
        self.scale = 0.22 / (s.std() + 1e-9)

    def _raw(self, *coords):
        d = self.dim
        f = [np.floor(c) for c in coords]
        iv = [fi.astype(np.int64) for fi in f]
        t = [coords[k] - f[k] for k in range(d)]
        w = [_QUINTIC(t[k]) for k in range(d)]
        acc = np.zeros_like(coords[0])
        for off in self.offs:
            h = (iv[0] + off[0]) & 255
            for k in range(1, d):
                h = self.perm[h] + ((iv[k] + off[k]) & 255)
            g = self.G[self.perm[h]]
            dot = np.zeros_like(acc)
            wgt = np.ones_like(acc)
            for k in range(d):
                dot += g[..., k] * (t[k] - off[k])
                wgt = wgt * (w[k] if off[k] else 1.0 - w[k])
            acc += wgt * dot
        return acc

    def __call__(self, *coords):
        coords = list(coords) + [np.zeros_like(coords[0])] * (self.dim - len(coords))
        return self._raw(*coords) * self.scale


class TorusTileNoise3:
    """Seamless tiling at ANY frequency: each tiled axis is wrapped onto a
    circle in a higher-D space and N-D noise is evaluated there — periodicity
    is exact by construction. Tile 1 axis → 4-D noise, 2 → 5-D, 3 → 6-D."""

    K = 3.0

    def __init__(self, seed=0, tiled_axes=2, period=4.0):
        t = int(tiled_axes)
        self.tiled, self.period = t, float(period)
        self.nd = GradientNoiseND(seed, dim=2 * t + (3 - t))

    def __call__(self, x, y, z):
        P = max(self.period, 1.0)
        R = self.K * P / (2.0 * math.pi)
        coords = []
        for a, c in enumerate((x, y, z)):
            if a < self.tiled:
                th = 2.0 * math.pi * c / P
                coords.append(R * np.cos(th) + R + 40.0 * a)
                coords.append(R * np.sin(th) + R + 40.0 * a + 17.0)
            else:
                coords.append(self.K * c + 90.0 * a)
        return self.nd(*coords)


class DualLatticeNoise3:
    """BCC-style: two cubic Perlin lattices a half-diagonal-cell apart,
    averaged. Each lattice's grid artifacts fall where the other's don't."""

    def __init__(self, seed=0):
        self.a = PerlinNoise3(seed)
        self.b = PerlinNoise3(seed + 77)

    def __call__(self, x, y, z):
        return 0.5 * self.a(x, y, z) + 0.5 * self.b(x + 0.5, y + 0.5, z + 0.5)


# ============================================================================
#  5. ALGORITHMS — RECURSIVE / MULTIRESOLUTION FAMILY
# ============================================================================

class MidpointNoise3:
    """3-D midpoint displacement ('diamond–square, cubed') on a 32³ torus:
    subdivide, average parents, add per-level noise. Randomness baked by
    recursion; sampling is plain trilinear (control grid visible)."""

    internal_fractal = True

    def __init__(self, seed=0, n=32, roughness=0.55):
        rng = np.random.default_rng(seed)
        self.n = int(n)
        g = np.zeros((self.n,) * 3)
        g[0, 0, 0] = rng.normal()
        step, amp = self.n, 1.0
        while step > 1:
            h = step // 2
            g0 = g
            for ax in range(3):                                   # edge midpoints
                e = 0.5 * (np.roll(g0, h, ax) + np.roll(g0, -h, ax))
                m = [slice(None)] * 3
                m[ax] = slice(h, None, step)
                g[tuple(m)] = e[tuple(m)] + amp * (rng.random(e[tuple(m)].shape) * 2 - 1)
            for ax in range(3):                                   # face centres
                a1, a2 = [q for q in range(3) if q != ax]
                f = 0.25 * (np.roll(np.roll(g0, h, a1), h, a2)
                            + np.roll(np.roll(g0, h, a1), -h, a2)
                            + np.roll(np.roll(g0, -h, a1), h, a2)
                            + np.roll(np.roll(g0, -h, a1), -h, a2))
                m = [slice(None, None, step)] * 3
                m[a1] = m[a2] = slice(h, None, step)
                g[tuple(m)] = f[tuple(m)] + amp * (rng.random(f[tuple(m)].shape) * 2 - 1)
            b = np.zeros_like(g)                                  # body centre
            for sx in (-h, h):
                for sy in (-h, h):
                    for sz in (-h, h):
                        b += np.roll(np.roll(np.roll(g0, sx, 0), sy, 1), sz, 2)
            m = (slice(h, None, step),) * 3
            g[m] = b[m] / 8.0 + amp * (rng.random(b[m].shape) * 2 - 1)
            amp *= roughness
            step = h
        g = (g - g.min()) / (g.max() - g.min())
        self.grid = g * 2.0 - 1.0

    def __call__(self, x, y, z):
        return _trilinear_periodic(self.grid, x, y, z)


class WaveletNoise3:
    """Wavelet noise (Cook & DeRose 2005): the field is a sum of frequency
    BANDS. Base volume is band-split with the 1-4-6-4-1 filter
    (B_l = V_l − up(V_l+1)); each band is looked up at its own scale —
    which is what lets renderers attenuate bands to prevent aliasing.
    Runtime = 4 periodic trilinear lookups; construction is precompute."""

    internal_fractal = True

    def __init__(self, seed=0, n=64, levels=4):
        N = self.N = int(n)
        base = ValueNoise3(seed)
        lin = (np.arange(N) + 0.5).astype(np.float64)
        z, y, x = np.meshgrid(lin, lin, lin, indexing="ij")
        vols = [base(x, y, z)]
        for _ in range(int(levels)):
            vols.append(self._down3(vols[-1]))
        self.bands = [vols[l] - self._up3(vols[l + 1])
                      for l in range(len(vols) - 1)] + [vols[-1]]
        rng = np.random.default_rng(seed + 5)
        pts = rng.random((8192, 3)) * N
        s = self._raw(pts[:, 0], pts[:, 1], pts[:, 2])
        self.scale = 0.26 / (s.std() + 1e-9)

    @staticmethod
    def _filt(v, ax, up):
        c = (np.roll(v, 2, ax) + 4.0 * np.roll(v, 1, ax) + 6.0 * v
             + 4.0 * np.roll(v, -1, ax) + np.roll(v, -2, ax))
        return c * (2.0 / 16.0 if up else 1.0 / 16.0)

    @classmethod
    def _down3(cls, v):
        for ax in range(3):
            v = cls._filt(v, ax, False)[tuple(
                slice(0, None, 2) if a == ax else slice(None) for a in range(3))]
        return v

    @classmethod
    def _up3(cls, v):
        for ax in range(3):
            sh = list(v.shape)
            sh[ax] *= 2
            o = np.zeros(sh)
            o[tuple(slice(0, None, 2) if a == ax else slice(None)
                    for a in range(3))] = v
            v = cls._filt(o, ax, True)
        return v

    def _raw(self, x, y, z):
        acc = np.zeros_like(x)
        for l, B in enumerate(self.bands):
            s = B.shape[0] / self.N
            acc += _trilinear_periodic(B, x * s, y * s, z * s)
        return acc

    def __call__(self, x, y, z):
        return self._raw(x, y, z) * self.scale


# ============================================================================
#  6. ALGORITHMS — FRACTAL SYSTEMS + HISTORICAL + EMERGENT
# ============================================================================

def _rotation_matrix(ax, ay, az, theta):
    n = math.sqrt(ax*ax + ay*ay + az*az)
    ax, ay, az = ax / n, ay / n, az / n
    c, s, C = math.cos(theta), math.sin(theta), 1.0 - math.cos(theta)
    return np.array([
        [c + ax*ax*C, ax*ay*C - az*s, ax*az*C + ay*s],
        [ay*ax*C + az*s, c + ay*ay*C, ay*az*C - ax*s],
        [az*ax*C - ay*s, az*ay*C + ax*s, c + az*az*C]])


class RotFbm3:
    """Rotated-octave fBm: the domain is rotated by a fixed matrix before
    each octave, so the octaves' lattice artifacts decorrelate into grain.
    The production fix for fBm's shared-lattice streaks."""

    internal_fractal = True

    def __init__(self, seed=0, octaves=5, gain=0.5, lacunarity=2.0, ftype="fBm"):
        self.base = PerlinNoise3(seed)
        self.oct, self.gain, self.lac = int(octaves), gain, lacunarity
        self.ftype = ftype
        self.R = _rotation_matrix(0.31, 0.55, 0.78, 2.399963)   # golden angle

    def __call__(self, x, y, z):
        total = np.zeros_like(x)
        amp, norm = 1.0, 0.0
        for _ in range(self.oct):
            nv = self.base(x, y, z)
            if self.ftype == "Ridged":
                nv = (1.0 - np.abs(nv)) ** 2 * 2.0 - 1.0
            elif self.ftype == "Billow":
                nv = np.abs(nv) * 2.0 - 1.0
            total += amp * nv
            norm += amp
            amp *= self.gain
            xyz = self.R @ np.stack([x, y, z])
            x, y, z = xyz[0] * self.lac, xyz[1] * self.lac, xyz[2] * self.lac
        return total / norm


class RidgedMultifractal3:
    """Musgrave ridged multifractal: each octave's signal (1−|n|)² also sets
    the NEXT octave's weight — crests amplify detail, valleys starve it."""

    internal_fractal = True

    def __init__(self, seed=0, octaves=5, lacunarity=2.07):
        self.base = PerlinNoise3(seed)
        self.oct, self.lac = int(octaves), lacunarity

    def __call__(self, x, y, z):
        value = np.zeros_like(x)
        weight = np.ones_like(x)
        freq = 1.0
        for _ in range(self.oct):
            n = self.base(x * freq, y * freq, z * freq)
            signal = (1.0 - np.abs(n)) ** 2
            value += signal * weight
            weight = np.clip(signal * 2.0, 0.0, 1.0)
            freq *= self.lac
        return np.clip(value / max(self.oct, 1) * 2.0 - 1.0, -1.0, 1.0)


class HeteroTerrain3:
    """Musgrave heterogeneous terrain: signal = (n+0.35)·weight with
    weight = clamp(signal·1.25) — valleys stay smooth, peaks go rough."""

    internal_fractal = True

    def __init__(self, seed=0, octaves=5, gain=0.5, lacunarity=2.0):
        self.base = PerlinNoise3(seed)
        self.oct, self.gain, self.lac = int(octaves), gain, lacunarity

    def __call__(self, x, y, z):
        value = np.zeros_like(x)
        weight = np.ones_like(x)
        amp, norm, freq = 1.0, 0.0, 1.0
        for _ in range(self.oct):
            n = self.base(x * freq, y * freq, z * freq)
            signal = (n + 0.35) * weight
            value += signal * amp
            norm += amp
            weight = np.clip(signal * 1.25, 0.0, 1.0)
            amp *= self.gain
            freq *= self.lac
        return np.clip(value / norm * 1.1, -1.0, 1.0)


class MarbleNoise3:
    """Peachey's marble (1989): turbulence T = Σ aᵢ·|n| (unnormalized!) fed
    into a sine's phase — the founding formula of procedural texturing."""

    internal_fractal = True

    def __init__(self, seed=0, octaves=6, gain=0.5):
        self.base = PerlinNoise3(seed)
        self.oct, self.gain = int(octaves), float(gain)

    def __call__(self, x, y, z):
        T = np.zeros_like(x)
        amp, freq = 1.0, 1.0
        for _ in range(self.oct):
            T += amp * np.abs(self.base(x * freq, y * freq, z * freq))
            amp *= self.gain
            freq *= 2.0
        return np.sin(2.0 * math.pi * 0.75 * x + 2.8 * T)


class AnisotropicNoise3:
    """Strata: per-axis stretch before an ordinary Perlin call — sediment
    layers, wood grain, wind streaks. The stretch IS the algorithm."""

    def __init__(self, seed=0, sx=1.0, sy=1.0, sz=6.0):
        self.base = PerlinNoise3(seed)
        self.sx, self.sy, self.sz = sx, sy, float(sz)

    def __call__(self, x, y, z):
        return self.base(x * self.sx, y * self.sy, z * self.sz)


class ReactionDiffusionResult:
    """Gray–Scott reaction–diffusion (Turing 1952), grown once on a periodic
    grid then trilinearly sampled. Patterns are GROWN, not computed."""

    internal_fractal = True

    def __init__(self, V):
        v = (V - V.min()) / (V.max() - V.min() + 1e-12)
        self.grid = v * 2.0 - 1.0

    def __call__(self, x, y, z):
        return _trilinear_periodic(self.grid, x, y, z)


def make_rd(seed, feed, kill, iters=140, n=40):
    """Gray–Scott simulation, cached per (seed, feed, kill)."""
    key = (int(seed), round(float(feed), 4), round(float(kill), 4))
    cache = getattr(make_rd, "_cache", None)
    if cache is None:
        cache = make_rd._cache = {}
    if key in cache:
        return cache[key]
    rng = np.random.default_rng(seed)
    U, V = np.ones((n, n, n)), np.zeros((n, n, n))
    for _ in range(14):
        c = rng.integers(6, n - 6, size=3)
        r = int(rng.integers(2, 4))
        zz, yy, xx = np.mgrid[-r:r + 1, -r:r + 1, -r:r + 1]
        m = xx*xx + yy*yy + zz*zz <= r*r
        sl = tuple(slice(int(c[q]) - r, int(c[q]) + r + 1) for q in range(3))
        V[sl][m] = 1.0
        U[sl][m] = 0.0

    def lap(v):
        s = np.zeros_like(v)
        for ax in range(3):
            s += np.roll(v, 1, ax) + np.roll(v, -1, ax)
        return s - 6.0 * v

    Du, Dv = 0.14, 0.07
    for _ in range(int(iters)):
        uvv = U * V * V
        U += Du * lap(U) - uvv + feed * (1.0 - U)
        V += Dv * lap(V) + uvv - (feed + kill) * V
        np.clip(U, 0.0, 1.0, out=U)
        np.clip(V, 0.0, 1.0, out=V)
    inst = ReactionDiffusionResult(V)
    cache[key] = inst
    return inst


# ============================================================================
#  7. FRACTAL LAYERING (fBm / ridged / billow / ping-pong)
# ============================================================================

def fbm(noise, x, y, z, octaves=4, gain=0.5, lacunarity=2.0, ftype="fBm"):
    """Layer octaves: lacunarity^i × frequency, gain^i × amplitude.
    Ridged = (1−|n|)², Billow = |n|, Ping-pong = libnoise triangle-fold
    feedback (banded, marble-like)."""
    if ftype == "Ping-pong":
        total = np.zeros_like(x)
        amp, freq, norm, prev = 1.0, 1.0, 0.0, np.zeros_like(x)
        for _ in range(int(octaves)):
            t = noise(x * freq, y * freq, z * freq) + prev
            tri = 1.0 - np.abs(((t + 1.0) % 2.0) - 1.0)
            total += amp * (tri * 2.0 - 1.0)
            norm += amp
            prev = tri * 2.0 - 1.0
            amp *= gain
            freq *= lacunarity
        return total / max(norm, 1e-9)
    total = np.zeros_like(x)
    amp, freq, norm = 1.0, 1.0, 0.0
    for _ in range(int(octaves)):
        n = noise(x * freq, y * freq, z * freq)
        if ftype == "Ridged":
            n = (1.0 - np.abs(n)) ** 2 * 2.0 - 1.0
        elif ftype == "Billow":
            n = np.abs(n) * 2.0 - 1.0
        total += amp * n
        norm += amp
        amp *= gain
        freq *= lacunarity
    return total / norm


def eval_field(inst, x, y, z, octaves, gain, lac, ftype):
    """Evaluate an algorithm, bypassing the fBm wrapper for internally
    layered algorithms."""
    if getattr(inst, "internal_fractal", False):
        return inst(x, y, z)
    if octaves <= 1 and ftype == "fBm":
        return inst(x, y, z)
    return fbm(inst, x, y, z, octaves, gain, lac, ftype)


# ============================================================================
#  8. FIELD BUILDERS + POST-PROCESSING
# ============================================================================

def _plane_coords(res, freq, t, plane):
    lin = (np.arange(res, dtype=np.float64) + 0.5) / res * freq
    if plane == "XY":
        y, x = np.meshgrid(lin, lin, indexing="ij")
        return x, y, np.full_like(x, float(t))
    if plane == "XZ":
        z, x = np.meshgrid(lin, lin, indexing="ij")
        return x, np.full_like(x, float(t)), z
    z, y = np.meshgrid(lin, lin, indexing="ij")
    return np.full_like(y, float(t)), y, z


def _warp_coords(noise, x, y, z, warp):
    wa = noise(x * 0.5 + 7.31, y * 0.5, z * 0.5) * warp
    wb = noise(x * 0.5, y * 0.5 + 3.17, z * 0.5) * warp
    wc = noise(x * 0.5, y * 0.5, z * 0.5 + 11.9) * warp
    return x + wa, y + wb, z + wc


def terrace(v, steps, sharpness=0.55):
    """Quantize [-1,1] into `steps` plateaus joined by smooth S-curves."""
    t = (v + 1.0) * 0.5 * steps
    k = np.floor(t)
    f = t - k
    g = np.clip((f - 0.5) * (1.0 + sharpness * 6.0) + 0.5, 0.0, 1.0)
    g = g * g * (3.0 - 2.0 * g)
    return ((k + g) / steps) * 2.0 - 1.0


def _finish(v, warp_noise, a, b, c, warp, steps):
    if warp > 0.0 and warp_noise is not None:
        a, b, c = _warp_coords(warp_noise, a, b, c, warp)
    v = v(a, b, c)
    if steps >= 2:
        v = terrace(v, steps)
    return v


def make_slice(noise, warp_noise, res, freq, t, octaves, gain, lac, ftype,
               warp=0.0, steps=0, plane="XY"):
    a, b, c = _plane_coords(res, freq, t, plane)
    return _finish(noise, warp_noise, a, b, c, warp, steps) if (
        warp > 0.0 or steps >= 2) else eval_field(noise, a, b, c,
                                                  octaves, gain, lac, ftype)


def make_field(noise, warp_noise, res, freq, octaves, gain, lac, ftype,
               warp=0.0, steps=0):
    lin = (np.arange(res, dtype=np.float64) + 0.5) / res * freq
    z, y, x = np.meshgrid(lin, lin, lin, indexing="ij")
    if warp > 0.0 or steps >= 2:
        return _finish(noise, warp_noise, x, y, z, warp, steps)
    return eval_field(noise, x, y, z, octaves, gain, lac, ftype)


def isoline_mask(v01, n_levels=8):
    m = np.zeros(v01.shape, dtype=bool)
    for k in range(1, n_levels):
        lv = k / n_levels
        m |= (v01[:-1, :] - lv) * (v01[1:, :] - lv) < 0
        m |= (v01[:, :-1] - lv) * (v01[:, 1:] - lv) < 0
    return m


def shaded_relief_rgb(v, lut, light=(-0.55, -0.62, 0.56), strength=1.0):
    """Lambert + specular shading; normals n ∝ (−∂v/∂x, −∂v/∂y, 1)."""
    gy, gx = np.gradient(v)
    s = 2.2 * strength
    nx, ny, nz = -gx * s, -gy * s, np.ones_like(v)
    m = np.sqrt(nx*nx + ny*ny + nz*nz)
    nx, ny, nz = nx / m, ny / m, nz / m
    L = np.array(light, dtype=np.float64)
    L /= np.linalg.norm(L)
    lam = np.clip(nx*L[0] + ny*L[1] + nz*L[2], 0.0, 1.0)
    hv = (nx + L[0], ny + L[1], nz + L[2])
    hm = np.sqrt(hv[0]**2 + hv[1]**2 + hv[2]**2) + 1e-9
    spec = np.clip((nx*hv[0] + ny*hv[1] + nz*hv[2]) / hm, 0.0, 1.0) ** 24.0
    v01 = np.clip((v + 1.0) * 0.5, 0.0, 1.0)
    rgb = lut[(v01 * 255.0).astype(np.uint8)].astype(np.float32)
    out = rgb * (0.22 + 0.88 * lam)[..., None] \
        + spec[..., None] * np.array([255, 250, 235], np.float32) * 0.55
    return np.clip(out, 0, 255).astype(np.uint8)


# ============================================================================
#  9. COLORMAPS + IMAGE HELPERS
# ============================================================================

def _build_lut(anchors):
    pos = np.array([p for p, _ in anchors], dtype=np.float64)
    cols = np.array([c for _, c in anchors], dtype=np.float64)
    t = np.linspace(0.0, 1.0, 256)
    return np.stack([np.interp(t, pos, cols[:, i]) for i in range(3)],
                    axis=-1).astype(np.uint8)


COLORMAPS = {
    "Viridis":   _build_lut([(0, (68, 1, 84)), (.25, (59, 82, 139)),
                             (.5, (33, 145, 140)), (.75, (94, 201, 98)),
                             (1, (253, 231, 37))]),
    "Magma":     _build_lut([(0, (0, 0, 4)), (.25, (80, 18, 123)),
                             (.5, (182, 54, 121)), (.75, (251, 136, 97)),
                             (1, (252, 253, 191))]),
    "Terrain":   _build_lut([(0, (38, 68, 120)), (.42, (52, 110, 160)),
                             (.5, (201, 185, 137)), (.62, (92, 132, 70)),
                             (.8, (120, 113, 96)), (1, (255, 255, 255))]),
    "Grayscale": _build_lut([(0, (0, 0, 0)), (1, (255, 255, 255))]),
    "Ocean":     _build_lut([(0, (4, 12, 48)), (.3, (8, 52, 110)),
                             (.55, (22, 110, 150)), (.75, (96, 190, 200)),
                             (1, (238, 248, 248))]),
    "Ice":       _build_lut([(0, (10, 26, 50)), (.35, (48, 96, 150)),
                             (.7, (140, 192, 226)), (1, (238, 247, 253))]),
    "Autumn":    _build_lut([(0, (28, 18, 12)), (.3, (116, 48, 20)),
                             (.6, (206, 108, 30)), (.85, (243, 188, 78)),
                             (1, (255, 246, 205))]),
}


def values_to_image(v01, lut):
    idx = np.clip(v01 * 255.0, 0, 255).astype(np.uint8)
    rgb = lut[idx]
    h, w = idx.shape
    return QImage(rgb.tobytes(), w, h, 3 * w,
                  QImage.Format.Format_RGB888).copy()


def rgb_to_image(rgb):
    h, w = rgb.shape[:2]
    return QImage(rgb.tobytes(), w, h, 3 * w,
                  QImage.Format.Format_RGB888).copy()


# ============================================================================
# 10. ALGORITHM REGISTRY + TEACHING TEXTS
# ============================================================================

ALGORITHMS = {}

INFO = {
"White": """<h3>White noise — the baseline</h3>
<p>One independent random number per cell; no correlation. Every other
algorithm is a strategy for <i>adding correlation</i>. Flat spectrum —
open Spectrum &amp; stats to see it. Layering in octaves does NOT smooth it.</p>""",
"Value": """<h3>Value noise</h3>
<p>A random number per lattice corner, blended with the active fade curve.
Cheapest of the family; visible axis-aligned blockiness. Try replacing the
fade with <i>nearest</i> and watch it become White's blocks with softer
edges.</p>""",
"Perlin": """<h3>Perlin gradient noise (improved 2002)</h3>
<p>Each corner stores a random GRADIENT; influence = dot(g, offset); the 8
influences blend through the quintic fade. Zero-crossings on every cell face
→ far less gridiness than value noise. Toggle the corner-gradient arrows in
the lattice overlay — the textbook diagram, live.</p>""",
"Perlin (classic '85)": """<h3>Perlin, the ORIGINAL 1985 formulation</h3>
<p>Cubic fade 3t²−2t³ (C¹ only — creases in the gradient view) and
cube-corner gradients (±1,±1,±1)/√3 (diagonal bias). Compare A/B against
improved Perlin: this picture is why the 2002 paper exists.</p>""",
"Simplex": """<h3>Simplex noise (2001)</h3>
<p>Tetrahedral grid: 4 corners instead of 8, radial kernels (0.6−r²)⁴·(g·d).
Fewer grid directions → fewer artifacts; cost scales (n+1) not 2ⁿ. The 3-D
patent (US 6,867,776) expired in 2022.</p>""",
"Worley": """<h3>Worley / cellular noise (1996)</h3>
<p>No interpolation: one hashed feature point per cell (white dots in the
overlay); F1/F2 = nearest / 2nd-nearest distances; F2−F1 draws Voronoi
walls; Cells colours pixels by nearest-cell id. METRIC is a design tool:
Euclidean/Manhattan/Chebyshev/Minkowski give round/diamond/square cells.</p>""",
"Gabor": """<h3>Gabor / sparse convolution (2009)</h3>
<p>One impulse per cell: gaussian window × oriented cosine. The BANDWIDTH
slider designs the spectrum — narrow shows a RING in the Spectrum tab.
Heavy: 27 cells × 7 hashes per sample; check the status bar vs Perlin.</p>""",
"Craters": """<h3>Craters</h3>
<p>Same sparse convolution; kernel = bowl + rim. Overlapping craters erode
like regolith. No lattice interpolation anywhere — pure kernel sums.</p>""",
"Fourier (spectral)": """<h3>Fourier noise — spectral synthesis</h3>
<p>Σ aᵢ·sin(2π kᵢ·x + φᵢ) over random integer wavevectors, aᵢ ∝ 1/|kᵢ|.
The Spectrum tab shows the dots ±kᵢ; period exactly 1 per axis; not compact
(every pixel evaluates every wave — the Waves slider is cost).</p>""",
"Midpoint displacement": """<h3>Midpoint displacement</h3>
<p>3-D diamond–square on a 32³ torus: subdivide, average, add per-level
noise, ×5 levels. Randomness baked by recursion; plain trilinear sampling —
turn the lattice overlay on and find the control grid.</p>""",
"Wavelet (antialiased)": """<h3>Wavelet noise (Cook &amp; DeRose 2005)</h3>
<p>Built as frequency BANDS (1-4-6-4-1 split); each band sampled at its own
scale — the antialiasing switch of production renderers. Runtime is just 4
lookups; the work moved into precompute. Spectrum tab: band-limited.</p>""",
"Colored noise (α)": """<h3>Colored noise — spectral shaping</h3>
<p>Random phases × |k|^(−α) filter, inverse-FFT. The tilt slider IS the
spectrum: 0 white · 1 pink (1/f — what fBm chases) · 2 brown · −1 blue.
One transform vs eight octave evaluations.</p>""",
"Hyper noise (N-D)": """<h3>Gradient noise in N dimensions</h3>
<p>Same recipe at any n: 2ⁿ hypercube corners. Raise the dimension and read
the status bar — the cost doubles per step. Amplitude is re-calibrated from
4096 measured samples (high-D dots shrink — concentration of measure).</p>""",
"Tiled lattice": """<h3>Tiling by hash-mod</h3>
<p>Lattice hash modulo a power-of-two period P — exact repetition, no
blending. Rule: frequency = period, lacunarity 2. Worley variant: wrap the
HASH, never the point positions.</p>""",
"Torus tile (any freq)": """<h3>Torus tiling — seamless at ANY frequency</h3>
<p>Tiled axes wrap onto circles in 4/5/6-D noise: periodicity is exact by
construction, any period, any frequency. Cost: 16/32/64 corners. Set period
= frequency and export — the seam is mathematically absent.</p>""",
"Dual-lattice (BCC)": """<h3>Dual-lattice (BCC-style)</h3>
<p>Two Perlin lattices a half-diagonal apart, averaged: artifacts cancel.
Compare against Rotated fBm — two medicines, one prescription (decorrelate
the grids).</p>""",
"Curl": """<h3>Curl noise (Bridson 2007), classic</h3>
<p>|∇×ψ| for a 3-field potential via central differences — divergence-free
flow (no sources/sinks). 12 evaluations per sample; compare with the
analytic version.</p>""",
"Curl (analytic)": """<h3>Curl with analytic derivatives</h3>
<p>Same construction; each potential returns its exact gradient in one
evaluation — 3 evals total vs 12. Check the status bar; identical physics.</p>""",
"Flow": """<h3>Flow noise (Perlin 2001, simplified)</h3>
<p>F = n + k·∂n/∂x — the analytic derivative shears features; sweeping z
makes them STREAM instead of boil. One evaluation. Derivatives are
animation fuel, not just normals.</p>""",
"Rotated fBm": """<h3>Rotated-octave fBm</h3>
<p>The domain rotates before each octave — the octaves' grid artifacts
decorrelate into grain. One matrix multiply per octave, essentially free.
See it kill Perlin's streaks in the Compare tab at octaves 7.</p>""",
"Musgrave ridged multi": """<h3>Ridged multifractal (Musgrave)</h3>
<p>Feedback: each octave's ridge strength becomes the next octave's weight —
crests amplify detail, valleys starve it. Crests CHAIN into ranges.</p>""",
"Hetero terrain": """<h3>Heterogeneous terrain (Musgrave)</h3>
<p>weight = clamp(signal·1.25): low areas silence following octaves —
plains stay glassy, mountains crumble. Elevation-dependent roughness.</p>""",
"Marble (Peachey '89)": """<h3>Marble — the 1989 formula</h3>
<p>T = Σ aᵢ·|n| (unnormalized, always positive — deliberate) fed into
sin(x + kT). Phase warping a decade before "domain warping" had a name.</p>""",
"Strata (anisotropic)": """<h3>Strata — anisotropic noise</h3>
<p>Per-axis stretch before Perlin: horizontal layers, wood grain, dunes.
Lattice geometry as an art direction dial.</p>""",
"Reaction–diffusion": """<h3>Reaction–diffusion (Gray–Scott / Turing 1952)</h3>
<p>Not a function — a SIMULATION: two chemicals react and diffuse; the V
concentration is the texture. Feed/kill select morphology (spots, coral,
worms, holes). Regrow ≈ 1 s — real simulation. Compare with Worley: similar
cells, opposite causality (geometry vs emergence).</p>""",
}

for _k, _c in [
    ("White", WhiteNoise3), ("Value", ValueNoise3), ("Perlin", PerlinNoise3),
    ("Perlin (classic '85)", PerlinClassic3), ("Simplex", SimplexNoise3),
    ("Worley", WorleyNoise3), ("Gabor", GaborNoise3), ("Craters", CraterNoise3),
    ("Fourier (spectral)", FourierNoise3), ("Midpoint displacement", MidpointNoise3),
    ("Wavelet (antialiased)", WaveletNoise3), ("Colored noise (α)", SpectralNoise3),
    ("Hyper noise (N-D)", GradientNoiseND), ("Tiled lattice", TiledLatticeNoise3),
    ("Torus tile (any freq)", TorusTileNoise3), ("Dual-lattice (BCC)", DualLatticeNoise3),
    ("Curl", CurlNoise3), ("Curl (analytic)", CurlAnalyticNoise3),
    ("Flow", FlowNoise3), ("Rotated fBm", RotFbm3),
    ("Musgrave ridged multi", RidgedMultifractal3), ("Hetero terrain", HeteroTerrain3),
    ("Marble (Peachey '89)", MarbleNoise3), ("Strata (anisotropic)", AnisotropicNoise3),
    ("Reaction–diffusion", ReactionDiffusionResult),
]:
    ALGORITHMS[_k] = {"cls": _c, "info": INFO[_k]}


# ============================================================================
# 11. THE CURRICULUM — 43 LESSONS
# ============================================================================

def _L(title, state, html):
    return {"t": title, "s": state, "h": html}


LESSONS = [
_L("What noise is, and how this course works", dict(algo="White", freq=8, oct=1),
"""<h3>Randomness, plus rules</h3><p><b>Noise</b> looks random but is
deterministic and smooth enough to build worlds from. White noise (loaded)
is useless alone — neighbouring pixels share nothing. Each lesson adds one
rule. <b>How to work</b>: press 🎯 Load lesson setup, do the <b>Do now</b>
tasks, read the <i>Notice</i>, then break something on purpose.</p>
<p><i>Do now:</i> drag z — pure static, nothing travels. That is what the
next lessons fix.</p>"""),

_L("White noise — the raw material", dict(algo="White", freq=6, oct=1),
"""<h3>White noise</h3><p>Independent numbers per cell — every other
algorithm correlates neighbours.</p><p><b>Do now</b></p><ul>
<li>Frequency 2 vs 32 — cells change size, never character.</li>
<li>Spectrum &amp; stats: flat energy at all frequencies.</li>
<li>Compare A/B: White vs Value, same seed.</li></ul>
<p><i>Notice:</i> the histogram is uniform; Perlin's will be a bell.</p>"""),

_L("Smoothness — the fade-curve ladder",
   dict(algo="Value", freq=4, oct=1, fade="nearest · no blend"),
"""<h3>Interpolation is where smoothness comes from</h3><p>Value noise
blends corner values; the blend curve decides everything: nearest (blocks)
→ linear C⁰ (creased) → smoothstep C¹ → quintic C² (the default).</p>
<p><b>Do now</b></p><ul><li>Step the fade combo up one notch at a time.</li>
<li>View → Gradient |∇|: creases glow, then vanish.</li>
<li>Cell anatomy tab: drag the sample point.</li></ul>
<p><i>Notice:</i> one shared code path — the anatomy view re-weights live.</p>"""),

_L("Perlin — gradients beat values", dict(algo="Perlin", freq=4, oct=1, view="Field"),
"""<h3>Store directions, not numbers</h3><p>Each corner holds a gradient
vector; influence = dot(g, offset) — so the field is forced through ZERO on
every cell face. That constraint kills most gridiness.</p><p><b>Do now</b></p>
<ul><li>The arrows on the canvas are the hashed corner gradients.</li>
<li>View → Iso-lines: contours hug the zero-crossings.</li>
<li>Hover across a corner: the value flips sign along its perpendicular.</li></ul>
<p><i>Notice:</i> value noise blobs; Perlin <i>tilts</i>.</p>"""),

_L("Frequency is zoom; seed is the dice", dict(algo="Perlin", freq=2, oct=1),
"""<h3>The two dials you'll use forever</h3><p><b>Frequency</b> = lattice
periods across the slice — pure zoom. <b>Seed</b> swaps the permutation
table: different world, same statistics.</p><p><b>Do now</b></p><ul>
<li>Frequency 2 → 4 → 8 → 16, watching the lattice overlay shrink.</li>
<li>Random seed ×3 — same flavour, different map.</li>
<li>Frequency 2 + octaves 5 — zoom breaks, texture appears (lesson 8).</li></ul>
<p><i>Notice:</i> octaves add detail <i>on top of</i> the base zoom.</p>"""),

_L("Simplex vs Perlin — the artifact hunt",
   dict(algo="Simplex", freq=5, oct=1, view="Gradient |∇|"),
"""<h3>Why simplex exists</h3><p>The cube grid skews into tetrahedra: 4
corners, fewer axes for artifacts to align with.</p><p><b>Do now</b></p>
<ul><li>Gradient view here vs lesson 4's Perlin — hunt the streaks.</li>
<li>Compare A/B, same seed: |A−B| is bright where they disagree.</li>
<li>Iso-lines: simplex cells look like wonky triangles — the skew, visible.</li></ul>
<p><i>Notice:</i> Perlin's diagonal streaks; simplex's are weaker, rotated.</p>"""),

_L("Worley — distance is the signal", dict(algo="Worley", freq=4, oct=1, wmode="F2-F1"),
"""<h3>No interpolation — just distances</h3><p>One feature point per cell
(white dots on the nearest z-layer). F2−F1 is bright exactly where points
are equidistant: the Voronoi diagram.</p><p><b>Do now</b></p><ul>
<li>Modes F1 → F2 → F2−F1 → Cells.</li>
<li>Animate z — walls pop as the plane crosses 3-D borders.</li>
<li>Octaves 3, gain 0.45 — eroded stone.</li></ul>
<p><i>Notice:</i> the walls are HARD — the function isn't even continuous
there. Distance fields play by different rules.</p>"""),

_L("fBm — octaves build detail", dict(algo="Perlin", freq=3, oct=6, gain=0.5, lac=2.0),
"""<h3>Fractal Brownian Motion</h3><p>Octave i runs at lacunarityⁱ×
frequency and gainⁱ× amplitude — an approximate 1/f spectrum, which is what
&quot;natural&quot; measures as.</p><p><b>Do now</b></p><ul>
<li>Octave lab tab: each octave alone, then the running sum.</li>
<li>Octaves 1 → 6 slowly, here.</li>
<li>Spectrum tab: the low-frequency shoulder grows.</li></ul>
<p><i>Notice:</i> octave images are auto-scaled for display — labels show
their true amplitude halving each step.</p>"""),

_L("Gain & lacunarity — the roughness dials",
   dict(algo="Perlin", freq=3, oct=5, gain=0.7, lac=2.0, cmap="Terrain"),
"""<h3>Two numbers shape the texture</h3><p><b>Gain</b> = octave loudness
(0.3 silky → 0.7 noisy). <b>Lacunarity</b> = frequency step (2 classic; off-
values detune octaves — and break tiling, lesson 14).</p><p><b>Do now</b></p>
<ul><li>Gain 0.3 → 0.5 → 0.7 at octaves 5.</li>
<li>Lacunarity 2.0 vs 2.3.</li>
<li>Octave lab: labels show f×Lⁱ, amp×gⁱ.</li></ul>
<p><i>Notice:</i> gain is the most-visited slider in production terrain.</p>"""),

_L("Ridged & billow — carving mountains",
   dict(algo="Perlin", freq=3, oct=5, gain=0.5, ftype="Ridged", cmap="Terrain"),
"""<h3>Shaping octaves</h3><p>Transform each octave before summing:
<b>Ridged</b> (1−|n|)² sharpens zero-crossings into crests; <b>Billow</b>
|n| puffs upward — clouds, cotton.</p><p><b>Do now</b></p><ul>
<li>Ridged + Terrain map: find the ridge lines.</li>
<li>Switch to Billow — same seed, opposite character.</li>
<li>Octave lab: per-octave images show the shaped versions.</li></ul>
<p><i>Notice:</i> ridges are SHARP though the base is C² — a cheap
nonlinearity buys infinite detail.</p>"""),

_L("Domain warping — marble & flow",
   dict(algo="Simplex", freq=3, oct=4, warp=0.8, cmap="Magma"),
"""<h3>Noise as coordinates</h3><p>Sample at (x+n₁k, y+n₂k, z+n₃k) — three
offset noise copies displace the input itself. Straight iso-bands shear
into marble veins.</p><p><b>Do now</b></p><ul>
<li>Warp 0 → 0.4 → 0.8 → 1.5.</li>
<li>Animate z — veins flow like liquid.</li>
<li>View → Gradient: derivatives go wild (the chain rule at work).</li></ul>
<p><i>Notice:</i> the status bar — warping costs ~4× evaluations.</p>"""),

_L("Terracing — stylised geology",
   dict(algo="Perlin", freq=3, oct=4, steps=6, cmap="Terrain", view="Iso-lines ×8"),
"""<h3>Quantize the field into plateaus</h3><p>Terracing snaps the field to
N levels joined by S-curves — mesas, paddies, stylised cliffs. Applied
<i>after</i> fBm, so edges follow organic contours.</p><p><b>Do now</b></p>
<ul><li>Steps 4 → 8 → 12.</li>
<li>Iso-lines (loaded): every cliff is a bundle of contours.</li>
<li>Steps off/on — compare silhouettes in the 1-D profile.</li></ul>
<p><i>Notice:</i> terracing is a transfer function on the value — it never
touches the noise itself.</p>"""),

_L("Curl — wind without sources",
   dict(algo="Curl (analytic)", freq=4, oct=1),
"""<h3>Divergence-free noise</h3><p>Three noise fields form a vector
potential ψ; |∇×ψ| is displayed. Curl fields have no sources or sinks —
incompressible flow. Production uses the curl VECTOR for particle wind.</p>
<p><b>Do now</b></p><ul><li>Animate z — filaments drift, never collapse.</li>
<li>Add warp 0.5 — shear the vortices.</li>
<li>Switch to classic Curl — same look, 12 evals vs 3 (status bar).</li></ul>
<p><i>Notice:</i> streamlines would circulate forever — that is what
divergence-free buys.</p>"""),

_L("Seamless tiling — textures that wrap",
   dict(algo="Tiled lattice", kind="perlin", period="4", freq=4, oct=1),
"""<h3>Hash modulo the period</h3><p>Hash lattice indices mod P: the field
repeats exactly every P units. The slice spans one period → it tiles.</p>
<p><b>Do now</b></p><ul><li>Export the slice; butt 4 copies together.</li>
<li>Lacunarity 2.3: octave 1 no longer spans a whole period — the seam
returns. The tiling rule, made visible.</li>
<li>Tile kind → worley: the same wrap on feature-point hashing.</li></ul>
<p><i>Notice:</i> nothing was blended or mirrored — exact by construction.</p>"""),

_L("Spectra — why 1/f reads as natural", dict(algo="Perlin", freq=4, oct=1),
"""<h3>Judging noise by its spectrum</h3><p>The Spectrum tab shows the FFT
power image, a radially-averaged log-log curve, and a white-noise
reference.</p><p><b>Do now</b></p><ul>
<li>Compare White vs this in the radial curves: flat vs falling.</li>
<li>Octaves 5: a 1/f shoulder grows on the low end.</li>
<li>Ridged, octaves 5: energy piles into crests — the curve bends.</li></ul>
<p><i>Notice:</i> &quot;natural&quot; is measurable: roughly equal energy
per octave, amplitude ∝ 1/f. fBm synthesises exactly that slope.</p>"""),

_L("Perlin classic vs improved — why 2002 happened",
   dict(algo="Perlin (classic '85)", freq=4, oct=1, view="Gradient |∇|",
        cmp=("Perlin (classic '85)", "Perlin")),
"""<h3>Two lines of code, ten years of look</h3><p>1985: cubic fade +
corner gradients. 2002: quintic fade + edge gradients. Both differences are
visible.</p><p><b>Do now</b></p><ul>
<li>Compare tab (preloaded): |A−B| is bright where formulations disagree.</li>
<li>Gradient |∇| on both: hunt the crease grid on classic.</li></ul>
<p><i>Notice:</i> &quot;improved&quot; changed two lines and fixed a decade
of graphics.</p>"""),

_L("Distance is a design tool — Worley metrics",
   dict(algo="Worley", freq=4, oct=1, wmode="Cells", metric="Manhattan"),
"""<h3>One algorithm, four geometries</h3><p>Nothing changed but
&quot;distance&quot;. Cells mode colours each pixel by its nearest cell —
a Voronoi diagram under your metric.</p><p><b>Do now</b></p><ul>
<li>Cycle Euclidean → Manhattan → Chebyshev → Minkowski 3 (parameters).</li>
<li>F2−F1 per metric — the walls bend with the geometry.</li>
<li>Chebyshev + terracing = blocky mesas.</li></ul>
<p><i>Notice:</i> Manhattan makes diagonals cost 2 — diagonal neighbours
stop being near.</p>"""),

_L("Gabor & craters — noise from kernels",
   dict(algo="Gabor", freq=4, oct=1, gbw=1.6),
"""<h3>Sparse convolution: sum kernels at random points</h3><p>No lattice
interpolation: one impulse per cell = gaussian × oriented cosine.
Bandwidth designs the spectrum.</p><p><b>Do now</b></p><ul>
<li>Bandwidth 1.5 → Spectrum tab: a RING, not a continuum. Raise it — the
ring fills in.</li>
<li>Switch to Craters — same machinery, bowl+rim kernel.</li>
<li>Compare the status bar against Perlin.</li></ul>
<p><i>Notice:</i> kernel noise decouples look from any grid — the price is
evaluation cost.</p>"""),

_L("Fourier noise — synthesis in the frequency domain",
   dict(algo="Fourier (spectral)", freq=4, oct=1, waves=48),
"""<h3>The oldest method: sum plane waves</h3><p>N = Σ aᵢ sin(2π kᵢ·x+φᵢ),
random integer wavevectors, aᵢ ∝ 1/|kᵢ|.</p><p><b>Do now</b></p><ul>
<li>Spectrum tab: the dots ARE the wavevectors (± pairs).</li>
<li>Frequency 2 → export → tiles seamlessly (period exactly 1).</li>
<li>Waves 8 vs 120: detail bought per wave, per pixel.</li></ul>
<p><i>Notice:</i> perfect spectral control, zero locality — the mirror of
spatial methods.</p>"""),

_L("Midpoint displacement — recursion bakes the randomness",
   dict(algo="Midpoint displacement", freq=4, oct=1, lattice=True),
"""<h3>Diamond–square, in 3-D</h3><p>Subdivide, average parents, add scaled
noise, recurse ×5 on a 32³ torus. No hash tables, no fades.</p>
<p><b>Do now</b></p><ul><li>Lattice overlay on: find the 32³ control grid
in the trilinear interpolation.</li>
<li>Compare vs Perlin: similar texture, opposite philosophy.</li>
<li>Animate z — the recursion's layers show through the interpolation.</li></ul>
<p><i>Notice:</i> functional noise is infinite and resolution-free;
generative grids are finite — the classic trade-off.</p>"""),

_L("Musgrave multifractals — terrain that grew",
   dict(algo="Musgrave ridged multi", freq=3, oct=5, cmap="Terrain"),
"""<h3>Feedback between octaves</h3><p>Two one-line upgrades over fBm:
ridged multi lets crests amplify what follows; hetero lets low areas
silence it.</p><p><b>Do now</b></p><ul>
<li>Switch to Hetero terrain: plains go glassy, slopes crumble.</li>
<li>Compare vs lesson 10's naive Ridged fBm — crests now CHAIN.</li>
<li>Octaves 3 vs 8: feedback changes character, not just detail.</li></ul>
<p><i>Notice:</i> elevation-dependent roughness is what reads as geology.</p>"""),

_L("Rotated fBm — the production-quality trick",
   dict(algo="Rotated fBm", freq=3, oct=7, view="Gradient |∇|",
        cmp=("Perlin", "Rotated fBm")),
"""<h3>One matrix multiply per octave</h3><p>Plain fBm stacks every
octave's grid artifacts coherently. Rotating the domain between octaves
decorrelates them.</p><p><b>Do now</b></p><ul>
<li>Compare (preloaded) at octaves 7, gradient view — streaks dissolve
into grain.</li>
<li>Same at octaves 2: nearly identical — the trick pays at depth.</li></ul>
<p><i>Notice:</i> this is why most terrain engines rotate octaves.</p>"""),

_L("Flow noise — derivatives as animation fuel",
   dict(algo="Flow", freq=4, oct=1, flow=1.2),
"""<h3>F = n + k·∂n/∂x</h3><p>The derivative term shears every feature;
sweeping z carries the shear — the field streams like fluid instead of
boiling in place.</p><p><b>Do now</b></p><ul>
<li>▶ Animate z. Strength 0 (boiling) vs 1.2 (streaming) vs 2.5 (smear).</li>
<li>Status bar: ONE evaluation with analytic derivatives.</li></ul>
<p><i>Notice:</i> curl makes a true flow FIELD; flow noise fakes the LOOK
for a fraction of the cost. Choose per budget.</p>"""),

_L("The family tree — capstone", dict(algo="Perlin", freq=4, oct=1),
"""<h3>All of it, on one page</h3>
<p><b>Lattice+interp</b>: White · Value · Perlin '85 · improved · Strata ·
Tiled.<br><b>Simplex</b>: Simplex. <b>Distance</b>: Worley F1/F2/F2−F1/
Cells · metrics.<br><b>Sparse kernels</b>: Gabor · Craters.<br>
<b>Spectral</b>: Fourier · Colored α. <b>Recursive</b>: Midpoint.<br>
<b>Multiresolution</b>: Wavelet. <b>N-D</b>: Hyper · Torus tile.<br>
<b>Dual lattice</b>: BCC-style. <b>Derivative</b>: Curl ×2 · Flow.<br>
<b>Fractal systems</b>: fBm · Ridged/Billow/Ping-pong · Rotated · Musgrave
×2 · Marble.<br><b>Emergent</b>: Reaction–diffusion.</p>
<p><b>Final exam</b>: pick any two in the Compare tab and predict |A−B|
before looking. If you can, this course worked.</p>"""),

_L("Wavelet noise — antialiasing by construction",
   dict(algo="Wavelet (antialiased)", freq=8, oct=1,
        cmp=("Wavelet (antialiased)", "Perlin"), seed=3),
"""<h3>Bands instead of one scale</h3><p>Lattice noise has one
characteristic scale, so sparse sampling aliases into shimmer. Wavelet
noise splits the field into octave bands; each is sampled at its own scale
and can be attenuated.</p><p><b>Do now</b></p><ul>
<li>Compare (preloaded) vs Perlin; animate z at frequency 16 on each —
Perlin's fine detail crawls, wavelet stays composed.</li>
<li>Spectrum tab: energy drops at the top — bands end.</li></ul>
<p><i>Notice:</i> runtime is 4 lookups; the work moved into precompute.
That trade is everywhere in graphics. (First build takes a moment.)</p>"""),

_L("Colored noise — drawing spectra directly",
   dict(algo="Colored noise (α)", freq=4, oct=1, alpha=1.0, seed=7),
"""<h3>The frequency domain, taken literally</h3><p>Every lesson so far
shaped the spectrum indirectly; here you set the exponent: energy ∝ 1/f^α.</p>
<p><b>Do now</b></p><ul><li>α = 0 → Spectrum: flat. α = 1: the 1/f slope.
α = 2: Brownian. α = −1: blue — energy at the TOP.</li>
<li>Compare α=1 vs Perlin octaves 5 — one FFT vs eight evaluations.</li>
<li>Integer frequency → export → tiles (the grid is periodic).</li></ul>
<p><i>Notice:</i> you cannot change α over space — spatial control is what
fBm/multifractals buy back. Blue noise dithers without clumps.</p>"""),

_L("Dimensions — the 2ⁿ tax and what you buy with it",
   dict(algo="Hyper noise (N-D)", freq=4, oct=1, hdim="4"),
"""<h3>Gradient noise generalises to any n</h3><p>4-D = tesseract corners
(16), 5-D = 32, 6-D = 64. Everything else carries over unchanged.</p>
<p><b>Do now</b></p><ul><li>Dimension 4 → 5 → 6 (parameters) while reading
the status bar — the tax doubles each step.</li>
<li>Animated-time noise is 4-D noise with t as the 4th axis — treat the
z-slider as time and you have the game-jam trick.</li></ul>
<p><i>Notice:</i> amplitude is re-calibrated at construction — high-D dots
shrink (concentration of measure). Measure, then rescale. Always.</p>"""),

_L("Torus tiling — seamless at ANY frequency",
   dict(algo="Torus tile (any freq)", freq=3.7, oct=1, torus_axes="2",
        torus_period=3.7, seed=11),
"""<h3>Circles are closed — use that</h3><p>Each tiled axis wraps onto a
circle: θ = 2πx/period; N-D noise there is EXACTLY periodic — no hash-mod,
no power-of-two rule.</p><p><b>Do now</b></p><ul>
<li>Frequency 3.7 = period 3.7 → export → butt tiles: seamless.</li>
<li>Now lacunarity 2.3, octaves 3, export again — find the seam yourself:
octave periods 3.7/2.3ⁱ never return to an integer.</li>
<li>Tile axes → 3: even the z-animation wraps every period.</li></ul>
<p><i>Notice:</i> you paid 5-D cost (32 corners) for freedom. Compare with
lesson 14: free-but-rigid vs costly-but-universal.</p>"""),

_L("Dual lattices — artifacts canceling artifacts",
   dict(algo="Dual-lattice (BCC)", freq=4, oct=7, view="Gradient |∇|",
        cmp=("Perlin", "Dual-lattice (BCC)"), seed=5),
"""<h3>Two grids, one point set</h3><p>Perlin + half-diagonal-offset Perlin
= a body-centered cubic point set. Each grid's streaks fall where the
other's don't.</p><p><b>Do now</b></p><ul>
<li>Compare (preloaded) at octaves 7, gradient view: streaks wash into
grain.</li>
<li>Compare against Rotated fBm (lesson 22): rotation costs one matrix
multiply; interleaving costs one evaluation. Which would you ship?</li></ul>
<p><i>Notice:</i> BCC is the &quot;rounder&quot; cousin of the cube — the
same geometry argument behind grid-free BCC noise research.</p>"""),

_L("Marble & ping-pong — 1989 calls",
   dict(algo="Marble (Peachey '89)", freq=3, oct=6, cmap="Magma", seed=2),
"""<h3>The formula that started the industry</h3><p>Turbulence T = Σ aᵢ·|n|
(unnormalized!) inside sin(x + kT). Every warp and octave you have learned
is in embryo here.</p><p><b>Do now</b></p><ul>
<li>Animate z — the veins are 3-D; the slice just visits them.</li>
<li>Switch to Perlin, octave-mix → <b>Ping-pong</b>: libnoise's cousin —
triangle folds instead of |n| folds.</li>
<li>Compare vs Simplex + warp 0.8 (lesson 11): modern vs 1989 marble.</li></ul>
<p><i>Notice:</i> |n| and ping-pong are the same move — FOLD the signal so
detail can only stack, never cancel.</p>"""),

_L("Reaction–diffusion — patterns that grow themselves",
   dict(algo="Reaction–diffusion", freq=4, oct=1, rd_feed=0.037,
        rd_kill=0.060, cmap="Magma", seed=9),
"""<h3>The first entry that is not a function</h3><p>Turing 1952: two
chemicals, one autocatalytic, diffusing at different rates, spontaneously
pattern. Gray–Scott on a 40³ periodic grid, ~140 steps, then sampled like
Midpoint. The texture was GROWN, then read.</p><p><b>Do now</b></p><ul>
<li>Feed 0.030 → 0.045 (kill 0.060): spots → coral → worms → holes. Each
change regrows (~1 s — real simulation).</li>
<li>Kill 0.065, feed 0.030: bubble regimes.</li>
<li>Compare vs Worley: similar cells, opposite causality.</li></ul>
<p><i>Notice:</i> parameters map to MORPHOLOGY, not amplitudes. This is the
bridge from noise to simulation — erosion, frost, growth all live here.
Capstone: every major way to synthesise structure from mathematics.</p>"""),

_L("Cross-sections — reading noise in 1-D", dict(algo="Perlin", freq=4, oct=1),
"""<h3>The profile is the ground truth</h3><p>Everything you see in colour
is the orange curve sampled across the marked row. Colour is display; the
profile is data.</p><p><b>Do now</b></p><ul>
<li>Drag z slowly — the whole curve morphs: what &quot;3-D noise&quot;
means from the inside.</li>
<li>Octaves 1 → 5: wiggles double, amplitude shrinks — the 1/f recipe as
one line.</li>
<li>Switch colormaps: the image changes, the curve does not.</li></ul>
<p><i>Notice:</i> peak-to-peak stays ~±1 while detail grows — fBm
normalisation at work.</p>"""),

_L("Level sets — iso-lines, isosurfaces, the 3-D tab",
   dict(algo="Perlin", freq=3, oct=1, view="Iso-lines ×8"),
"""<h3>One field, two projections</h3><p>Iso-lines mark where the slice
crosses 8 fixed levels; in 3-D each level becomes an ISOSURFACE — the 3-D
tab renders exactly that.</p><p><b>Do now</b></p><ul>
<li>3-D tab, Isosurface: drag the threshold — you are sweeping those same
levels in 3-D.</li>
<li>Animate z in 2-D: each contour is a cross-section of a deforming blob.</li>
<li>Threshold 0.5 vs 0.2 in 3-D: islands vs continents — how marching cubes
&quot;finds&quot; shapes.</li></ul>
<p><i>Notice:</i> contour density IS the gradient: bunched on slopes.</p>"""),

_L("Light from derivatives — the shaded view",
   dict(algo="Perlin", freq=3, oct=5, view="Shaded (light)", cmap="Terrain"),
"""<h3>∇ is not just for edge detection</h3><p>The <b>Shaded</b> view treats
the field as a heightfield: normal n ∝ (−∂v/∂x, −∂v/∂y, 1), lambert + a
touch of specular from a fixed lamp.</p><p><b>Do now</b></p><ul>
<li>View Field ↔ Shaded: relief appears from nowhere — no new data, just ∇.</li>
<li>Gradient |∇| ↔ Shaded: same math, two uses (analysis vs appearance).</li>
<li>Ridged mix: crests cast shaded flanks — instant mountains.</li></ul>
<p><i>Notice:</i> this is why games bake normal maps — light for free.</p>"""),

_L("Terrain I — noise as elevation",
   dict(algo="Perlin", freq=3, oct=6, cmap="Terrain", tab="terrain"),
"""<h3>The heightmap pipeline</h3><p>The <b>Terrain lab</b> displaces the
slice into 2.5-D relief with a sea plane. Nothing new is computed: value →
elevation, threshold → water.</p><p><b>Do now</b></p><ul>
<li>Sea 0 → 80%: bays → archipelagos → one island. Coastlines are level
sets (lesson 33) in disguise.</li>
<li>Exaggeration ×0.5 vs ×3 — same geology, different drama.</li>
<li>Wireframe on: the mesh every engine would deform.</li></ul>
<p><i>Notice:</i> plain fBm gives generic lumps — real ranges need
structure. Next lesson.</p>"""),

_L("Terrain II — mountains where they belong",
   dict(algo="Musgrave ridged multi", freq=3, oct=7, cmap="Terrain",
        warp=0.5, tab="terrain", sea=42),
"""<h3>Structure, then detail</h3><p>Ridged multifractal chains crests;
domain warp shears them into ranges. The standard recipe: multifractal +
warp + water.</p><p><b>Do now</b></p><ul>
<li>Warp 0 → 0.5 → 1.2: crests → ranges → chaos.</li>
<li>Hetero terrain: valleys glass over.</li>
<li>Terrace steps 6: plateaus clip the ranges (Arizona mode).</li></ul>
<p><i>Notice:</i> in the shaded view the ridges self-light — sharp |1−|n||²
folds make hard flanks from a C² field.</p>"""),

_L("Clouds, fog & volumes",
   dict(algo="Perlin", freq=3, oct=6, gain=0.55, ftype="Billow", cmap="Magma"),
"""<h3>Billow stacks toward +1</h3><p>|n| folds everything positive, so
fBm-of-billow accumulates puffs — DENSITY fields, not heights. Density
wants volume rendering.</p><p><b>Do now</b></p><ul>
<li>3-D tab → Volume render: additive translucent fog.</li>
<li>3-D tab → Isosurface at 0.7: solid puffs.</li>
<li>Back in 2-D, animate z; buy streaming motion with Flow (lesson 23).</li></ul>
<p><i>Notice:</i> same array, three readings — image, isosurface, volume.
Choosing the projection IS art direction.</p>"""),

_L("Wood & grain", dict(algo="Strata (anisotropic)", freq=3, oct=3,
                        stretch=8.0, steps=5, cmap="Autumn"),
"""<h3>Stretch, fold, quantize</h3><p>Wood = rings along one axis +
turbulence in the phase. Recipe: Strata (stretch 8) for grain direction,
terracing (5) for growth rings, ping-pong for figure.</p><p><b>Do now</b></p>
<ul><li>Stretch 2 → 10: bands merge into grain.</li>
<li>Algorithm → Perlin, octave-mix → Ping-pong: banded figure; add terrace
steps for annual rings.</li>
<li>Plane XZ: rings become end-grain. One parameter, two materials.</li></ul>
<p><i>Notice:</i> texture families are TRANSFORMS, not algorithms.</p>"""),

_L("Water, foam & caustics", dict(algo="Worley", freq=5, oct=1,
                                  wmode="F2-F1", cmap="Ocean"),
"""<h3>Distance fields that move read as water</h3><p>F2−F1's cell walls are
exactly where caustic light focuses; animated, the network crawls like
light on a pool floor.</p><p><b>Do now</b></p><ul>
<li>▶ Animate z — walls drift and pop: caustics/foam.</li>
<li>Octaves 2, gain 0.45: foam fractures into bubbles.</li>
<li>Mode F1: the same motion reads as pebbles. Context is everything.</li></ul>
<p><i>Notice:</i> nothing about Worley knows water — palette + motion +
expectation do the work. Noise is a Rorschach test.</p>"""),

_L("Planets & moons — recipe card", dict(algo="Craters", freq=3, oct=2,
                                         cmap="Terrain", tab="terrain", sea=0),
"""<h3>Three worlds from one registry</h3><p><b>Moon</b> (loaded): Craters,
octaves 2, no sea.</p><p><b>Do now</b></p><ul>
<li><b>Ice world</b>: Colored noise α=2, Ice map, terrain, sea 45% —
brownian drift as glaciers.</li>
<li><b>Gas giant</b>: Strata stretch 12, warp 0.6, octaves 4, Magma —
warped latitude bands; add Flow 1.2 for storm shear.</li>
<li><b>Earthlike</b>: lesson 36's recipe + Ocean map, sea ~52%.</li></ul>
<p><i>Notice:</i> every planet is the same algorithms in different clothes
— pick by feature inventory (craters? bands? continents?).</p>"""),

_L("Cost engineering — the budget", dict(algo="Perlin", freq=4, oct=1),
"""<h3>Read the status bar like a producer</h3><p>Ballpark per-sample cost
(2-D slice timings; take ratios, not absolutes): White/Value ≈1× · Perlin ≈2×
· Simplex ≈1.5× · Worley ≈8× (27 cells) · Gabor ≈25× · Curl classic 12 evals
→ analytic 3 · Hyper-6D 64 corners · Wavelet 4 lookups + MBs of precompute
· RD ~1 s once, then free.</p><p><b>Do now</b></p><ul>
<li>Cycle algorithms; watch 2-D ms — build your own table.</li>
<li>Octaves 6: cost = base × octaves × pixels.</li></ul>
<p><i>Notice:</i> quality per millisecond is the real metric.</p>"""),

_L("The artifact clinic — symptoms → cures",
   dict(algo="Perlin", freq=3, oct=7, view="Gradient |∇|",
        cmp=("Perlin", "Rotated fBm")),
"""<h3>Diagnose before you reach for octaves</h3><p>Loaded: fBm's classic
sin — diagonal streaks from shared-lattice octaves.</p><ul>
<li><b>Diagonal streaks</b> → Rotated fBm, Dual-lattice, or Simplex.</li>
<li><b>Crease grid</b> → cubic fade → quintic.</li>
<li><b>Blocks</b> → nearest fade → any blend.</li>
<li><b>Shimmer when animated</b> → single scale → Wavelet bands.</li>
<li><b>Banding</b> → anisotropy — sometimes the feature (wood!).</li></ul>
<p><b>Do now</b></p><ul><li>Compare (preloaded) vs Rotated fBm, octaves 7.</li>
<li>Swap B for Dual-lattice: different cure, same disease.</li></ul>
<p><i>Notice:</i> artifacts are diagnostic data — read them, don't smother
them.</p>"""),

_L("FINAL EXAM — the identification quiz", dict(algo="Perlin", freq=4,
                                                oct=1, quiz=True),
"""<h3>Prove it</h3><p>The <b>Final exam</b> tab renders a mystery field
(pool of 12 signature algorithms) plus its 1-D profile as a clue. Four
buttons, one truth, running score.</p><p><b>Strategy</b></p><ul>
<li>Grid streaks/blocks → lattice family (blobs=Value, tilts=Perlin).</li>
<li>Walls or dots → Worley (stones=F1, walls=F2−F1).</li>
<li>Sine veining → Marble · dot pairs in the spectrum → Fourier ·
soft squares → Midpoint · cradle-density → Gabor/Craters.</li></ul>
<p><b>Do now</b></p><ul><li>Score 8/10 to pass — repeat until you can
verbalise WHY each guess was right. Features, not vibes.</li></ul>
<p><i>Notice:</i> this is forensic graphics — course complete, go make
worlds.</p>"""),
]


# ============================================================================
# 12. WIDGETS
# ============================================================================

class ProfilePlot(QWidget):
    """Minimal 1-D plot of the marked row."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._y = None
        self._label = ""
        self.setMinimumHeight(110)
        self.setMaximumHeight(150)

    def set_curve(self, y, label=""):
        self._y, self._label = y, label
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(18, 18, 22))
        w, h = self.width(), self.height()
        mid = h / 2.0
        p.setPen(QPen(QColor(60, 62, 74))); p.drawLine(0, int(mid), w, int(mid))
        p.setPen(QPen(QColor(45, 47, 58)))
        p.drawLine(0, 6, w, 6); p.drawLine(0, h - 6, w, h - 6)
        if self._y is not None and len(self._y) > 1:
            n = len(self._y)
            amp = (h - 16) / 2.0
            poly = QPolygonF()
            for i in range(n):
                poly.append(QPointF(w * i / (n - 1), mid - float(self._y[i]) * amp))
            p.setPen(QPen(QColor(255, 200, 80), 1.4))
            p.drawPolyline(poly)
        p.setPen(QPen(QColor(150, 152, 165)))
        p.drawText(8, 16, self._label)
        p.drawText(w - 70, 16, "+1")
        p.drawText(w - 70, h - 10, "−1")
        p.end()


class SliceCanvas(QWidget):
    """The teaching canvas: colormap image, lattice lines + intersection
    dots, Perlin corner-gradient arrows, Worley/Gabor feature points,
    analytic sample-gradient arrows, colorbar, hover readout, profile row."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._img = None
        self._freq = 4.0
        self._show_lattice = True
        self._bar = None
        self._corner = []
        self._feat = []
        self._gx = None
        self._field = None
        self._zworld = 0.0
        self._hover = None
        self.setMinimumSize(240, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

    # -- setters -----------------------------------------------------------
    def set_image(self, img, freq, show_lattice):
        self._img, self._freq, self._show_lattice = img, max(freq, 1e-6), show_lattice
        self.update()

    def set_colorbar(self, lut):
        row = lut.reshape(1, 256, 3)
        self._bar = QImage(row.tobytes(), 256, 1, 3 * 256,
                           QImage.Format.Format_RGB888).copy()
        self.update()

    def set_corner_arrows(self, pts):
        self._corner = pts or []
        self.update()

    def set_feature_points(self, pts):
        self._feat = pts or []
        self.update()

    def set_arrows(self, gx, gy):
        if gx is None:
            self._gx = None
        else:
            m = np.hypot(gx, gy)
            self._inv = 1.0 / (np.percentile(m, 98) + 1e-12)
            self._gx, self._gy = gx, gy
        self.update()

    def set_readout(self, field, freq, zworld):
        self._field, self._zworld = field, zworld

    def mouseMoveEvent(self, ev):
        self._hover = ev.position().toPoint()
        self.update()

    def leaveEvent(self, ev):
        self._hover = None
        self.update()

    # -- painting ------------------------------------------------------------
    def _arrow(self, p, ax, ay, bx, by, col, width=1.4):
        p.setPen(QPen(col, width))
        p.drawLine(QPointF(ax, ay), QPointF(bx, by))
        L = math.hypot(bx - ax, by - ay)
        if L < 2:
            return
        ang = math.atan2(by - ay, bx - ax)
        head = max(3.0, L * 0.3)
        for da in (2.55, -2.55):
            p.drawLine(QPointF(bx, by),
                       QPointF(bx + head * math.cos(ang + da),
                               by + head * math.sin(ang + da)))

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        p.fillRect(self.rect(), QColor(18, 18, 22))
        if self._img is None:
            p.end()
            return
        tgt = self._img.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        x0 = (self.width() - tgt.width()) // 2
        y0 = (self.height() - tgt.height()) // 2
        tw, th = tgt.width(), tgt.height()
        p.drawImage(QRect(x0, y0, tw, th), self._img)

        if self._show_lattice:
            stepx, stepy = max(tw / self._freq, 1.0), max(th / self._freq, 1.0)
            p.setPen(QPen(QColor(255, 255, 255, 30)))
            k, xp = 0, float(x0)
            while xp <= x0 + tw + 0.5:
                p.drawLine(QPointF(xp, y0), QPointF(xp, y0 + th))
                k += 1; xp = x0 + k * stepx
            k, yp = 0, float(y0)
            while yp <= y0 + th + 0.5:
                p.drawLine(QPointF(x0, yp), QPointF(x0 + tw, yp))
                k += 1; yp = y0 + k * stepy
            if stepx > 14:                                   # intersection dots
                p.setPen(QPen(QColor(255, 255, 255, 70)))
                k, xp = 0, float(x0)
                while xp <= x0 + tw + 0.5:
                    j, yp = 0, float(y0)
                    while yp <= y0 + th + 0.5:
                        p.drawEllipse(QPointF(xp, yp), 1.6, 1.6)
                        j += 1; yp = y0 + j * stepy
                    k += 1; xp = x0 + k * stepx

        p.setPen(QPen(QColor(70, 70, 82)))
        p.drawRect(x0 - 1, y0 - 1, tw + 2, th + 2)
        rowy = y0 + th // 2
        p.setPen(QPen(QColor(255, 210, 90, 160), 1, Qt.PenStyle.DashLine))
        p.drawLine(x0, rowy, x0 + tw, rowy)

        if self._bar is not None:                            # colorbar
            bx, by, bh = x0 + 4, y0 + th - 16, 10
            p.drawImage(QRect(bx, by, tw - 60, bh), self._bar)
            p.setPen(QPen(QColor(220, 220, 228)))
            p.drawText(QPointF(bx + tw - 56, by + 9), "−1")
            p.drawText(QPointF(bx + tw - 22, by + 9), "+1")

        if self._corner:                                     # hashed gradients
            stepx, stepy = tw / self._freq, th / self._freq
            Lg = min(stepx, stepy) * 0.36
            for ux, uy, vx, vy in self._corner:
                m = math.hypot(vx, vy) or 1.0
                ax, ay = x0 + ux * tw, y0 + uy * th
                self._arrow(p, ax, ay, ax + vx / m * Lg, ay - vy / m * Lg,
                            QColor(120, 200, 255, 210), 1.5)

        if self._feat:                                       # feature points
            p.setPen(QPen(QColor(20, 20, 24)))
            p.setBrush(QColor(255, 255, 255, 215))
            for ux, uy in self._feat:
                p.drawEllipse(QPointF(x0 + ux * tw, y0 + uy * th), 2.6, 2.6)
            p.setBrush(Qt.BrushStyle.NoBrush)

        if self._gx is not None:                             # sample gradients
            n = self._gx.shape[0]
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            st = 16
            for py in range(st // 2, th, st):
                iy = min(int(py / th * n), n - 1)
                for px in range(st // 2, tw, st):
                    ix = min(int(px / tw * n), n - 1)
                    vx = self._gx[iy, ix] * self._inv
                    vy = -self._gy[iy, ix] * self._inv
                    m = math.hypot(vx, vy)
                    if m < 0.05:
                        continue
                    L = 3.0 + st * 0.85 * min(m, 1.0) ** 0.7
                    self._arrow(p, x0 + px, y0 + py,
                                x0 + px + vx / m * L, y0 + py + vy / m * L,
                                QColor(255, 255, 255, 165), 1.2)

        if self._hover is not None and self._field is not None:   # readout
            hx, hy = self._hover.x(), self._hover.y()
            if x0 <= hx < x0 + tw and y0 <= hy < y0 + th:
                n = self._field.shape[0]
                iy = min(int((hy - y0) / th * n), n - 1)
                ix = min(int((hx - x0) / tw * n), n - 1)
                val = float(self._field[iy, ix])
                xw = (ix + 0.5) / n * self._freq
                yw = (iy + 0.5) / n * self._freq
                txt = (f"x {xw:5.2f}   y {yw:5.2f}   z {self._zworld:5.2f}"
                       f"    N {val:+.3f}")
                p.setPen(QPen(QColor(15, 15, 18, 200)))
                p.setBrush(QColor(15, 15, 18, 200))
                p.drawRect(x0 + 6, y0 + 6, 250, 20)
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.setPen(QColor(235, 235, 240))
                f = QFont("Consolas"); f.setStyleHint(QFont.StyleHint.Monospace)
                f.setPointSize(9); p.setFont(f)
                p.drawText(x0 + 12, y0 + 20, txt)
        p.end()


class FadeCurvePlot(QWidget):
    """All four fade curves; the active one highlighted."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._active = DEFAULT_FADE
        self.setMinimumHeight(96)

    def set_active(self, name):
        self._active = name
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(18, 18, 22))
        m, bottom = 10, 16
        w, h = self.width() - 2 * m, self.height() - m - bottom
        p.setPen(QPen(QColor(50, 52, 62)))
        p.drawRect(m, m, w, h)
        t = np.linspace(0.0, 1.0, 96)
        for name, f in FADE_CURVES.items():
            active = name == self._active
            p.setPen(QPen(QColor(255, 184, 77) if active else QColor(120, 126, 142, 90),
                          2.4 if active else 1.2))
            poly = QPolygonF()
            for tv in t:
                poly.append(QPointF(m + tv * w, m + h - float(f(tv)) * h))
            p.drawPolyline(poly)
        p.setPen(QPen(QColor(150, 152, 165)))
        p.drawText(m, self.height() - 4, "0")
        p.drawText(m + w - 34, self.height() - 4, "t = 1")
        p.end()


class WorkerSignals(QObject):
    finished = Signal(object, float)


class FunctionWorker(QRunnable):
    def __init__(self, fn, signals):
        super().__init__()
        self._fn, self._signals = fn, signals

    def run(self):
        try:
            payload, ms = self._fn()
        except Exception:
            import traceback
            payload, ms = traceback.format_exc(), 0.0
        self._signals.finished.emit(payload, ms)


class VolumeView(QWidget):
    """3-D tab: pyqtgraph.opengl isosurface / additive volume, or fallback."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.ok, self.view, self.iso, self.vol = False, None, None, None
        if not _GL_OK:
            lbl = QLabel("3-D view unavailable.\n\nEnable with:\n"
                         "    pip install pyqtgraph PyOpenGL")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setWordWrap(True)
            lay.addWidget(lbl)
            return
        try:
            self.view = _gl.GLViewWidget()
            self.view.setBackgroundColor("#101014")
            self.view.setCameraPosition(distance=2.9, elevation=22, azimuth=-58)
            self.iso = _gl.GLIsosurfaceItem(smooth=True)
            try:
                self.iso.setColor(QColor(105, 175, 255, 255))
            except Exception:
                pass
            self.iso.setVisible(False)
            grid = _gl.GLGridItem()
            try:
                grid.setSize(1.0, 1.0)
                grid.setSpacing(0.125)
                grid.translate(0.5, 0.5, 0.0)
            except Exception:
                pass
            self.view.addItem(grid)
            self.view.addItem(self.iso)
            lay.addWidget(self.view)
            self.ok = True
        except Exception as e:
            lay.addWidget(QLabel(f"3-D view failed to start:\n{e}"))

    def set_data(self, vol01, level, mode, lut):
        if not self.ok:
            return
        n = vol01.shape[0]
        try:
            self.view.setRange(xRange=[0, n], yRange=[0, n], zRange=[0, n],
                               padding=0.05)
        except Exception:
            self.view.setCameraPosition(distance=n * 2.3, elevation=22,
                                        azimuth=-58)
        if mode == "Isosurface":
            self.iso.setData(np.ascontiguousarray(vol01, dtype=np.float32),
                             level=float(level))
            self.iso.setVisible(True)
            if self.vol is not None:
                self.vol.setVisible(False)
        else:
            if self.vol is not None:
                self.view.removeItem(self.vol)
                self.vol = None
            idx = np.clip(vol01 * 255.0, 0, 255).astype(np.uint8)
            rgba = lut[idx].astype(np.float32) / 255.0
            rgba[..., 3] = np.clip((vol01 - 0.32) * 1.1, 0.0, 1.0) * 0.16
            self.vol = _gl.GLVolumeItem(np.ascontiguousarray(rgba),
                                        glOptions="additive")
            self.view.addItem(self.vol)
            self.iso.setVisible(False)


# ---- GPU preview (GLSL) -----------------------------------------------------

VSRC = """
#version 330 core
out vec2 vUv;
void main() {
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    vUv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""

FSRC = """
#version 330 core
in vec2 vUv;
out vec4 fragColor;
uniform float uFreq, uZ, uGain, uLac;
uniform int   uOct, uAlgo;
uniform sampler1D uLut;
float hash13(vec3 p){
    p = fract(p * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return fract((p.x + p.y) * p.z);
}
vec3 hash33(vec3 p){
    p = fract(p * vec3(0.1031, 0.1030, 0.0973));
    p += dot(p, p.yxz + 33.33);
    return fract((p.xxy + p.yxx) * p.zyx);
}
float gdot(vec3 c, vec3 f){ return dot(normalize(hash33(c) * 2.0 - 1.0), f); }
float valueNoise(vec3 p){
    vec3 i = floor(p), f = fract(p);
    vec3 u = f*f*f*(f*(f*6.0 - 15.0) + 10.0);
    return mix(
        mix(mix(hash13(i),               hash13(i + vec3(1,0,0)), u.x),
            mix(hash13(i + vec3(0,1,0)), hash13(i + vec3(1,1,0)), u.x), u.y),
        mix(mix(hash13(i + vec3(0,0,1)),   hash13(i + vec3(1,0,1)), u.x),
            mix(hash13(i + vec3(0,1,1)),   hash13(i + vec3(1,1,1)), u.x), u.y),
        u.z) * 2.0 - 1.0;
}
float perlinNoise(vec3 p){
    vec3 i = floor(p), f = fract(p);
    vec3 u = f*f*f*(f*(f*6.0 - 15.0) + 10.0);
    return 1.8 * mix(
        mix(mix(gdot(i, f),                            gdot(i + vec3(1,0,0), f - vec3(1,0,0)), u.x),
            mix(gdot(i + vec3(0,1,0), f - vec3(0,1,0)), gdot(i + vec3(1,1,0), f - vec3(1,1,0)), u.x), u.y),
        mix(mix(gdot(i + vec3(0,0,1), f - vec3(0,0,1)), gdot(i + vec3(1,0,1), f - vec3(1,0,1)), u.x),
            mix(gdot(i + vec3(0,1,1), f - vec3(0,1,1)), gdot(i + vec3(1,1,1), f - vec3(1,1,1)), u.x), u.y),
        u.z);
}
float field(vec3 p){ return (uAlgo == 0) ? valueNoise(p) : perlinNoise(p); }
float fbm(vec3 p){
    float total = 0.0, amp = 1.0, norm = 0.0;
    for (int o = 0; o < 8; o++) {
        if (o >= uOct) break;
        total += amp * field(p);
        norm  += amp;
        amp   *= uGain;
        p     *= uLac;
    }
    return total / max(norm, 1e-4);
}
void main(){
    float n = fbm(vec3(vUv * uFreq, uZ));
    float t = clamp(n * 0.5 + 0.5, 0.0, 1.0);
    fragColor = vec4(texture(uLut, t).rgb, 1.0);
}
"""


class _GLCanvas(QOpenGLWidget):
    UNIFORMS = ("uFreq", "uZ", "uGain", "uLac", "uOct", "uAlgo", "uLut")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ok, self.err, self.prog = False, "", None
        self.u = {}
        self.params = dict(freq=4.0, z=0.0, oct=4, gain=0.5, lac=2.0, algo=1)
        self._lut = COLORMAPS["Viridis"].copy()
        self.setMinimumSize(200, 200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)

    def initializeGL(self):
        self.ok = False
        try:
            prog = gl.glCreateProgram()
            for kind, src in ((gl.GL_VERTEX_SHADER, VSRC),
                              (gl.GL_FRAGMENT_SHADER, FSRC)):
                sh = gl.glCreateShader(kind)
                gl.glShaderSource(sh, src)
                gl.glCompileShader(sh)
                if not gl.glGetShaderiv(sh, gl.GL_COMPILE_STATUS):
                    raise RuntimeError(gl.glGetShaderInfoLog(sh).decode(
                        errors="ignore")[:600])
                gl.glAttachShader(prog, sh)
            gl.glLinkProgram(prog)
            if not gl.glGetProgramiv(prog, gl.GL_LINK_STATUS):
                raise RuntimeError(gl.glGetProgramInfoLog(prog).decode(
                    errors="ignore")[:600])
            self.prog = prog
            self.u = {n: gl.glGetUniformLocation(prog, n) for n in self.UNIFORMS}
            self.vao = gl.glGenVertexArrays(1)
            gl.glBindVertexArray(self.vao)
            self.tex = gl.glGenTextures(1)
            gl.glBindTexture(gl.GL_TEXTURE_1D, self.tex)
            gl.glTexParameteri(gl.GL_TEXTURE_1D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_1D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_1D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
            self._upload_lut()
            self.ok = True
        except Exception as e:
            self.err = str(e)

    def _upload_lut(self):
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        gl.glTexImage1D(gl.GL_TEXTURE_1D, 0, gl.GL_RGB8, 256, 0, gl.GL_RGB,
                        gl.GL_UNSIGNED_BYTE, self._lut.tobytes())

    def set_colormap(self, lut):
        self._lut = np.ascontiguousarray(lut)
        if self.ok and self.isValid():
            self.makeCurrent()
            gl.glBindTexture(gl.GL_TEXTURE_1D, self.tex)
            self._upload_lut()
            self.update()

    def paintGL(self):
        if not self.ok:
            p = QPainter(self)
            p.fillRect(self.rect(), QColor(18, 18, 22))
            p.setPen(QColor(225, 105, 105))
            p.drawText(self.rect(),
                       Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                       "GLSL preview failed:\n" + self.err)
            p.end()
            return
        gl.glClearColor(0.07, 0.07, 0.09, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glUseProgram(self.prog)
        pr = self.params
        gl.glUniform1f(self.u["uFreq"], float(pr["freq"]))
        gl.glUniform1f(self.u["uZ"], float(pr["z"]))
        gl.glUniform1f(self.u["uGain"], float(pr["gain"]))
        gl.glUniform1f(self.u["uLac"], float(pr["lac"]))
        gl.glUniform1i(self.u["uOct"], int(pr["oct"]))
        gl.glUniform1i(self.u["uAlgo"], int(pr["algo"]))
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_1D, self.tex)
        gl.glUniform1i(self.u["uLut"], 0)
        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)


class GpuNoiseView(QWidget):
    """GLSL preview of the same value/Perlin fBm, at display refresh rate."""

    def __init__(self, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)
        if not _GLW_OK:
            msg = QLabel("GPU preview unavailable.\n\nEnable with:\n"
                         "    pip install PyOpenGL\n(and an OpenGL 3.3 driver).")
            msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
            msg.setWordWrap(True)
            v.addWidget(msg)
            self.ok = False
            return
        self.ok = True
        self.gl = _GLCanvas()
        row = QHBoxLayout()
        self.play_btn = QPushButton("▶  Animate z (GPU)")
        self.algo_combo = QComboBox()
        self.algo_combo.addItems(["value", "perlin"])
        self.fps_lbl = QLabel("– fps")
        row.addWidget(self.play_btn)
        row.addSpacing(10)
        row.addWidget(QLabel("shader:"))
        row.addWidget(self.algo_combo)
        row.addSpacing(10)
        row.addWidget(self.fps_lbl)
        row.addStretch(1)
        hint = QLabel("The same value/Perlin + fBm as the CPU path, compiled to "
                      "one fragment shader — every pixel in parallel. Fields "
                      "differ from the NumPy versions (arithmetic hash vs "
                      "permutation table) but share the statistics.")
        hint.setWordWrap(True)
        v.addWidget(self.gl, 1)
        v.addLayout(row)
        v.addWidget(hint)
        self._playing, self._frames = False, 0
        self._anim = QTimer(self, interval=15, timeout=self._tick)
        self._fps_t = QTimer(self, interval=1000, timeout=self._fps_tick)
        self.play_btn.clicked.connect(self._toggle)
        self.algo_combo.currentIndexChanged.connect(
            lambda i: self.set_params(algo=i))

    def _toggle(self):
        self._playing = not self._playing
        self.play_btn.setText("⏸  Pause" if self._playing else "▶  Animate z (GPU)")
        if self._playing:
            self._anim.start()
            self._fps_t.start()
        else:
            self._anim.stop()
            self._fps_t.stop()

    def _tick(self):
        self.gl.params["z"] += 0.045
        self._frames += 1
        self.gl.update()

    def _fps_tick(self):
        self.fps_lbl.setText(f"{self._frames} fps")
        self._frames = 0

    def set_params(self, **kw):
        if not self.ok:
            return
        self.gl.params.update(kw)
        self.gl.update()

    def set_colormap(self, lut):
        if self.ok:
            self.gl.set_colormap(lut)


# ---- Compare A/B ------------------------------------------------------------

class _CompareCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(300)
        self.imgs, self.labels = [], []
        self.note = ""

    def compute(self, name_a, name_b, seed, freq, octaves, gain, lac, ftype,
                wmode, lut):
        def build(name):
            inst = ALGORITHMS[name]["cls"](seed=seed)
            if isinstance(inst, (WorleyNoise3, TiledLatticeNoise3)):
                inst.mode = wmode
            return inst

        res = 128
        lin = (np.arange(res, dtype=np.float64) + 0.5) / res * freq
        yy, xx = np.meshgrid(lin, lin, indexing="ij")
        zz = np.zeros_like(xx)
        a, b = build(name_a), build(name_b)
        raw = (getattr(a, "internal_fractal", False)
               or getattr(b, "internal_fractal", False))

        def ev(inst):
            return eval_field(inst, xx, yy, zz, octaves, gain, lac, ftype)

        fa, fb = ev(a), ev(b)
        self.imgs, self.labels = [], []
        for inst, name in ((a, name_a), (b, name_b)):
            nv = ev(inst)
            self.imgs.append(values_to_image(
                np.clip((nv + 1.0) * 0.5, 0.0, 1.0), lut))
            self.labels.append(name)
        d = np.abs(fa - fb)
        dmax = float(np.percentile(d, 99.5) + 1e-9)
        self.imgs.append(values_to_image(np.clip(d / dmax, 0, 1),
                                         COLORMAPS["Magma"]))
        self.labels.append(f"|A−B|  (max≈{dmax:.2f})")
        self.note = (f"same seed {seed} · freq {freq:g}"
                     + ("  ·  raw (internal fractal)"
                        if raw else f" · oct {octaves}")
                     + " — bright = disagreement")
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(18, 18, 22))
        if not self.imgs:
            p.setPen(QColor(200, 200, 210))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "pick two algorithms below")
            p.end()
            return
        p.setPen(QPen(QColor(160, 162, 175)))
        p.drawText(10, 18, self.note)
        w, h = self.width(), self.height() - 48
        cell = (w - 30) / 3.0
        for k, img in enumerate(self.imgs):
            side = min(cell - 8, h)
            tx, ty = 10 + k * cell + (cell - side) / 2, 28
            p.drawImage(QRect(int(tx), int(ty), int(side), int(side)), img)
            p.setPen(QPen(QColor(70, 72, 84)))
            p.drawRect(int(tx) - 1, int(ty) - 1, int(side) + 2, int(side) + 2)
            p.setPen(QPen(QColor(120, 140, 200)))
            p.drawText(QRect(int(10 + k * cell), int(ty + side + 4), int(cell), 18),
                       Qt.AlignmentFlag.AlignHCenter, self.labels[k])
            p.setPen(QPen(QColor(160, 162, 175)))
        p.end()


class CompareView(QWidget):
    def __init__(self):
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        self.canvas = _CompareCanvas()
        v.addWidget(self.canvas)
        row = QHBoxLayout()
        self.cmp_a = QComboBox(); self.cmp_a.addItems(list(ALGORITHMS))
        self.cmp_b = QComboBox(); self.cmp_b.addItems(list(ALGORITHMS))
        self.cmp_a.setCurrentText("Perlin"); self.cmp_b.setCurrentText("Simplex")
        row.addWidget(QLabel("A:")); row.addWidget(self.cmp_a)
        row.addSpacing(8)
        row.addWidget(QLabel("B:")); row.addWidget(self.cmp_b)
        row.addStretch(1)
        v.insertLayout(0, row)


# ---- Octave lab ---------------------------------------------------------------

class OctaveLab(QWidget):
    def __init__(self):
        super().__init__()
        self._imgs, self._cums = [], []
        self._note = "Load a lesson or change settings, then open this tab."
        self.setMinimumSize(420, 320)

    def compute(self, noise, freq, z, octaves, gain, lac, ftype, lut):
        res = 112
        lin = (np.arange(res, dtype=np.float64) + 0.5) / res * freq
        yy, xx = np.meshgrid(lin, lin, indexing="ij")
        self._imgs, self._cums = [], []
        total = np.zeros_like(xx)
        norm = 0.0
        for i in range(int(octaves)):
            fi, ai = lac ** i, gain ** i
            n = noise(xx * fi, yy * fi, np.full_like(xx, z * fi))
            span = float(n.max() - n.min()) or 1.0
            self._imgs.append((values_to_image((n - n.min()) / span, lut),
                               f"oct {i} · f×{fi:g} · amp×{ai:g}"))
            shaped = n
            if ftype == "Ridged":
                shaped = (1.0 - np.abs(n)) ** 2 * 2.0 - 1.0
            elif ftype == "Billow":
                shaped = np.abs(n) * 2.0 - 1.0
            elif ftype == "Ping-pong":
                shaped = n
            total += ai * shaped
            norm += ai
            v01 = np.clip((total / norm + 1.0) * 0.5, 0.0, 1.0)
            self._cums.append((values_to_image(v01, lut),
                               f"sum through oct {i}"
                               + ("  (= full fBm)" if i == octaves - 1 else "")))
        self._note = (f"{octaves} octaves · gain {gain:g} · lacunarity {lac:g}"
                      f" · {ftype} · top row auto-scaled, bottom row true scale")
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(18, 18, 22))
        w, h = self.width(), self.height() - 26
        p.setPen(QPen(QColor(160, 162, 175)))
        p.drawText(10, 18, self._note)

        def row(items, y, hh, title):
            if not items:
                return
            p.setPen(QPen(QColor(255, 200, 80)))
            p.drawText(10, y - 6, title)
            cell = (w - 20) / len(items)
            for k, (img, lbl) in enumerate(items):
                side = min(cell - 8, hh)
                tx, ty = 10 + k * cell + (cell - side) / 2, y
                p.drawImage(QRect(int(tx), int(ty), int(side), int(side)), img)
                p.setPen(QPen(QColor(120, 140, 200)))
                p.drawText(QRect(int(10 + k * cell), int(ty + side + 2),
                                 int(cell), 18),
                           Qt.AlignmentFlag.AlignHCenter, lbl)
                p.setPen(QPen(QColor(160, 162, 175)))

        half = (h - 60) / 2.0
        row(self._imgs, 34, half, "individual octaves (each normalised)")
        row(self._cums, 44 + half + 22, half,
            "running sum → fBm (fixed −1…+1 scale)")
        p.setPen(QPen(QColor(120, 122, 135)))
        p.drawText(10, self.height() - 8,
                   "Detail doubles in frequency and halves in amplitude each "
                   "step — that IS the 1/f spectrum.")
        p.end()


# ---- Cell anatomy ---------------------------------------------------------------

class AnatomyView(QWidget):
    def __init__(self):
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        self.canvas = _AnatomyCanvas()
        self.lbl = QLabel("anatomy of one lattice cell — drag the sliders; "
                          "arrows = hashed gradients (Perlin), coloured dots = "
                          "hashed values (Value).")
        self.lbl.setWordWrap(True)
        v.addWidget(self.lbl)
        v.addWidget(self.canvas, 1)
        row = QHBoxLayout()
        self.fx = QSlider(Qt.Orientation.Horizontal)
        self.fx.setRange(2, 98); self.fx.setValue(35)
        self.fy = QSlider(Qt.Orientation.Horizontal)
        self.fy.setRange(2, 98); self.fy.setValue(62)
        for text, s in (("x in cell", self.fx), ("y in cell", self.fy)):
            row.addWidget(QLabel(text)); row.addWidget(s, 1)
        v.addLayout(row)
        self.fx.valueChanged.connect(self._poke)
        self.fy.valueChanged.connect(self._poke)
        self.noise, self.kind = None, None

    def _poke(self, *_):
        self.canvas.set_pos(self.fx.value() / 100.0, self.fy.value() / 100.0)

    def set_lut(self, lut):
        self.canvas.lut = lut
        self.canvas.update()

    def set_source(self, algo_text, seed):
        kind = {"Value": "value", "Perlin": "perlin",
                "Perlin (classic '85)": "classic"}.get(algo_text)
        self.kind = kind
        if kind == "value":
            self.noise = ValueNoise3(seed)
        elif kind == "perlin":
            self.noise = PerlinNoise3(seed)
        elif kind == "classic":
            self.noise = PerlinClassic3(seed)
        else:
            self.noise = None
        self.canvas.set_noise(self.noise, kind)
        self._poke()


class _AnatomyCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(300)
        self.noise, self.kind = None, None
        self.fx = self.fy = 0.5
        self.lut = COLORMAPS["Viridis"]

    def set_noise(self, noise, kind):
        self.noise, self.kind = noise, kind
        self.update()

    def set_pos(self, fx, fy):
        self.fx, self.fy = fx, fy
        self.update()

    def _corner_data(self, i, j):
        if self.kind == "value":
            h = self.noise.perm[self.noise.perm[self.noise.perm[i & 255] + (j & 255)]]
            return self.noise.values[h], None
        h = self.noise.perm[self.noise.perm[self.noise.perm[i & 255] + (j & 255)]]
        if self.kind == "classic":
            b = h & 7
            return None, np.array([(-1.0 if b & 1 else 1.0) * 0.5774,
                                   (-1.0 if b & 2 else 1.0) * 0.5774])
        return None, _GRAD3[h % 12]

    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(18, 18, 22))
        w, h = self.width(), self.height()
        side = min(w - 190, h - 60)
        if side < 120 or self.noise is None:
            p.setPen(QColor(200, 200, 210))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "Cell anatomy supports Value and Perlin — switch "
                       "algorithm, then drag the sliders below.")
            p.end()
            return
        x0, y0 = 16, (h - side) / 2 + 6
        cell = side
        u, v = _fade(self.fx), _fade(self.fy)
        sx, sy = x0 + self.fx * cell, y0 + self.fy * cell

        data = {}
        for c, ci, cj in (("00", 0, 0), ("10", 1, 0), ("01", 0, 1), ("11", 1, 1)):
            val, g = self._corner_data(ci, cj)
            if self.kind == "value":
                data[c] = (x0 + ci * cell, y0 + cj * cell, val, None)
            else:
                dx, dy = self.fx - ci, self.fy - cj
                data[c] = (x0 + ci * cell, y0 + cj * cell, g[0]*dx + g[1]*dy, g)
        contrib = "corner VALUES vᵢ" if self.kind == "value" else "corner DOTS gᵢ·dᵢ"

        wts = {"00": (1-u)*(1-v), "10": u*(1-v), "01": (1-u)*v, "11": u*v}
        result = sum(wts[c] * data[c][2] for c in wts)

        p.setPen(QPen(QColor(70, 72, 84), 1))
        p.setBrush(QColor(26, 26, 32))
        p.drawRect(int(x0), int(y0), int(cell), int(cell))
        p.setBrush(Qt.BrushStyle.NoBrush)

        for c, (px, py, val, g) in data.items():
            col = QColor(120, 180, 255)
            col.setAlphaF(0.12 + 0.75 * wts[c])
            p.setPen(QPen(col, 1 + 2.5 * wts[c]))
            p.drawLine(QPointF(px, py), QPointF(sx, sy))

        for c, (px, py, val, g) in data.items():
            if self.kind == "value":
                idx = int(np.clip((val + 1) * .5, 0, 1) * 255)
                p.setBrush(QColor(int(self.lut[idx][0]), int(self.lut[idx][1]),
                                  int(self.lut[idx][2])))
            else:
                L = 0.30 * cell
                p.setPen(QPen(QColor(255, 170, 70), 2.2))
                p.drawLine(QPointF(px, py),
                           QPointF(px + g[0] * L, py - g[1] * L))
                p.setBrush(QColor(255, 170, 70))
            p.setPen(QPen(QColor(20, 20, 24)))
            p.drawEllipse(QPointF(px, py), 5, 5)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QColor(200, 202, 212)))
            txt = (f"{val:+.2f}" if self.kind == "value" else f"g·d {val:+.2f}")
            p.drawText(QPointF(px + 8, py - 8),
                       f"c{c}: {txt}   w={wts[c]*100:.0f}%")

        idx = int(np.clip((result + 1) * .5, 0, 1) * 255)
        p.setPen(QPen(QColor(255, 255, 255), 1.6))
        p.setBrush(QColor(int(self.lut[idx][0]), int(self.lut[idx][1]),
                          int(self.lut[idx][2])))
        p.drawEllipse(QPointF(sx, sy), 7, 7)
        p.setBrush(Qt.BrushStyle.NoBrush)

        tx = x0 + cell + 18
        p.setPen(QPen(QColor(220, 220, 228)))
        f = QFont("Consolas"); f.setStyleHint(QFont.StyleHint.Monospace)
        f.setPointSize(10); p.setFont(f)
        lines = [f"x in cell  fx = {self.fx:.2f}",
                 f"u = fade(fx) = {u:.3f}",
                 f"y in cell  fy = {self.fy:.2f}",
                 f"v = fade(fy) = {v:.3f}", "", contrib,
                 f"w00={wts['00']:.2f}  w10={wts['10']:.2f}",
                 f"w01={wts['01']:.2f}  w11={wts['11']:.2f}", "",
                 "N = Σ wᵢ·cᵢ", f"  = {result:+.4f}"]
        for k, ln in enumerate(lines):
            p.drawText(QPointF(tx, y0 + 18 + k * 17), ln)

        iw, ih, iy2 = 150, 74, y0 + side - 96
        p.setPen(QPen(QColor(60, 62, 74))); p.drawRect(tx, iy2, iw, ih)
        t = np.linspace(0, 1, 64)
        poly = QPolygonF()
        for tv in t:
            poly.append(QPointF(tx + tv * iw,
                                iy2 + ih - float(_fade(tv)) * ih))
        p.setPen(QPen(QColor(150, 152, 165))); p.drawPolyline(poly)
        for mx, mv, col in ((self.fx, u, QColor(255, 170, 70)),
                            (self.fy, v, QColor(120, 180, 255))):
            p.setPen(QPen(col, 1, Qt.PenStyle.DashLine))
            p.drawLine(QPointF(tx + mx * iw, iy2 + ih),
                       QPointF(tx + mx * iw, iy2 + ih - mv * ih))
            p.setBrush(col)
            p.drawEllipse(QPointF(tx + mx * iw, iy2 + ih - mv * ih), 3, 3)
            p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor(140, 142, 155)))
        p.drawText(QPointF(tx, iy2 + ih + 14),
                   "orange: x-fade   blue: y-fade")
        p.setPen(QPen(QColor(150, 152, 165)))
        p.drawText(QPointF(x0, y0 + side + 18),
                   "z pinned to a lattice plane (its weight is 0) — the third "
                   "axis blends identically.")
        p.end()


# ---- Spectrum & stats --------------------------------------------------------

class SpectrumView(QWidget):
    def __init__(self):
        super().__init__()
        self._fft_img = None
        self._rad_f = self._rad_w = None
        self._hist = None
        self._note = ""
        self.setMinimumSize(420, 300)

    def update_field(self, v):
        v01 = np.clip((v + 1.0) * 0.5, 0.0, 1.0)
        n = v01.shape[0]
        win = np.hanning(n)[:, None] * np.hanning(n)[None, :]

        def power(a):
            F = np.fft.fftshift(np.fft.fft2((a - a.mean()) * win))
            S = np.log1p(np.abs(F))
            return np.clip(S / (np.percentile(S, 99.5) + 1e-9), 0, 1)

        S = power(v01)
        self._fft_img = values_to_image(S, COLORMAPS["Magma"])
        cy = cx = n // 2
        yy, xx = np.mgrid[0:n, 0:n]
        rr = np.hypot(yy - cy, xx - cx).astype(int)

        def radial(img):
            prof = []
            for r in range(1, n // 2):
                m = rr == r
                if m.any():
                    prof.append(img[m].mean())
            return np.array(prof)

        self._rad_f = radial(S)
        rng = np.random.default_rng(0)
        self._rad_w = radial(power(rng.random((n, n))))
        hist, _ = np.histogram(v01, bins=32, range=(0, 1))
        self._hist = hist / (hist.max() + 1e-9)
        self._note = f"slice {n}×{n} · Hann-windowed FFT · DC centred"
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(18, 18, 22))
        p.setPen(QPen(QColor(160, 162, 175)))
        p.drawText(10, 18, self._note)
        w, h = self.width(), self.height()
        side = min(w * 0.42, h - 46)
        ix, iy = 10, 30
        if self._fft_img is not None:
            p.drawImage(QRect(int(ix), int(iy), int(side), int(side)),
                        self._fft_img)
            p.setPen(QPen(QColor(120, 122, 135)))
            p.drawText(int(ix), int(iy + side + 14),
                       "power: centre = DC · edges = highest frequency")
        px, py = ix + side + 24, 30
        pw, ph = max(w - ix - side - 40, 120), int(h * 0.52)
        p.setPen(QPen(QColor(60, 62, 74))); p.drawRect(int(px), int(py), int(pw), int(ph))
        p.setPen(QPen(QColor(120, 122, 135)))
        p.drawText(int(px + 4), int(py + 14), "radial energy  (log-log)")
        if self._rad_f is not None and len(self._rad_f) > 3:
            both = np.concatenate([self._rad_f, self._rad_w])
            lo, hi = np.log10(both.min() + 1e-9), np.log10(both.max() + 1e-9)
            span = (hi - lo) or 1.0
            for curve, col, dash in ((self._rad_w, QColor(130, 132, 145), True),
                                     (self._rad_f, QColor(255, 190, 80), False)):
                pen = QPen(col, 1.8)
                pen.setStyle(Qt.PenStyle.DashLine if dash else Qt.PenStyle.SolidLine)
                p.setPen(pen)
                poly = QPolygonF()
                lx, ly = np.log10(np.arange(len(curve)) + 1), np.log10(curve + 1e-9)
                for k in range(len(curve)):
                    poly.append(QPointF(px + lx[k] / lx[-1] * pw,
                                        py + ph - (ly[k] - lo) / span * ph))
                p.drawPolyline(poly)
            p.setPen(QPen(QColor(150, 152, 165)))
            p.drawText(int(px + 6), int(py + ph - 8),
                       "grey dashed = white noise · orange = this field")
        hx, hy = px, py + ph + 16
        hw, hh = pw, max(h - (py + ph) - 40, 60)
        p.setPen(QPen(QColor(60, 62, 74))); p.drawRect(int(hx), int(hy), int(hw), int(hh))
        p.setPen(QPen(QColor(120, 122, 135)))
        p.drawText(int(hx + 4), int(hy + 14), "value histogram  0…1")
        if self._hist is not None:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(110, 150, 220))
            bw = hw / len(self._hist)
            for k, hv in enumerate(self._hist):
                p.drawRect(int(hx + k * bw), int(hy + hh - hv * (hh - 22)),
                           max(int(bw) - 1, 1), int(hv * (hh - 22)))
            p.setBrush(Qt.BrushStyle.NoBrush)
        p.end()


# ---- Terrain lab ---------------------------------------------------------------

class TerrainCanvas(QWidget):
    """Oblique 2.5-D relief: value = elevation, threshold = water."""

    def __init__(self):
        super().__init__()
        self._field = None
        self._lut = COLORMAPS["Terrain"]
        self.sea, self.exag, self.wire = 0.42, 1.6, False
        self.setMinimumSize(360, 300)

    def set_field(self, field, lut):
        self._field = field
        if lut is not None:
            self._lut = lut
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.fillRect(self.rect(), QColor(14, 14, 18))
        if self._field is None:
            p.setPen(QColor(200, 200, 210))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "load a lesson or change settings — the slice becomes "
                       "terrain here")
            p.end()
            return
        w, h = self.width(), self.height() - 26
        f = self._field
        n = min(f.shape[0], 112)
        g = (f[::max(f.shape[0] // n, 1), ::max(f.shape[0] // n, 1)])[:n, :n]
        h01 = np.clip((g + 1.0) * 0.5, 0.0, 1.0)
        rows = n
        sea = self.sea
        lift = min(0.42 * h * (self.exag / 1.6), h * 0.62)
        cw = w / (rows - 1)
        cd = h / (rows - 1) * 0.72
        top, left = 22, 26

        def pt(i, j, e):
            return QPointF(left + i * cw + (rows - 1 - j) * cw * 0.9 * 0.18,
                           top + j * cd - e * lift)

        water_hi, water_lo = QColor(46, 104, 148), QColor(10, 34, 74)
        for j in range(rows - 1):
            for i in range(rows - 1):
                q = 0.25 * (h01[j, i] + h01[j, i+1] + h01[j+1, i] + h01[j+1, i+1])
                if q < sea:
                    e = sea
                    dpt = (sea - q) / max(sea, 1e-6)
                    col = QColor(int(water_lo.red() + (water_hi.red() - water_lo.red()) * (1 - dpt)),
                                 int(water_lo.green() + (water_hi.green() - water_lo.green()) * (1 - dpt)),
                                 int(water_lo.blue() + (water_hi.blue() - water_lo.blue()) * (1 - dpt)))
                else:
                    e = q
                    c = self._lut[int(q * 255)]
                    shade = 0.72 + 0.28 * ((h01[j, i] + h01[j+1, i+1]) * 0.5)
                    col = QColor(int(c[0]*shade), int(c[1]*shade), int(c[2]*shade))
                poly = QPolygonF([pt(i, j, e), pt(i+1, j, e),
                                  pt(i+1, j+1, e), pt(i, j+1, e)])
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(col)
                p.drawPolygon(poly)
                if self.wire and q >= sea:
                    p.setPen(QPen(QColor(0, 0, 0, 46), 0.6))
                    p.setBrush(Qt.BrushStyle.NoBrush)
                    p.drawPolygon(poly)
        p.setPen(QPen(QColor(150, 152, 165)))
        p.drawText(10, 16, f"2.5-D relief · sea {self.sea:.0%} · "
                           f"exaggeration ×{self.exag:.1f}"
                           + ("  ·  wireframe" if self.wire else ""))
        p.drawText(10, self.height() - 6,
                   "same field as the 2-D tab — colour, displacement and "
                   "light are three readings of one array")
        p.end()


class TerrainLab(QWidget):
    def __init__(self):
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        self.canvas = TerrainCanvas()
        row = QHBoxLayout()
        self.sea_sl = QSlider(Qt.Orientation.Horizontal)
        self.sea_sl.setRange(0, 100); self.sea_sl.setValue(42)
        self.exag_sp = QDoubleSpinBox()
        self.exag_sp.setRange(0.5, 4.0); self.exag_sp.setValue(1.6)
        self.exag_sp.setSingleStep(0.2)
        self.wire_chk = QCheckBox("wireframe")
        row.addWidget(QLabel("sea")); row.addWidget(self.sea_sl, 1)
        row.addWidget(QLabel("exag")); row.addWidget(self.exag_sp)
        row.addWidget(self.wire_chk)
        v.addWidget(self.canvas, 1)
        v.addLayout(row)
        self.sea_sl.valueChanged.connect(self._poke)
        self.exag_sp.valueChanged.connect(self._poke)
        self.wire_chk.toggled.connect(self._poke)

    def _poke(self, *_):
        self.canvas.sea = self.sea_sl.value() / 100.0
        self.canvas.exag = self.exag_sp.value()
        self.canvas.wire = self.wire_chk.isChecked()
        self.canvas.update()

    def refresh(self, field, lut):
        self.canvas.set_field(field, lut)


# ---- Final exam ---------------------------------------------------------------

QRES = 96


class QuizCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self._img = None
        self._prof = None
        self._msg = "press “Next puzzle” to begin"
        self.setMinimumHeight(300)

    def set_puzzle(self, img, prof, msg):
        self._img, self._prof, self._msg = img, prof, msg
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(18, 18, 22))
        w, h = self.width(), self.height()
        side = min(w - 20, h - 90)
        if self._img is not None:
            p.drawImage(QRect(10, 12, side, side), self._img)
            p.setPen(QPen(QColor(70, 72, 84)))
            p.drawRect(9, 11, side + 2, side + 2)
        p.setPen(QPen(QColor(150, 152, 165)))
        p.drawText(12, h - 44, self._msg)
        if self._prof is not None:
            px, py, pw, ph = 10, h - 36, w - 20, 26
            p.setPen(QPen(QColor(50, 52, 62)))
            p.drawRect(px, py, pw, ph)
            y, n = self._prof, len(self._prof)
            poly = QPolygonF()
            for i in range(n):
                poly.append(QPointF(px + pw * i / (n - 1),
                                    py + ph - 3 - float(y[i]) * (ph - 8)))
            p.setPen(QPen(QColor(255, 200, 80), 1.3))
            p.drawPolyline(poly)
            p.setPen(QPen(QColor(120, 122, 135)))
            p.drawText(px + 4, py + 10, "1-D clue")
        p.end()


def _worley_q(mode):
    def mk(seed):
        n = WorleyNoise3(seed)
        n.mode = mode
        return n
    return mk


QUIZ_POOL = [
    ("Value", lambda s: ValueNoise3(s)),
    ("Perlin", lambda s: PerlinNoise3(s)),
    ("Simplex", lambda s: SimplexNoise3(s)),
    ("Worley · F1", _worley_q("F1")),
    ("Worley · F2−F1", _worley_q("F2-F1")),
    ("Gabor", lambda s: GaborNoise3(s)),
    ("Craters", lambda s: CraterNoise3(s)),
    ("Fourier", lambda s: FourierNoise3(s)),
    ("Midpoint", lambda s: MidpointNoise3(s)),
    ("Colored · pink", lambda s: SpectralNoise3(s, 1.0)),
    ("Wavelet", lambda s: WaveletNoise3(s)),
    ("Marble '89", lambda s: MarbleNoise3(s)),
]


class QuizLab(QWidget):
    def __init__(self):
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        self.canvas = QuizCanvas()
        v.addWidget(self.canvas, 1)
        grid = QGridLayout()
        self.buttons = []
        for k in range(4):
            b = QPushButton("?")
            b.clicked.connect(lambda _c=False, i=k: self._answer(i))
            grid.addWidget(b, k // 2, k % 2)
            self.buttons.append(b)
        v.addLayout(grid)
        row = QHBoxLayout()
        self.next_btn = QPushButton("Next puzzle →")
        self.score_lbl = QLabel("score 0 / 0")
        row.addWidget(self.next_btn, 1)
        row.addWidget(self.score_lbl)
        v.addLayout(row)
        self.next_btn.clicked.connect(self.new_puzzle)
        self._score = self._asked = 0
        self._answer_idx = -1

    def _answer(self, i):
        if self._answer_idx < 0:
            return
        self._asked += 1
        ok = i == self._answer_idx
        if ok:
            self._score += 1
        for k, b in enumerate(self.buttons):
            b.setEnabled(False)
            if k == self._answer_idx:
                b.setText("✅ " + b.text())
            elif k == i and not ok:
                b.setText("❌ " + b.text())
        self.score_lbl.setText(f"score {self._score} / {self._asked}")
        self.canvas._msg = (("correct — " if ok else "no — ") + "that was "
                            + self.buttons[self._answer_idx].text()[2:])
        self.canvas.update()

    def new_puzzle(self):
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            names = [n for n, _ in QUIZ_POOL]
            rng = np.random.default_rng()
            correct = int(rng.integers(len(QUIZ_POOL)))
            others = [k for k in range(len(QUIZ_POOL)) if k != correct]
            picks = [correct] + list(rng.choice(others, size=3, replace=False))
            picks = list(rng.permutation(picks))
            seed = int(rng.integers(1, 65536))
            freq = float(rng.integers(3, 7)) + float(rng.integers(0, 2)) * 0.5
            z = float(rng.random() * 2.0)
            noise = QUIZ_POOL[correct][1](seed)
            v = eval_field(noise, *_plane_coords(QRES, freq, z, "XY")[0:1]
                           + _plane_coords(QRES, freq, z, "XY")[1:3],
                           1, 0.5, 2.0, "fBm")
            v01 = np.clip((v + 1.0) * 0.5, 0.0, 1.0)
            img = values_to_image(v01, COLORMAPS["Viridis"])
            prof = np.clip((v[QRES // 2] + 1.0) * 0.5, 0.0, 1.0)
        finally:
            QApplication.restoreOverrideCursor()
        self._answer_idx = picks.index(correct)
        for k, b in enumerate(self.buttons):
            b.setEnabled(True)
            b.setText(names[picks[k]])
        self.canvas.set_puzzle(img, prof,
                               f"mystery field · seed {seed} · freq {freq:g} — "
                               "which algorithm made it?")


# ============================================================================
# 13. MAIN WINDOW
# ============================================================================

HEADER_HTML = """
<h2>Noise Lab — Unified Edition</h2>
<p><b>25 algorithms</b> across 12 families · <b>43 guided lessons</b> ·
9 tabs. Pick a lesson in the navigator (or press 🎯), or explore freely:
the right panel always explains the active algorithm.</p>
<p><b>Course map</b>: 1–15 concepts (lattice, Perlin, simplex, Worley, fBm,
warp, tiling, spectra) · 16–24 frontiers (classic-vs-improved, kernels,
Fourier, midpoint, multifractals, rotation, flow) · 25–31 beyond (wavelet,
colored, N-D, torus, BCC, marble, reaction–diffusion) · 32–43 craft
(relief, terrain, clouds, wood, water, planets, cost, artifacts,
<b>final exam</b>).</p><hr>
"""

INTERNAL_HTML = """<hr><h3>Internally-layered algorithm</h3>
<p>This algorithm builds its own octaves (or none) — Octaves/Gain/Lacunarity
feed its constructor directly and the generic fBm wrapper is bypassed.
Press <b>View source</b> to read the internal loop.</p>"""

FADE_HTML = """<hr><h3>Interpolation quality — active override</h3>
<ul><li><b>nearest</b> — no blend: raw lattice blocks.</li>
<li><b>linear (C⁰)</b> — continuous, creased on cell faces.</li>
<li><b>smoothstep (C¹)</b> — slope continuous; faint traces remain.</li>
<li><b>quintic (C²)</b> — the industry default.</li></ul>
<p>Simplex (radial kernels) and Worley (distances) ignore this control.</p>"""

WARP_HTML = """<hr><h3>Domain warping — active</h3>
<p>Sampling at (x+n₁k, y+n₂k, z+n₃k): the coordinates themselves become
noisy — marble swirls. Cost ≈ 4× evaluations.</p>"""

TERRACE_HTML = """<hr><h3>Terracing — active</h3>
<p>The field is quantized into plateaus joined by S-curves — applied after
fBm, so cliff edges follow organic contours.</p>"""

PINGPONG_HTML = """<hr><h3>Ping-pong octaves — active</h3>
<p>libnoise's triangle-fold feedback: each octave bounces between 0 and 1
and feeds the next — banded, marble-like layering on any base noise.</p>"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Noise Lab — 25 algorithms · 43 lessons · one file")
        self.resize(1460, 900)

        self._noise = None
        self._warp_noise = None
        self._deriv_noise = None
        self._slice_ms = 0.0
        self._vol_ms = 0.0
        self._slice_res_used = 0
        self._last_field = None
        self._last_freq = 4.0
        self._vol01 = None
        self._vol_gen = 0
        self._vol_dirty = True
        self._playing = False
        self._completed = set()
        self._lesson_active = False
        self._lesson_idx = 0
        self._ready = False

        left = self._build_left_panel()
        center = self._build_center()
        right = self._build_info_panel()
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(left)
        split.addWidget(center)
        split.addWidget(right)
        split.setSizes([340, 770, 350])
        self.setCentralWidget(split)

        self._slice_timer = QTimer(self, singleShot=True, timeout=self.regenerate_slice)
        self._vol_timer = QTimer(self, singleShot=True, timeout=self._run_volume_job)
        self._play_timer = QTimer(self, interval=33, timeout=self._play_tick)

        self._connect_signals()
        self._ready = True
        self._rebuild_noise()
        self._reload_lesson_combo(0)
        self._update_z_label()
        self.regenerate_slice()
        self.statusBar().showMessage("pick a lesson, or explore — hover the "
                                     "slice for live coordinates")

    # ---------- UI construction ----------

    def _spin(self, lo, hi, val, step, dec):
        s = QDoubleSpinBox()
        s.setRange(lo, hi); s.setValue(val)
        s.setSingleStep(step); s.setDecimals(dec)
        return s

    def _build_left_panel(self):
        panel = QWidget()
        v = QVBoxLayout(panel); v.setSpacing(10)

        g1 = QGroupBox("Field"); f1 = QFormLayout(g1)
        self.algo_combo = QComboBox(); self.algo_combo.addItems(list(ALGORITHMS))
        f1.addRow("Algorithm", self.algo_combo)
        seed_row = QHBoxLayout()
        self.seed_spin = QSpinBox(); self.seed_spin.setRange(0, 999_999)
        rnd = QPushButton("Random")
        seed_row.addWidget(self.seed_spin, 1); seed_row.addWidget(rnd)
        f1.addRow("Seed", seed_row)
        self.freq_spin = self._spin(0.5, 16.0, 4.0, 0.5, 2)
        self.freq_spin.setToolTip("Lattice periods across the slice — zoom.")
        f1.addRow("Frequency", self.freq_spin)
        self.wmode_combo = QComboBox()
        self.wmode_combo.addItems(["F1", "F2", "F2-F1", "Cells"])
        self.wmode_combo.setEnabled(False)
        f1.addRow("Worley mode", self.wmode_combo)

        g2 = QGroupBox("Fractal layering"); f2 = QFormLayout(g2)
        self.oct_spin = QSpinBox(); self.oct_spin.setRange(1, 8); self.oct_spin.setValue(1)
        f2.addRow("Octaves", self.oct_spin)
        self.gain_spin = self._spin(0.05, 1.0, 0.5, 0.05, 2)
        f2.addRow("Gain (persistence)", self.gain_spin)
        self.lac_spin = self._spin(1.0, 4.0, 2.0, 0.1, 2)
        f2.addRow("Lacunarity", self.lac_spin)
        self.ftype_combo = QComboBox()
        self.ftype_combo.addItems(["fBm", "Ridged", "Billow", "Ping-pong"])
        f2.addRow("Octave mix", self.ftype_combo)

        gF = QGroupBox("Interpolation (fade curve)"); fv = QVBoxLayout(gF)
        self.fade_combo = QComboBox(); self.fade_combo.addItems(list(FADE_CURVES))
        self.fade_combo.setCurrentText(DEFAULT_FADE)
        self.fade_plot = FadeCurvePlot()
        note = QLabel("Applies to Value & Perlin (and Tiled lattice). "
                      "Simplex and Worley do not blend lattice corners.")
        note.setWordWrap(True)
        fv.addWidget(self.fade_combo); fv.addWidget(self.fade_plot)
        fv.addWidget(note)
        self.fade_plot.set_active(DEFAULT_FADE)

        g3 = QGroupBox("Algorithm parameters"); f3 = QFormLayout(g3)
        self.kind_combo = QComboBox(); self.kind_combo.addItems(["perlin", "value", "worley"])
        f3.addRow("Tile kind", self.kind_combo)
        self.period_combo = QComboBox(); self.period_combo.addItems(["2", "4", "8", "16"])
        self.period_combo.setCurrentText("4")
        f3.addRow("Tile period", self.period_combo)
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(["Euclidean", "Manhattan", "Chebyshev", "Minkowski 3"])
        f3.addRow("Worley metric", self.metric_combo)
        self.gbw_spin = self._spin(1.2, 8.0, 4.0, 0.2, 2)
        self.gbw_spin.setToolTip("Gabor bandwidth — spread of impulse frequencies.")
        f3.addRow("Gabor bandwidth", self.gbw_spin)
        self.waves_spin = QDoubleSpinBox()
        self.waves_spin.setRange(8, 128); self.waves_spin.setValue(48)
        self.waves_spin.setDecimals(0); self.waves_spin.setSingleStep(8)
        f3.addRow("Fourier waves", self.waves_spin)
        self.stretch_spin = self._spin(1.0, 12.0, 6.0, 0.5, 1)
        self.stretch_spin.setToolTip("Strata: z-axis stretch.")
        f3.addRow("Strata stretch", self.stretch_spin)
        self.flow_spin = self._spin(0.0, 3.0, 1.0, 0.1, 1)
        f3.addRow("Flow strength", self.flow_spin)
        self.alpha_spin = self._spin(-1.0, 2.5, 1.0, 0.05, 2)
        self.alpha_spin.setToolTip("Colored noise spectral tilt.")
        f3.addRow("Spectral tilt α", self.alpha_spin)
        self.hdim_combo = QComboBox(); self.hdim_combo.addItems(["4", "5", "6"])
        f3.addRow("Dimension (N)", self.hdim_combo)
        self.taxes_combo = QComboBox(); self.taxes_combo.addItems(["1", "2", "3"])
        self.taxes_combo.setCurrentText("2")
        f3.addRow("Torus tile axes", self.taxes_combo)
        self.tperiod_spin = self._spin(2.0, 32.0, 4.0, 0.1, 1)
        f3.addRow("Torus period", self.tperiod_spin)
        self.rd_feed = QDoubleSpinBox()
        self.rd_feed.setRange(0.015, 0.080); self.rd_feed.setValue(0.037)
        self.rd_feed.setDecimals(3); self.rd_feed.setSingleStep(0.001)
        f3.addRow("RD feed", self.rd_feed)
        self.rd_kill = QDoubleSpinBox()
        self.rd_kill.setRange(0.030, 0.075); self.rd_kill.setValue(0.060)
        self.rd_kill.setDecimals(3); self.rd_kill.setSingleStep(0.001)
        f3.addRow("RD kill", self.rd_kill)
        for w in (self.kind_combo, self.period_combo, self.metric_combo,
                  self.gbw_spin, self.waves_spin, self.stretch_spin,
                  self.flow_spin, self.alpha_spin, self.hdim_combo,
                  self.taxes_combo, self.tperiod_spin, self.rd_feed,
                  self.rd_kill):
            w.setEnabled(False)

        gT = QGroupBox("Transforms"); fT = QFormLayout(gT)
        self.warp_spin = self._spin(0.0, 2.0, 0.0, 0.05, 2)
        self.warp_spin.setToolTip("Domain warp strength (in periods).")
        fT.addRow("Domain warp", self.warp_spin)
        self.steps_spin = QSpinBox(); self.steps_spin.setRange(0, 12)
        self.steps_spin.setSpecialValueText("off")
        fT.addRow("Terrace steps", self.steps_spin)

        g4 = QGroupBox("2-D slice (x–y plane)"); f4 = QFormLayout(g4)
        self.slice_res_combo = QComboBox()
        self.slice_res_combo.addItems(["64", "128", "256"])
        self.slice_res_combo.setCurrentIndex(1)
        f4.addRow("Resolution", self.slice_res_combo)
        self.view_combo = QComboBox()
        self.view_combo.addItems(["Field", "Gradient |∇|", "Iso-lines ×8",
                                  "Shaded (light)"])
        f4.addRow("View", self.view_combo)
        self.plane_combo = QComboBox(); self.plane_combo.addItems(["XY", "XZ", "YZ"])
        f4.addRow("Plane", self.plane_combo)
        self.cmap_combo = QComboBox(); self.cmap_combo.addItems(list(COLORMAPS))
        f4.addRow("Colormap", self.cmap_combo)
        self.lattice_chk = QCheckBox("Show lattice cells")
        self.lattice_chk.setChecked(True)
        f4.addRow(self.lattice_chk)
        self.arrows_chk = QCheckBox("Gradient arrows (analytic ∂)")
        self.arrows_chk.setEnabled(False)
        f4.addRow(self.arrows_chk)
        zbox = QVBoxLayout()
        self.z_slider = QSlider(Qt.Orientation.Horizontal)
        self.z_slider.setRange(0, 300)
        self.z_lbl = QLabel("z = 0.00 periods")
        zbox.addWidget(self.z_slider); zbox.addWidget(self.z_lbl)
        self.play_btn = QPushButton("▶  Animate z")
        zbox.addWidget(self.play_btn)
        f4.addRow("Height", zbox)

        g5 = QGroupBox("3-D volume"); f5 = QFormLayout(g5)
        self.vol_res_combo = QComboBox()
        self.vol_res_combo.addItems(["32", "48", "64", "96"])
        self.vol_res_combo.setCurrentIndex(1)
        f5.addRow("Grid", self.vol_res_combo)
        self.render_combo = QComboBox()
        self.render_combo.addItems(["Isosurface", "Volume"])
        f5.addRow("Render", self.render_combo)
        lrow = QHBoxLayout()
        self.level_slider = QSlider(Qt.Orientation.Horizontal)
        self.level_slider.setRange(0, 100); self.level_slider.setValue(60)
        self.level_lbl = QLabel("iso level 0.60")
        lrow.addWidget(self.level_slider, 1); lrow.addWidget(self.level_lbl)
        f5.addRow("Threshold", lrow)

        gE = QGroupBox("Export"); fe = QVBoxLayout(gE)
        self.png_btn = QPushButton("Save 2-D slice as PNG…")
        self.stack_btn = QPushButton("Save volume as PNG stack…")
        fe.addWidget(self.png_btn); fe.addWidget(self.stack_btn)

        src_btn = QPushButton("🔍  View source of this algorithm")

        for g in (g1, g2, gF, g3, gT, g4, g5, gE, src_btn):
            v.addWidget(g if not isinstance(g, QPushButton) else g)
        v.addStretch(1)
        scroll = QScrollArea(); scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(330); scroll.setMaximumWidth(395)
        self._src_btn, self._rnd_btn = src_btn, rnd
        return scroll

    def _build_center(self):
        self.tabs = QTabWidget()
        slice_tab = QWidget()
        sl = QVBoxLayout(slice_tab); sl.setContentsMargins(4, 4, 4, 4)
        self.slice_canvas = SliceCanvas()
        self.profile = ProfilePlot()
        sl.addWidget(self.slice_canvas, 1)
        sl.addWidget(self.profile, 0)
        self.volume_view = VolumeView()
        self.gpu = GpuNoiseView()
        self.compare_view = CompareView()
        self.octave_lab = OctaveLab()
        self.anatomy = AnatomyView()
        self.spec_view = SpectrumView()
        self.terrain_lab = TerrainLab()
        self.quiz_lab = QuizLab()
        for w, name in ((slice_tab, "2-D slice"),
                        (self.volume_view, "3-D volume / isosurface"),
                        (self.gpu, "GPU preview (GLSL)"),
                        (self.compare_view, "Compare A/B"),
                        (self.octave_lab, "Octave lab"),
                        (self.anatomy, "Cell anatomy"),
                        (self.spec_view, "Spectrum & stats"),
                        (self.terrain_lab, "Terrain lab (2.5-D)"),
                        (self.quiz_lab, "Final exam (quiz)")):
            self.tabs.addTab(w, name)
        return self.tabs

    def _build_info_panel(self):
        panel = QWidget()
        v = QVBoxLayout(panel); v.setContentsMargins(0, 0, 0, 0)
        nav = QHBoxLayout()
        self.prev_btn = QPushButton("◀")
        self.next_btn = QPushButton("▶")
        self.lesson_combo = QComboBox()
        nav.addWidget(self.prev_btn); nav.addWidget(self.lesson_combo, 1)
        nav.addWidget(self.next_btn)
        row2 = QHBoxLayout()
        self.load_btn = QPushButton("🎯  Load lesson setup")
        self.prog_lbl = QLabel("")
        row2.addWidget(self.load_btn, 1); row2.addWidget(self.prog_lbl)
        v.addLayout(nav); v.addLayout(row2)
        self.info_browser = QTextBrowser()
        self.info_browser.setMinimumWidth(290)
        v.addWidget(self.info_browser)
        return panel

    def _connect_signals(self):
        for sig in (self.algo_combo.currentTextChanged,
                    self.seed_spin.valueChanged, self.freq_spin.valueChanged,
                    self.oct_spin.valueChanged, self.gain_spin.valueChanged,
                    self.lac_spin.valueChanged,
                    self.ftype_combo.currentIndexChanged,
                    self.wmode_combo.currentIndexChanged,
                    self.kind_combo.currentTextChanged,
                    self.period_combo.currentTextChanged,
                    self.metric_combo.currentIndexChanged,
                    self.gbw_spin.valueChanged, self.waves_spin.valueChanged,
                    self.stretch_spin.valueChanged, self.flow_spin.valueChanged,
                    self.alpha_spin.valueChanged,
                    self.hdim_combo.currentIndexChanged,
                    self.taxes_combo.currentIndexChanged,
                    self.tperiod_spin.valueChanged, self.rd_feed.valueChanged,
                    self.rd_kill.valueChanged, self.warp_spin.valueChanged,
                    self.steps_spin.valueChanged):
            sig.connect(self._on_field_changed)
        self.slice_res_combo.currentTextChanged.connect(
            lambda _t: self._slice_timer.start(60))
        self.view_combo.currentIndexChanged.connect(lambda _i: self._render_slice())
        self.plane_combo.currentIndexChanged.connect(
            lambda _i: self._slice_timer.start(40))
        self.cmap_combo.currentTextChanged.connect(self._on_cmap_changed)
        self.lattice_chk.toggled.connect(lambda _c: self._slice_timer.start(30))
        self.arrows_chk.toggled.connect(lambda _c: self._slice_timer.start(30))
        self.z_slider.valueChanged.connect(lambda _v: self._slice_timer.start(30))
        self.play_btn.clicked.connect(self._toggle_play)
        self.vol_res_combo.currentTextChanged.connect(self._on_volume_param_changed)
        self.render_combo.currentIndexChanged.connect(self._update_3d_items)
        self.level_slider.valueChanged.connect(self._update_3d_items)
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self._src_btn.clicked.connect(self._show_source)
        self._rnd_btn.clicked.connect(self._random_seed)
        self.fade_combo.currentTextChanged.connect(self._on_fade_changed)
        self.png_btn.clicked.connect(self._export_slice)
        self.stack_btn.clicked.connect(self._export_stack)
        self.lesson_combo.currentIndexChanged.connect(self._on_lesson_selected)
        self.prev_btn.clicked.connect(lambda: self._step_lesson(-1))
        self.next_btn.clicked.connect(lambda: self._step_lesson(+1))
        self.load_btn.clicked.connect(
            lambda: self._show_lesson(self._lesson_idx, load=True)
            if self._lesson_active else None)
        self.compare_view.cmp_a.currentIndexChanged.connect(
            lambda _i: self._refresh_compare())
        self.compare_view.cmp_b.currentIndexChanged.connect(
            lambda _i: self._refresh_compare())
        self.gpu.play_btn  # (own wiring)
        self._vol_timer  # noqa

    # ---------- lesson machinery ----------

    def _reload_lesson_combo(self, keep):
        self.lesson_combo.blockSignals(True)
        self.lesson_combo.clear()
        self.lesson_combo.addItem("🧭  Free exploration")
        for i, L in enumerate(LESSONS):
            self.lesson_combo.addItem(
                ("✅ " if i in self._completed else "⬚  ") + f"{i+1:02d} · {L['t']}")
        self.lesson_combo.setCurrentIndex(keep + 1)
        self.lesson_combo.blockSignals(False)
        self.prog_lbl.setText(f"{len(self._completed)}/{len(LESSONS)} done")

    def _on_lesson_selected(self, idx):
        if idx <= 0:
            self._lesson_active = False
            self._update_info()
            return
        self._lesson_active = True
        self._lesson_idx = idx - 1
        self._update_info()

    def _step_lesson(self, d):
        nxt = max(0, min(self.lesson_combo.currentIndex() + d,
                         self.lesson_combo.count() - 1))
        self.lesson_combo.setCurrentIndex(nxt)

    def _show_lesson(self, i, load):
        self._lesson_active = True
        self._lesson_idx = i
        self.lesson_combo.blockSignals(True)
        self.lesson_combo.setCurrentIndex(i + 1)
        self.lesson_combo.blockSignals(False)
        self._update_info()
        if load:
            self._completed.add(i)
            self._reload_lesson_combo(i)
            self._apply_state(LESSONS[i]["s"])
            self.statusBar().showMessage(
                f"lesson {i+1} loaded — follow the “Do now” tasks")

    def _apply_state(self, st):
        if self._playing:
            self._toggle_play()

        def C(c, t):
            c.blockSignals(True)
            if t in [c.itemText(i) for i in range(c.count())]:
                c.setCurrentText(t)
            c.blockSignals(False)

        def S(s, val):
            s.blockSignals(True); s.setValue(val); s.blockSignals(False)

        C(self.algo_combo, st.get("algo", "Perlin"))
        S(self.seed_spin, st.get("seed", 0))
        S(self.freq_spin, st.get("freq", 4.0))
        S(self.oct_spin, st.get("oct", 1))
        S(self.gain_spin, st.get("gain", 0.5))
        S(self.lac_spin, st.get("lac", 2.0))
        C(self.ftype_combo, st.get("ftype", "fBm"))
        C(self.wmode_combo, st.get("wmode", "F1"))
        C(self.metric_combo, st.get("metric", "Euclidean"))
        S(self.gbw_spin, st.get("gbw", 4.0))
        S(self.waves_spin, float(st.get("waves", 48)))
        S(self.stretch_spin, st.get("stretch", 6.0))
        S(self.flow_spin, st.get("flow", 1.0))
        S(self.alpha_spin, st.get("alpha", 1.0))
        C(self.hdim_combo, str(st.get("hdim", "4")))
        C(self.kind_combo, st.get("kind", "perlin"))
        C(self.period_combo, str(st.get("period", "4")))
        C(self.taxes_combo, str(st.get("torus_axes", "2")))
        S(self.tperiod_spin, float(st.get("torus_period", 4.0)))
        S(self.rd_feed, float(st.get("rd_feed", 0.037)))
        S(self.rd_kill, float(st.get("rd_kill", 0.060)))
        S(self.warp_spin, st.get("warp", 0.0))
        S(self.steps_spin, st.get("steps", 0))
        C(self.fade_combo, st.get("fade", DEFAULT_FADE))
        self.fade_plot.set_active(self.fade_combo.currentText())
        C(self.view_combo, st.get("view", "Field"))
        C(self.plane_combo, st.get("plane", "XY"))
        C(self.cmap_combo, st.get("cmap", "Viridis"))
        self.lattice_chk.setChecked(st.get("lattice", True))
        self.arrows_chk.setChecked(st.get("arrows", False))
        self.z_slider.blockSignals(True)
        self.z_slider.setValue(int(round(st.get("z", 0.0) * 100)))
        self.z_slider.blockSignals(False)

        self._rebuild_noise()
        self._update_z_label()
        self.regenerate_slice()
        self._schedule_volume()
        self._sync_gpu()
        if st.get("tab") == "terrain":
            self.tabs.setCurrentWidget(self.terrain_lab)
            if st.get("sea") is not None:
                self.terrain_lab.sea_sl.blockSignals(True)
                self.terrain_lab.sea_sl.setValue(int(st["sea"] * 100))
                self.terrain_lab.sea_sl.blockSignals(False)
                self.terrain_lab._poke()
            self.terrain_lab.refresh(self._last_field,
                                     COLORMAPS[self.cmap_combo.currentText()])
        if st.get("quiz"):
            self.tabs.setCurrentWidget(self.quiz_lab)
        if "cmp" in st:
            C(self.compare_view.cmp_a, st["cmp"][0])
            C(self.compare_view.cmp_b, st["cmp"][1])
            self._refresh_compare()

    # ---------- behaviour ----------

    def _rebuild_noise(self):
        name = self.algo_combo.currentText()
        seed = self.seed_spin.value()
        oct_ = self.oct_spin.value()
        gain, lac = self.gain_spin.value(), self.lac_spin.value()
        ftype = self.ftype_combo.currentText()
        makers = {
            "Tiled lattice": lambda: TiledLatticeNoise3(
                seed, self.kind_combo.currentText(),
                int(self.period_combo.currentText())),
            "Worley": lambda: WorleyNoise3(
                seed, self.metric_combo.currentText(),
                self.wmode_combo.currentText()),
            "Gabor": lambda: GaborNoise3(seed, self.gbw_spin.value()),
            "Fourier (spectral)": lambda: FourierNoise3(seed, self.waves_spin.value()),
            "Strata (anisotropic)": lambda: AnisotropicNoise3(
                seed, sz=self.stretch_spin.value()),
            "Flow": lambda: FlowNoise3(seed, self.flow_spin.value()),
            "Colored noise (α)": lambda: SpectralNoise3(seed, self.alpha_spin.value()),
            "Hyper noise (N-D)": lambda: GradientNoiseND(
                seed, self.hdim_combo.currentText()),
            "Torus tile (any freq)": lambda: TorusTileNoise3(
                seed, self.taxes_combo.currentText(), self.tperiod_spin.value()),
            "Rotated fBm": lambda: RotFbm3(seed, oct_, gain, lac, ftype),
            "Musgrave ridged multi": lambda: RidgedMultifractal3(seed, oct_, lac),
            "Hetero terrain": lambda: HeteroTerrain3(seed, oct_, gain, lac),
            "Marble (Peachey '89)": lambda: MarbleNoise3(seed, oct_, gain),
            "Reaction–diffusion": lambda: make_rd(seed, self.rd_feed.value(),
                                                  self.rd_kill.value()),
        }
        self._noise = makers[name]() if name in makers \
            else ALGORITHMS[name]["cls"](seed=seed)
        self._warp_noise = None
        try:
            self._warp_noise = type(self._noise)(seed=seed + 1013)
            if isinstance(self._noise, (WorleyNoise3, TiledLatticeNoise3)):
                self._warp_noise.mode = self._noise.mode
                if isinstance(self._noise, WorleyNoise3):
                    self._warp_noise.metric = self._noise.metric
                else:
                    self._warp_noise.kind = self._noise.kind
        except Exception:
            self._warp_noise = PerlinNoise3(seed + 1013)
        self._deriv_noise = PerlinDeriv3(seed) if name == "Perlin" else None
        self.arrows_chk.setEnabled(self._deriv_noise is not None)

        is_w = isinstance(self._noise, (WorleyNoise3, TiledLatticeNoise3))
        self.wmode_combo.setEnabled(is_w)
        self.metric_combo.setEnabled(name == "Worley")
        self.kind_combo.setEnabled(name == "Tiled lattice")
        self.period_combo.setEnabled(name == "Tiled lattice")
        self.gbw_spin.setEnabled(name == "Gabor")
        self.waves_spin.setEnabled(name == "Fourier (spectral)")
        self.stretch_spin.setEnabled(name == "Strata (anisotropic)")
        self.flow_spin.setEnabled(name == "Flow")
        self.alpha_spin.setEnabled(name == "Colored noise (α)")
        self.hdim_combo.setEnabled(name == "Hyper noise (N-D)")
        self.taxes_combo.setEnabled(name == "Torus tile (any freq)")
        self.tperiod_spin.setEnabled(name == "Torus tile (any freq)")
        is_rd = name == "Reaction–diffusion"
        self.rd_feed.setEnabled(is_rd); self.rd_kill.setEnabled(is_rd)

    def _on_fade_changed(self, name):
        set_fade(name)
        self.fade_plot.set_active(name)
        self._slice_timer.start(80)
        self._schedule_volume()
        self._update_info()

    def _on_field_changed(self, *_):
        self._rebuild_noise()
        self._slice_timer.start(120)
        self._schedule_volume()
        self._update_info()
        self._sync_gpu()

    def _on_cmap_changed(self, *_):
        set_fade(self.fade_combo.currentText())  # no-op safety
        self.regenerate_slice()
        self._update_3d_items()
        self.gpu.set_colormap(COLORMAPS[self.cmap_combo.currentText()])
        self.anatomy.set_lut(COLORMAPS[self.cmap_combo.currentText()])

    def _on_volume_param_changed(self, *_):
        self._vol_dirty = True
        if self.volume_view.ok and self.tabs.currentIndex() == 1:
            self._vol_timer.start(120)

    def _on_tab_changed(self, idx):
        if idx == 1 and self._vol_dirty:
            self._vol_timer.start(0)
        if idx == self.tabs.indexOf(self.octave_lab):
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                self.octave_lab.compute(
                    self._noise, self.freq_spin.value(),
                    self.z_slider.value() / 100.0, self.oct_spin.value(),
                    self.gain_spin.value(), self.lac_spin.value(),
                    self.ftype_combo.currentText(),
                    COLORMAPS[self.cmap_combo.currentText()])
            finally:
                QApplication.restoreOverrideCursor()
        elif idx == self.tabs.indexOf(self.spec_view):
            if self._last_field is not None:
                self.spec_view.update_field(self._last_field)
        elif idx == self.tabs.indexOf(self.compare_view):
            self._refresh_compare()
        elif idx == self.tabs.indexOf(self.terrain_lab):
            self.terrain_lab.refresh(self._last_field,
                                     COLORMAPS[self.cmap_combo.currentText()])
        if hasattr(self, "gpu") and self.gpu.ok:
            pass  # GPU animation is user-driven via its play button

    def _refresh_compare(self):
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.compare_view.canvas.compute(
                self.compare_view.cmp_a.currentText(),
                self.compare_view.cmp_b.currentText(),
                self.seed_spin.value(), self.freq_spin.value(),
                self.oct_spin.value(), self.gain_spin.value(),
                self.lac_spin.value(), self.ftype_combo.currentText(),
                self.wmode_combo.currentText(),
                COLORMAPS[self.cmap_combo.currentText()])
        finally:
            QApplication.restoreOverrideCursor()

    def _schedule_volume(self):
        self._vol_dirty = True
        if self.volume_view.ok and self.tabs.currentIndex() == 1:
            self._vol_timer.start(350)

    def _random_seed(self):
        self.seed_spin.setValue(int(np.random.default_rng().integers(0, 1_000_000)))

    def _toggle_play(self):
        self._playing = not self._playing
        self.play_btn.setText("⏸  Pause" if self._playing else "▶  Animate z")
        if self._playing:
            self._play_timer.start()
        else:
            self._play_timer.stop()
            self.regenerate_slice()

    def _play_tick(self):
        v = self.z_slider.value() + 2
        if v > self.z_slider.maximum():
            v = 0
        self.z_slider.blockSignals(True)
        self.z_slider.setValue(v)
        self.z_slider.blockSignals(False)
        self.regenerate_slice()

    def _current_slice_res(self):
        res = int(self.slice_res_combo.currentText())
        return min(res, 160) if self._playing else res

    # ---------- computation ----------

    def regenerate_slice(self):
        if self._noise is None:
            return
        res = self._current_slice_res()
        freq = self.freq_spin.value()
        zt = self.z_slider.value() / 100.0
        t0 = time.perf_counter()
        self._last_field = make_slice(
            self._noise, self._warp_noise, res, freq, zt * freq,
            self.oct_spin.value(), self.gain_spin.value(),
            self.lac_spin.value(), self.ftype_combo.currentText(),
            warp=self.warp_spin.value(), steps=self.steps_spin.value(),
            plane=self.plane_combo.currentText())
        self._slice_ms = (time.perf_counter() - t0) * 1000.0
        self._slice_res_used = res
        self._last_freq = freq
        self._render_slice()
        self._update_canvas_overlays()
        lut = COLORMAPS[self.cmap_combo.currentText()]
        self.slice_canvas.set_colorbar(lut)
        self.slice_canvas.set_readout(self._last_field, freq, zt * freq)
        self.anatomy.set_lut(lut)
        self.anatomy.set_source(self.algo_combo.currentText(), seed=self.seed_spin.value())
        self._update_z_label()
        self._update_status()
        if self.tabs.currentIndex() == self.tabs.indexOf(self.terrain_lab):
            self.terrain_lab.refresh(self._last_field, lut)

    def _render_slice(self):
        v = self._last_field
        if v is None:
            return
        lut = COLORMAPS[self.cmap_combo.currentText()]
        view = self.view_combo.currentText()
        if view.startswith("Gradient"):
            gy, gx = np.gradient(v)
            mag = np.hypot(gx, gy)
            mag /= (np.percentile(mag, 99.5) + 1e-9)
            rgb = lut[(np.clip(mag, 0.0, 1.0) * 255.0).astype(np.uint8)]
        elif view.startswith("Shaded"):
            rgb = shaded_relief_rgb(v, lut)
        else:
            v01 = np.clip((v + 1.0) * 0.5, 0.0, 1.0)
            rgb = lut[(v01 * 255.0).astype(np.uint8)].copy()
            if view.startswith("Iso"):
                mask = isoline_mask(v01, 8)
                rgb[mask] = (rgb[mask].astype(np.float32) * 0.25).astype(np.uint8)
        self.slice_canvas.set_image(rgb_to_image(rgb.astype(np.uint8)),
                                    self._last_freq,
                                    self.lattice_chk.isChecked())
        res = v.shape[0]
        self.profile.set_curve(
            np.clip((v[res // 2] + 1.0) * 0.5, 0.0, 1.0),
            f"1-D profile along the marked row · {view}")

    def _update_canvas_overlays(self):
        canvas = self.slice_canvas
        canvas.set_corner_arrows([])
        canvas.set_feature_points([])
        if (self.plane_combo.currentText() != "XY"
                or not self.lattice_chk.isChecked()):
            canvas.set_arrows(None, None)
            return
        freq = self._last_freq
        zw = self.z_slider.value() / 100.0 * freq
        k = int(round(zw)) & 255
        n = int(math.ceil(freq)) + 1
        algo = self.algo_combo.currentText()
        if algo in ("Perlin", "Perlin (classic '85)"):
            p = self._noise.perm
            classic = algo != "Perlin"
            pts = []
            for i in range(n):
                for j in range(n):
                    h = p[p[(i & 255) + (j & 255)] + k]
                    if classic:
                        b = h & 7
                        vx = (-1.0 if b & 1 else 1.0) * 0.5774
                        vy = (-1.0 if b & 2 else 1.0) * 0.5774
                    else:
                        g = _GRAD3[h % 12]
                        vx, vy = float(g[0]), float(g[1])
                    pts.append((i / freq, j / freq, vx, vy))
            canvas.set_corner_arrows(pts)
        hook = getattr(self._noise, "slice_feature_points", None)
        if hook is not None:
            canvas.set_feature_points(hook(n, freq, k))
        if (self._deriv_noise is not None and self.arrows_chk.isChecked()
                and self.warp_spin.value() == 0.0
                and self.steps_spin.value() < 2):
            lin = (np.arange(res2 := self._slice_res_used,
                             dtype=np.float64) + 0.5) / res2 * freq
            yy, xx = np.meshgrid(lin, lin, indexing="ij")
            _, gx, gy, _ = self._deriv_noise(xx, yy, np.full_like(xx, zw))
            canvas.set_arrows(gx, gy)
        else:
            canvas.set_arrows(None, None)

    def _run_volume_job(self):
        if not (self.volume_view.ok and self._noise is not None):
            return
        self._vol_dirty = False
        self._vol_gen += 1
        gen = self._vol_gen
        noise, warp_noise = self._noise, self._warp_noise
        res = int(self.vol_res_combo.currentText())
        args = (self.freq_spin.value(), self.oct_spin.value(),
                self.gain_spin.value(), self.lac_spin.value(),
                self.ftype_combo.currentText(),
                self.warp_spin.value(), self.steps_spin.value())

        def task():
            t0 = time.perf_counter()
            vol = make_field(noise, warp_noise, res, *args)
            return vol, (time.perf_counter() - t0) * 1000.0

        sig = WorkerSignals()
        sig.finished.connect(
            lambda payload, ms: self._volume_ready(gen, payload, ms))
        QThreadPool.globalInstance().start(FunctionWorker(task, sig))
        self.statusBar().showMessage(f"computing {res}³ volume…")

    def _volume_ready(self, gen, payload, ms):
        if gen != self._vol_gen:
            return
        if isinstance(payload, str):
            QMessageBox.warning(self, "3-D computation failed", payload)
            return
        vol, self._vol_ms = payload
        self._vol01 = np.clip((vol + 1.0) * 0.5, 0.0, 1.0)
        self._update_3d_items()
        self._update_status()

    def _update_3d_items(self, *_):
        pct = self.level_slider.value()
        self.level_lbl.setText(f"iso level {pct / 100:.2f}")
        if self._vol01 is None:
            return
        self.volume_view.set_data(self._vol01, pct / 100.0,
                                  self.render_combo.currentText(),
                                  COLORMAPS[self.cmap_combo.currentText()])

    def _sync_gpu(self):
        if not (hasattr(self, "gpu") and self.gpu.ok):
            return
        self.gpu.set_params(
            freq=self.freq_spin.value(),
            z=self.z_slider.value() / 100.0 * self.freq_spin.value(),
            oct=self.oct_spin.value(), gain=self.gain_spin.value(),
            lac=self.lac_spin.value(),
            algo=0 if self.algo_combo.currentText() == "Value"
                 else self.gpu.algo_combo.currentIndex())

    def _update_z_label(self):
        zt = self.z_slider.value() / 100.0
        plane = self.plane_combo.currentText()
        axis = {"XY": "z", "XZ": "y", "YZ": "x"}[plane]
        self.z_lbl.setText(f"{axis} = {zt:.2f} periods · world "
                           f"{zt * self._last_freq:.2f} · plane {plane}")

    def _update_status(self):
        msg = f"2-D slice ({self._slice_res_used}px): {self._slice_ms:.1f} ms"
        if self.volume_view.ok:
            msg += f"   ·   3-D volume: {self._vol_ms:.0f} ms"
        else:
            msg += "   ·   3-D disabled (pip install pyqtgraph PyOpenGL)"
        self.statusBar().showMessage(msg)

    # ---------- info ----------

    def _update_info(self):
        if self._lesson_active:
            L = LESSONS[self._lesson_idx]
            html = (f"<h2>Lesson {self._lesson_idx + 1} / {len(LESSONS)}</h2>"
                    + L["h"])
        else:
            html = HEADER_HTML + ALGORITHMS[self.algo_combo.currentText()]["info"]
        internal = getattr(self._noise, "internal_fractal", False)
        if self.oct_spin.value() > 1 and not internal \
                and self.ftype_combo.currentText() != "fBm":
            pass
        if self.oct_spin.value() > 1 or self.ftype_combo.currentText() != "fBm":
            if not internal:
                html += ("<hr><h3>Fractal layering (fBm) — active</h3>"
                         "<p>Octave <i>i</i> uses lacunarity<sup>i</sup>× the "
                         "frequency and gain<sup>i</sup>× the amplitude; "
                         "normalising keeps output in [−1, 1].</p>")
        if self.fade_combo.currentText() != DEFAULT_FADE:
            html += FADE_HTML
        if self.warp_spin.value() > 0:
            html += WARP_HTML
        if self.steps_spin.value() >= 2:
            html += TERRACE_HTML
        if self.ftype_combo.currentText() == "Ping-pong":
            html += PINGPONG_HTML
        if internal:
            html += INTERNAL_HTML
        self.info_browser.setHtml(html)

    def _show_source(self):
        cls = ALGORITHMS[self.algo_combo.currentText()]["cls"]
        try:
            src = inspect.getsource(cls)
            extra = ("\n\n# ---- fractal layering used when Octaves > 1 ----\n\n"
                     + inspect.getsource(fbm))
            if self.algo_combo.currentText() == "Perlin":
                extra += ("\n\n# ---- analytic derivatives ----\n\n"
                          + inspect.getsource(PerlinDeriv3))
        except (OSError, TypeError):
            src, extra = "# source not available", ""
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Implementation — {cls.__name__}")
        lay = QVBoxLayout(dlg)
        edit = QPlainTextEdit()
        edit.setPlainText(f"# {cls.__name__}  "
                          f"('{self.algo_combo.currentText()}')\n\n" + src + extra)
        edit.setReadOnly(True)
        f = QFont("Consolas")
        f.setStyleHint(QFont.StyleHint.Monospace)
        f.setPointSize(10)
        edit.setFont(f)
        lay.addWidget(edit)
        dlg.resize(860, 720)
        dlg.show()

    # ---------- export ----------

    def _export_slice(self):
        img = self.slice_canvas._img
        if img is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export slice",
                                              "noise_slice.png",
                                              "PNG image (*.png)")
        if path:
            img.save(path)
            self.statusBar().showMessage(f"saved {path}")

    def _export_stack(self):
        if self._vol01 is None:
            QMessageBox.information(
                self, "No volume yet",
                "Open the 3-D tab once so the volume gets computed, then export.")
            return
        d = QFileDialog.getExistingDirectory(
            self, "Choose a folder for slice_###.png files")
        if not d:
            return
        lut = COLORMAPS[self.cmap_combo.currentText()]
        n = self._vol01.shape[0]
        for k in range(n):
            values_to_image(self._vol01[k], lut).save(
                os.path.join(d, f"slice_{k:03d}.png"))
        self.statusBar().showMessage(f"exported {n} slices to {d}")


# ============================================================================
# 14. ENTRY POINT
# ============================================================================

def apply_dark_theme(app):
    p = QPalette()
    c = QColor
    p.setColor(QPalette.ColorRole.Window, c(30, 30, 36))
    p.setColor(QPalette.ColorRole.WindowText, c(220, 220, 225))
    p.setColor(QPalette.ColorRole.Base, c(24, 24, 29))
    p.setColor(QPalette.ColorRole.AlternateBase, c(35, 35, 42))
    p.setColor(QPalette.ColorRole.Text, c(220, 220, 225))
    p.setColor(QPalette.ColorRole.Button, c(45, 45, 52))
    p.setColor(QPalette.ColorRole.ButtonText, c(220, 220, 225))
    p.setColor(QPalette.ColorRole.Highlight, c(70, 130, 200))
    p.setColor(QPalette.ColorRole.HighlightedText, c(255, 255, 255))
    p.setColor(QPalette.ColorRole.ToolTipBase, c(50, 50, 58))
    p.setColor(QPalette.ColorRole.ToolTipText, c(220, 220, 225))
    app.setPalette(p)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    apply_dark_theme(app)
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.Profile.CoreProfile)
    fmt.setDepthBufferSize(24)
    QSurfaceFormat.setDefaultFormat(fmt)          # before any GL widget exists

    def cleanup():
        pool = QThreadPool.globalInstance()
        pool.clear()
        pool.waitForDone(2000)

    app.aboutToQuit.connect(cleanup)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()