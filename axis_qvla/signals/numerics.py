"""Host-independent numerics for the locked signal files whose bytes depend on how a host rounds a
transcendental function.

OFFLINE TIER: numpy only.

log2_extended. `approach_aware.chunk.tags` stamps `meta.q_given_task_entropy_ratio`, a sum of
3,628 `p * log2(p)` terms (3,623 conditional, 5 marginal). `np.log2` is not correctly rounded
everywhere: which implementation runs depends on numpy's SIMD dispatch and on the libm, and on a
host without AVX-512 whose libm rounds otherwise, 206 conditional and 4 marginal terms come out one
ulp off and the ratio's last digits move (0.9368617343176477 for the locked 0.936861734317648). The
locked float is what a correctly rounded log2 gives (60-digit decimal reference). Evaluating log2
in long double and rounding once to float64 gives the correctly rounded value on 3,627 of the
3,628 terms with x86-64's 64-bit mantissa (one double rounding, which does not move the sum) and
the locked float exactly. `EXTENDED` says whether this host's long double has a 64-bit or wider
mantissa; where it is plain double, the generator map (conf/signals/generators.json) declares a
tolerance on that one key.

log2_libm. `object_centric.chunk.tags` stamps the same kind of ratio, over 3,643 terms, and its
locked float is what glibc 2.28's double log2 gives: the fdlibm `__ieee754_log2` of its x86-64
build (reduction to [sqrt(2)/2, sqrt(2)), a degree-14 polynomial in s = f/(2+f)), which rounds 208
of the terms differently from the correctly rounded value. numpy's AVX-512 float64 kernel rounds
210 of the terms differently and moves the ratio by one ulp, and the table-based log2 of glibc 2.29
and later rounds 211 differently (measured under glibc 2.36). `log2_libm` is that glibc 2.28
function itself: the same integer bit manipulation and the same float64 operations in the same
order, each a correctly rounded IEEE operation (the build has no FMA variant, so none is fused), so
it returns glibc 2.28's value on any host, CPU or C library. Measured: on 14,009,389 inputs over
every binade (subnormals, the reduction boundaries and every k/n with n <= 2000 included) it equals
glibc 2.28's `log2` value for value, and its digest is the same on an x86-64 CPU without AVX2, one
with AVX2+FMA3, one with AVX-512, and under glibc 2.36, whose own `log2` differs on 1,821,602 of
them.

tanh_f32. The approach-aware chunk AWR weights take `contact = 1 - np.tanh(K * d)` of float32
distances. Where the CPU has native FMA (x86-64 AVX2+FMA3 or AVX-512, aarch64 NEON), numpy computes
float32 tanh with its own SIMD kernel (an SVML port) and the locked file was written that way; on
a CPU without it (an x86-64 without AVX2) numpy calls the C library's tanhf, which rounds 44.6% of
the builder's 1,906,469 inputs 1-3 ulp differently and moves 16.7% of the weights by 1-6 ulp.
`tanh_f32` is that kernel's own algorithm -- the same 32-entry table, range reduction and degree-6
Horner chain -- with each float32 fused multiply-add computed exactly in float64: the product of two
float32 is exact in float64, the sum is rounded to odd (TwoSum and a parity fix), and rounding to
odd at 53 bits, then to nearest at 24, is one correct rounding (Boldo & Melquiond 2008). Every
operation is a correctly rounded IEEE float64 op, so the kernel's bytes come out on every host: it
equals numpy's SIMD tanh on all 2^32 float32 inputs (AVX-512 and AVX2 hosts) and hashes the same on
a host without AVX2. It takes a float32 array and nothing else, and refuses any other dtype rather
than passing it to numpy: float64 tanh is dispatched on FMA the same way.
"""
from __future__ import annotations

import numpy as np

EXTENDED = np.finfo(np.longdouble).nmant >= 63


def log2_extended(x):
    """log2 evaluated in long double, rounded once to float64."""
    r = np.log2(np.asarray(x, dtype=np.longdouble)).astype(np.float64)
    return r[()] if r.ndim == 0 else r


# glibc 2.28 sysdeps/ieee754/dbl-64/wordsize-64/e_log2.c (the x86-64 build) constants, as bit
# patterns: ln2, 2^54, the 1/3 of its `|f| < 2**-20` branch, and the polynomial's Lg1..Lg7.
_LN2, _TWO54, _THIRD, _LG1, _LG2, _LG3, _LG4, _LG5, _LG6, _LG7 = np.array([
    0x3FE62E42FEFA39EF, 0x4350000000000000, 0x3FD5555555555555, 0x3FE5555555555593, 0x3FD999999997FA04,
    0x3FD2492494229359, 0x3FCC71C51D8E78AF, 0x3FC7466496CB03DE, 0x3FC39A09D078C69F, 0x3FC2F112DF3E5244,
], dtype=np.uint64).view(np.float64)
_MANT = np.int64(0x000FFFFFFFFFFFFF)


def log2_libm(x):
    """log2 of float64 exactly as glibc 2.28's x86-64 `__ieee754_log2` computes it, on any host.

    The result has the input's shape (a float64 scalar for a 0-d input); zero gives -inf, a
    negative value nan and +inf +inf, as `np.log2` does."""
    a = np.asarray(x, dtype=np.float64)
    v = a.ravel()
    out = np.empty_like(v)
    hx = v.view(np.int64)
    with np.errstate(all="ignore"):
        zero, nan = v == 0.0, np.isnan(v)
        neg = (hx < 0) & ~zero & ~nan                                  # includes -inf
        top = nan | (hx >= np.int64(0x7FF0000000000000))               # nan, +inf: x + x
        out[zero], out[neg], out[top] = -np.inf, np.nan, v[top] + v[top]
        m = ~(zero | neg | top)
        x, h = v[m].copy(), hx[m]
        sub = h < np.int64(0x0010000000000000)                         # subnormal: scale up by 2^54
        x[sub] *= _TWO54
        h = np.where(sub, x.view(np.int64), h)
        k = np.where(sub, -54, 0).astype(np.int64) + (h >> 52) - 1023
        h &= _MANT
        i = (h + np.int64(0x95F6400000000)) & np.int64(0x10000000000000)
        x = (h | (i ^ np.int64(0x3FF0000000000000))).view(np.float64)    # x or x/2, in [sqrt(2)/2, sqrt(2))
        dk = (k + (i >> 52)).astype(np.float64)
        f = x - 1.0
        r = np.empty_like(f)
        tiny = (_MANT & (2 + h)) < 3
        ft, dt = f[tiny], dk[tiny]
        rt = ft * ft * (0.5 - _THIRD * ft)
        r[tiny] = np.where(ft == 0.0, dt, dt - (rt - ft) / _LN2)
        f, dk, h = f[~tiny], dk[~tiny], h[~tiny]
        s = f / (2.0 + f)
        z = s * s
        w = z * z
        t1 = w * (_LG2 + w * (_LG4 + w * _LG6))
        t2 = z * (_LG1 + w * (_LG3 + w * (_LG5 + w * _LG7)))
        rr = t2 + t1
        hfsq = 0.5 * f * f
        wide = ((h - np.int64(0x6147A00000000)) | (np.int64(0x6B85100000000) - h)) > 0
        r[~tiny] = np.where(wide, dk - ((hfsq - (s * (hfsq + rr))) - f) / _LN2, dk - ((s * (f - rr)) - f) / _LN2)
        out[m] = r
    out = out.reshape(a.shape)
    return out[()] if out.ndim == 0 else out


# numpy's float32 tanh table (lut32x8, loops_hyperbolic.dispatch.cpp.src; the same in numpy 1.26.4
# and 2.4.4) as bit patterns: rows b, c0, c1, ..., c6 of 32 entries each.
_TANH_LUT = np.array([
    [0x0,        0x3d700000, 0x3d900000, 0x3db00000, 0x3dd00000, 0x3df00000, 0x3e100000, 0x3e300000,
     0x3e500000, 0x3e700000, 0x3e900000, 0x3eb00000, 0x3ed00000, 0x3ef00000, 0x3f100000, 0x3f300000,
     0x3f500000, 0x3f700000, 0x3f900000, 0x3fb00000, 0x3fd00000, 0x3ff00000, 0x40100000, 0x40300000,
     0x40500000, 0x40700000, 0x40900000, 0x40b00000, 0x40d00000, 0x40f00000, 0x41100000, 0x0],
    [0x0,        0x3d6fb9c9, 0x3d8fc35f, 0x3daf9169, 0x3dcf49ab, 0x3deee849, 0x3e0f0ee8, 0x3e2e4984,
     0x3e4d2f8e, 0x3e6bb32e, 0x3e8c51cd, 0x3ea96163, 0x3ec543f1, 0x3edfd735, 0x3f028438, 0x3f18abf0,
     0x3f2bc480, 0x3f3bec1c, 0x3f4f2e5b, 0x3f613c53, 0x3f6ce37d, 0x3f743c4f, 0x3f7a5feb, 0x3f7dea85,
     0x3f7f3b3d, 0x3f7fb78c, 0x3f7fefd4, 0x3f7ffdd0, 0x3f7fffb4, 0x3f7ffff6, 0x3f7fffff, 0x3f800000],
    [0x3f800000, 0x3f7f1f84, 0x3f7ebd11, 0x3f7e1e5f, 0x3f7d609f, 0x3f7c842d, 0x3f7b00e5, 0x3f789580,
     0x3f75b8ad, 0x3f726fd9, 0x3f6cc59b, 0x3f63fb92, 0x3f59ff97, 0x3f4f11d7, 0x3f3d7573, 0x3f24f360,
     0x3f0cbfe7, 0x3eec1a69, 0x3eb0a801, 0x3e6753a2, 0x3e132f1a, 0x3db7e7d3, 0x3d320845, 0x3c84d3d4,
     0x3bc477b7, 0x3b10d3da, 0x3a01601e, 0x388c1a3b, 0x3717b0da, 0x35a43bce, 0x338306c6, 0x0],
    [0xb0343c7b, 0xbd6ee69d, 0xbd8f0da7, 0xbdae477d, 0xbdcd2a1f, 0xbdeba80d, 0xbe0c443b, 0xbe293cf3,
     0xbe44f282, 0xbe5f3651, 0xbe81c7c0, 0xbe96d7ca, 0xbea7fb8e, 0xbeb50e9e, 0xbec12efe, 0xbec4be92,
     0xbebce070, 0xbead510e, 0xbe8ef7d6, 0xbe4b8704, 0xbe083237, 0xbdaf7449, 0xbd2e1ec4, 0xbc83bf06,
     0xbbc3e0b5, 0xbb10aadc, 0xba0157db, 0xb88c18f2, 0xb717b096, 0xb5a43bae, 0xb383012c, 0x0],
    [0xbeaaaaa5, 0xbeab0612, 0xbea7f01f, 0xbea4e120, 0xbea387b7, 0xbea15962, 0xbe9d57f7, 0xbe976b5a,
     0xbe90230d, 0xbe880dff, 0xbe7479b3, 0xbe4c3d88, 0xbe212482, 0xbdeb8cba, 0xbd5e78ad, 0x3c6b5e6e,
     0x3d839143, 0x3dc21ee1, 0x3de347af, 0x3dcbec96, 0x3d99ef2d, 0x3d542ea1, 0x3cdde701, 0x3c2cca67,
     0x3b81cb27, 0x3ac073a1, 0x39ac3032, 0x383a94d9, 0x36ca081d, 0x355abd4c, 0x332b3cb6, 0x0],
    [0xb76dd6b9, 0xbe1c276d, 0x3c1dcf2f, 0x3dc1a78d, 0x3d96f985, 0x3da2b61b, 0x3dc13397, 0x3dd2f670,
     0x3df48a0a, 0x3e06c5a8, 0x3e1a3aba, 0x3e27c405, 0x3e2e78d0, 0x3e2c3e44, 0x3e1d3097, 0x3df4a8f4,
     0x3da38508, 0x3d31416a, 0x3b562657, 0xbcaeeac9, 0xbcce9419, 0xbcaaeac4, 0xbc49e7d0, 0xbba71ddd,
     0xbb003b0e, 0xba3f9a05, 0xb92c08a7, 0xb7ba9232, 0xb64a0b0f, 0xb4dac169, 0xb2ab78ac, 0x0],
    [0x3e0910e9, 0x43761143, 0x4165ecdc, 0xc190f756, 0xc08c097d, 0xc02ba813, 0xbf7f6bda, 0x3f2b1dc0,
     0x3ece105d, 0x3f426a94, 0xbadb0dc4, 0x3da43b17, 0xbd51ab88, 0xbcaea23d, 0xbd3b6d8d, 0xbd6caaad,
     0xbd795bed, 0xbd5fddda, 0xbd038f3b, 0xbc1cad63, 0x3abb4766, 0x3b95f10b, 0x3b825873, 0x3afaea66,
     0x3a49f878, 0x39996bf3, 0x388f3e6c, 0x371bb0e3, 0x35a8a5e6, 0x34369b17, 0x322487b0, 0x0],
    [0xbc0e2f66, 0x460bda12, 0x43d638ef, 0xc3e11c3e, 0xc2baa4e9, 0xc249da2d, 0xc1859b82, 0x40dd5b57,
     0x40494640, 0x40c730a8, 0xbf0f160e, 0x3e30e76f, 0xbea81387, 0xbdb26a1c, 0xbd351e57, 0xbb4c01a0,
     0x3c1d7bfb, 0x3c722cd1, 0x3c973f1c, 0x3c33a31b, 0x3b862ef4, 0x3a27b3d0, 0xba3b5907, 0xba0efc22,
     0xb97f9f0f, 0xb8c8af50, 0xb7bdddfb, 0xb64f2950, 0xb4e085b1, 0xb3731dfa, 0xb15a1f04, 0x0],
], dtype=np.uint32).view(np.float32)


def _fma_f32(a, b, c):
    """a*b + c of float32 arrays with ONE rounding, from correctly rounded float64 ops only."""
    p = a.astype(np.float64) * b.astype(np.float64)            # exact: 24 + 24 significant bits
    c = c.astype(np.float64)
    s = p + c                                                    # rounded to nearest
    bb = s - p
    e = (p - (s - bb)) + (c - bb)                                # TwoSum: s + e == p + c exactly
    fix = (e != 0) & ((s.view(np.int64) & 1) == 0)               # round to odd: inexact and even -> odd
    s = np.where(fix, np.nextafter(s, np.where(e > 0, np.inf, -np.inf)), s)
    return s.astype(np.float32)                                  # RN24(RO53(x)) == RN24(x)


def tanh_f32(x):
    """np.tanh of float32, exactly as numpy's FMA SIMD kernel computes it, on any host.

    `x` is a float32 array or scalar; the result has its shape (a float32 scalar for a 0-d
    input, as the ufunc gives). Any other dtype is refused, never cast."""
    x = np.asarray(x)
    if x.dtype != np.float32:
        raise TypeError(f"tanh_f32 takes float32 only, got {x.dtype}")
    b, c0, c1, c2, c3, c4, c5, c6 = _TANH_LUT
    flat = x.ravel()
    xi = flat.view(np.int32)
    ndnan = xi & np.int32(0x7FE00000)
    special = ndnan <= np.int32(0x7F000000)                      # False: inf, nan, |x| >= 1.25 * 2^127
    idx = (np.clip(ndnan - np.int32(0x3D400000), 0, 0x3E00000).view(np.uint32) >> np.uint32(21)).astype(np.intp)
    with np.errstate(invalid="ignore"):                         # inf/nan lanes; overwritten below
        y = np.abs(flat) - b[idx]                                # float32 subtract, as the kernel does
        r = _fma_f32(c6[idx], y, c5[idx])
        for c in (c4, c3, c2, c1, c0):
            r = _fma_f32(r, y, c[idx])
    r = np.where(special, r, np.float32(1.0))
    r = (r.view(np.uint32) | (xi.view(np.uint32) & np.uint32(0x80000000))).view(np.float32)
    r = np.where(np.isnan(flat), np.float32(np.nan), r).reshape(x.shape)
    return r[()] if r.ndim == 0 else r
