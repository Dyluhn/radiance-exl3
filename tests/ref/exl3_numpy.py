"""exl3_numpy.py -- an independent NumPy transcription of the two EXL3 expert ops, for checking kernels/exl3ref.

Nothing here calls the C code under test or common/: the trellis decode is transcribed from Dylan's torch reference
(exl3-moe kernels/trellis.py, itself from exllamav3 @ 12414d0) and checked bit for bit against the vllm-exl3 golden
(exl3_golden.npz) by `selfcheck_decode`; the op math is transcribed from docs/OPS-exl3.md.

Radiance's view of one weight is [rows = n (out), cols = k (in)]; the canonical codes plane holds, at (i, j), the K-bit
symbol of the trellis position that decodes W_hat[k = j][n = i] (include/exl3_encoding.h, common/exl3_decode.h).

Rounding is done where the op contract says and nowhere else: f32 arithmetic is numpy float32 (IEEE single, no FMA),
f16 narrowing is numpy's float32 -> float16 cast (round to nearest even), bf16 narrowing is `to_bf16` below.
"""
import numpy as np

HAD = 128
MUL1_MULT = 0x83DCD12D
MCG_MULT = 0xCBAC1FED
INST3_MULT, INST3_ADD = 89226354, 64248484
K_INV = float(np.frombuffer(np.uint16(0x1EEE).tobytes(), np.float16)[0])
K_BIAS = float(np.frombuffer(np.uint16(0xC931).tobytes(), np.float16)[0])
CODEBOOKS = {"3inst": 0, "mcg": 1, "mul1": 2}
RS = np.float32(1.0 / np.sqrt(128.0))  # (float)(1/sqrt(128)), one rounding of the double


# ----------------------------------------------------------------------------------------------- narrowing
def to_f16(x):
    return np.asarray(x, np.float32).astype(np.float16)


def to_bf16_bits(x):
    """float32 -> bf16 bits, round to nearest even, NaN kept quiet (abi/rad_plugin.h rad_f32_to_bf16)."""
    u = np.asarray(x, np.float32).view(np.uint32).astype(np.uint64)
    nan = (u & 0x7FFFFFFF) > 0x7F800000
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    return np.where(nan, ((u >> 16) | 0x40).astype(np.uint16), r)


def bf16_bits_to_f32(b):
    return (np.asarray(b, np.uint16).astype(np.uint32) << 16).view(np.float32)


# ----------------------------------------------------------------------------------------------- the tile
def pos_to_tile():
    """(row, col) inside the 16x16 tile of each trellis position p = 0..255 (row = k axis, col = n axis)."""
    p = np.arange(256)
    lane, j = p // 8, p % 8
    row = 2 * (lane % 4) + np.array([0, 1, 8, 9])[j % 4]
    col = lane // 4 + 8 * (j // 4)
    return row, col


def codebook_values(state, codebook):
    """16-bit trellis states -> fp16 values, one rounding of the exact sum (exllamav3 decode_3inst<cb>)."""
    x = np.asarray(state, np.uint64) & 0xFFFF
    if codebook == "mul1":
        x = (x * MUL1_MULT) & 0xFFFFFFFF
        s = (x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + (x >> 24)
        return ((1024 + s).astype(np.float64) * K_INV + K_BIAS).astype(np.float16)
    if codebook == "mcg":
        x = (x * MCG_MULT) & 0xFFFFFFFF
    elif codebook == "3inst":
        x = (x * INST3_MULT + INST3_ADD) & 0xFFFFFFFF
    else:
        raise ValueError(codebook)
    x = (x & 0x8FFF8FFF) ^ 0x3B603B60
    lo = (x & 0xFFFF).astype(np.uint16).view(np.float16).astype(np.float64)
    hi = (x >> 16).astype(np.uint16).view(np.float16).astype(np.float64)
    return (lo + hi).astype(np.float16)


def tile_states(sym, bits):
    """[..., 256] K-bit symbols -> [..., 256] 16-bit states: the window of the MSB-first symbol stream ending at
    symbol p's last bit, circular within the tile (tail-biting). Read as an integer that window is symbol p in the
    low K bits, symbol p-1 in the next K, and so on, so it is the low 16 bits of the last ceil(16/K) symbols
    shifted together."""
    s = sym.astype(np.uint64)
    state = np.zeros_like(s)
    for j in range(-(-16 // bits)):
        state |= np.roll(s, j, axis=-1) << np.uint64(j * bits)
    return state & 0xFFFF


def tile_states_by_window(sym, bits):
    """The same states read off the bit stream literally (the definition); `selfcheck_decode` holds the two equal."""
    shifts = np.arange(bits - 1, -1, -1, dtype=np.uint64)
    stream = ((sym.astype(np.uint64)[..., None] >> shifts) & 1).reshape(*sym.shape[:-1], 256 * bits)
    end = (np.arange(256) + 1) * bits
    index = (end[:, None] - 16 + np.arange(16)) % (256 * bits)  # [256, 16], MSB first
    window = stream[..., index]  # [..., 256, 16]
    return (window << np.arange(15, -1, -1, dtype=np.uint64)).sum(-1)


def symbols_from_stream(tile_words, bits):
    """exllamav3 trellis tile(s) [..., 16*K] int16 -> [..., 256] symbols: the u32 little-endian words as one MSB-first
    stream, symbol p = bits [p*K, (p+1)*K)."""
    words = np.ascontiguousarray(tile_words).view(np.uint32).astype(np.uint64)
    stream = ((words[..., None] >> np.arange(31, -1, -1, dtype=np.uint64)) & 1).reshape(*words.shape[:-1], -1)
    stream = stream.reshape(*stream.shape[:-1], 256, bits)
    return (stream << np.arange(bits - 1, -1, -1, dtype=np.uint64)).sum(-1)


# ----------------------------------------------------------------------------------------------- the codes plane
def codes_row_bytes(cols, bits):
    return (cols * bits + 7) // 8


def pack_codes(sym_plane, bits):
    """[rows, cols] K-bit values -> the uK plane bytes, rows dense, LSB-first along each row."""
    rows, cols = sym_plane.shape
    b = ((sym_plane.astype(np.uint32)[..., None] >> np.arange(bits, dtype=np.uint32)) & 1).astype(np.uint8)
    flat = b.reshape(rows, cols * bits)
    pad = codes_row_bytes(cols, bits) * 8 - cols * bits
    if pad:
        flat = np.concatenate([flat, np.zeros((rows, pad), np.uint8)], 1)
    return np.packbits(flat, axis=1, bitorder="little")


def unpack_codes(plane_bytes, rows, cols, bits):
    bitsarr = np.unpackbits(plane_bytes.reshape(rows, -1), axis=1, bitorder="little")[:, : cols * bits]
    b = bitsarr.reshape(rows, cols, bits).astype(np.uint32)
    return (b << np.arange(bits, dtype=np.uint32)).sum(-1)


def what_from_codes(sym_plane, bits, codebook):
    """Canonical codes plane [rows = n, cols = k] of symbols -> W_hat in radiance's view, float32 [n][k] (each an
    exact fp16 value)."""
    rows, cols = sym_plane.shape
    r, c = pos_to_tile()
    nt, kt = rows // 16, cols // 16
    # tile (tk, tn), position p sits at plane (i = 16 tn + c_p, j = 16 tk + r_p)
    tiles = sym_plane.reshape(nt, 16, kt, 16)  # [tn, c, tk, r]
    sym = tiles[:, c, :, r]  # advanced indices first: [256, tn, tk]
    sym = np.moveaxis(sym, 0, -1)  # [tn, tk, 256]
    vals = codebook_values(tile_states(sym, bits), codebook).astype(np.float32)  # [tn, tk, 256]
    out = np.empty((nt, 16, kt, 16), np.float32)
    out[:, c, :, r] = np.moveaxis(vals, -1, 0)
    return out.reshape(rows, cols)


def selfcheck_decode(golden_path):
    """The decoder above against the vllm-exl3 golden, bit for bit, via the canonical plane. Returns a list of
    (case, ok) pairs."""
    z = np.load(golden_path)
    r, c = pos_to_tile()
    out = []
    for bits in (3, 4):
        for cb in ("mcg", "mul1"):
            trellis = z[f"trellis_k{bits}_{cb}"]  # [kt, nt, 16K]
            want = z[f"w_hat_k{bits}_{cb}"].view(np.float16)  # [k, n]
            kt, nt = trellis.shape[:2]
            sym = symbols_from_stream(trellis, bits)  # [kt, nt, 256]
            plane = np.zeros((nt * 16, kt * 16), np.uint32)
            for tk in range(kt):
                for tn in range(nt):
                    plane[16 * tn + c, 16 * tk + r] = sym[tk, tn]
            got = what_from_codes(plane, bits, cb).astype(np.float16)  # [n][k]
            out.append((f"k{bits}_{cb}", np.array_equal(got.T.view(np.uint16), want.view(np.uint16))))
    rng = np.random.default_rng(7)
    for bits in range(1, 9):
        sym = rng.integers(0, 1 << bits, size=(3, 256))
        out.append((f"states_k{bits}", np.array_equal(tile_states(sym, bits), tile_states_by_window(sym, bits))))
    return out


# ----------------------------------------------------------------------------------------------- the rotation
def had128(v):
    """Normalised natural-order Sylvester Hadamard over each 128-block of the last axis: the butterfly in f32,
    stages s = 1, 2, ..., 64 (a+b low, a-b high), then one f32 multiply by (float)(1/sqrt(128))."""
    v = np.array(v, np.float32, copy=True)
    shape = v.shape
    n = shape[-1]
    assert n % HAD == 0
    v = v.reshape(-1, n // HAD, HAD)
    s = 1
    while s < HAD:
        w = v.reshape(v.shape[0], v.shape[1], HAD // (2 * s), 2, s)
        a, b = w[..., 0, :].copy(), w[..., 1, :].copy()
        w[..., 0, :] = a + b
        w[..., 1, :] = a - b
        v = w.reshape(v.shape[0], v.shape[1], HAD)
        s *= 2
    return (v * RS).reshape(shape)


# ----------------------------------------------------------------------------------------------- one expert
def plane_bytes(p, rows, cols, bits):
    """Bytes of one entry of plane table p (0 codes, 1 suh, 2 suh.1, 3 svh, 4 dn_suh) of a [rows, cols] weight."""
    return rows * codes_row_bytes(cols, bits) if p == 0 else 2 * {1: cols, 2: cols, 3: rows, 4: rows // 2}[p]


class Expert:
    """One expert's weight for one op: the canonical planes and W_hat (decoded here)."""

    def __init__(self, sym, suh, svh, suh1, bits, codebook):
        self.sym, self.bits, self.codebook = sym, bits, codebook
        self.suh, self.svh, self.suh1 = suh, svh, suh1  # float16 arrays (suh1 None unless stacked)
        self.dn_suh = None  # stacked gate_up only: DOWN's suh of the same expert, for h_rot = 1
        self.rows, self.cols = sym.shape
        self._what = None

    @property
    def what(self):
        if self._what is None:
            self._what = what_from_codes(self.sym, self.bits, self.codebook)
        return self._what


def random_expert(rng, rows, cols, bits, codebook, stacked):
    """Uniform random symbols (every sequence is a valid trellis), suh/svh random signs times scales near the
    magnitudes real checkpoints carry (trellis.py random_tensor)."""
    sym = rng.integers(0, 1 << bits, size=(rows, cols), dtype=np.uint32)
    sign = lambda m: rng.integers(0, 2, size=m) * 2 - 1  # noqa: E731
    scale = lambda m: 0.5 + rng.random(m)  # noqa: E731
    suh = (sign(cols) * scale(cols) * 0.01).astype(np.float16)
    svh = (sign(rows) * scale(rows)).astype(np.float16)
    suh1 = (sign(cols) * scale(cols) * 0.01).astype(np.float16) if stacked else None
    # No dn_suh drawn here: it is DOWN's suh of the same expert, set by whoever pairs the two (and drawing it
    # here would move every seeded stream that callers of version 2 rely on).
    return Expert(sym, suh, svh, suh1, bits, codebook)


# ----------------------------------------------------------------------------------------------- the routing
def moe_scatter(expert_ids, n_expert, expert_base=0, protect=()):
    """libref's moe_scatter (libref/ref_moe.cpp:106-151 at radiance 1.3.0): a stable counting sort of the flattened
    (token, slot) pairs by expert; a slot outside [expert_base, expert_base + n_expert) is dropped and the tail of
    sorted_tok is -1; `protect` experts are placed last in the order listed."""
    ids = np.asarray(expert_ids).reshape(-1)
    T = ids.size
    place = [e for e in range(n_expert) if e not in protect] + list(protect)
    rank = {e: i for i, e in enumerate(place)}
    local = ids - expert_base
    where = np.array([rank[e] if 0 <= e < n_expert else -1 for e in local], np.int64)
    count = np.array([(where == e).sum() for e in range(n_expert)], np.int32)
    off = np.zeros(n_expert + 1, np.int32)
    off[1:] = np.cumsum(count)
    sorted_tok = np.full(T, -1, np.int32)
    w = 0
    for e in range(n_expert):
        for i in np.nonzero(where == e)[0]:
            sorted_tok[w] = i
            w += 1
    return sorted_tok, off, count


def owner(off, i):
    """The expert whose offset range holds row i: the largest e with off[e] <= i."""
    return int(np.searchsorted(off, i, side="right") - 1)


# ----------------------------------------------------------------------------------------------- the ops
def _gemm_rows(xr, w_hat):
    """acc[r, n] = sum over k of xr[r, k] * w_hat[n, k], f32, ascending k, one product and one add per step (the
    products of two fp16 values are exact in f32, so only the order of the sum can differ -- and it does not)."""
    acc = np.zeros((xr.shape[0], w_hat.shape[0]), np.float32)
    for k in range(xr.shape[1]):
        acc = (acc + xr[:, k : k + 1] * w_hat[None, :, k]).astype(np.float32)
    return acc


def _prologue(x, suh):
    """f16(had128(f16(x * suh))) as float32."""
    xs = to_f16(x.astype(np.float32) * suh.astype(np.float32)[None, :]).astype(np.float32)
    return to_f16(had128(xs)).astype(np.float32)


def silu(g):
    """g * (1 / (1 + exp(-g))) in f32; exp is the correctly rounded f32 of the double (libm's expf agrees but for
    the rarest ties, which is why gate_up is compared to one ulp and down to the bit)."""
    e = np.exp(-g.astype(np.float64)).astype(np.float32)
    return (g * (np.float32(1) / (np.float32(1) + e))).astype(np.float32)


def _live_rows(T, sorted_tok, off, n_expert):
    """{expert: [rows]} for the rows below off[n_expert] whose sorted_tok is not negative."""
    rows = {}
    for i in range(min(T, int(off[n_expert]))):
        if sorted_tok[i] >= 0:
            rows.setdefault(owner(off[: n_expert + 1], i), []).append(i)
    return rows


UNWRITTEN = 0xFFFF  # what the harness prefills an output with (0xFF bytes): an op that does not write a row leaves it


def prep_gu(a, experts, sorted_tok, off, n_expert, top_k, x_cols, swap_suh=False):
    """moe_exl3_prep_gu: xr f16 bits, uint16 [T, 2 * x_cols]. Row i of a run: gate's rotated input in [0, C), up's in
    [C, 2C) (C the expert's input width) -- gate_up's step 1 for both halves; zero if sorted_tok < 0. Rows outside
    the runs and columns past 2C are not written (UNWRITTEN here, the harness's prefill)."""
    T = sorted_tok.size
    xr = np.full((T, 2 * x_cols), UNWRITTEN, np.uint16)
    a32 = a.astype(np.float32)
    for e in range(n_expert):
        ex = experts[e]
        C = ex.cols
        s0, s1 = (ex.suh1, ex.suh) if swap_suh else (ex.suh, ex.suh1)
        for i in range(int(off[e]), int(off[e + 1])):
            if sorted_tok[i] < 0:
                xr[i, : 2 * C] = 0
                continue
            x = a32[sorted_tok[i] // top_k : sorted_tok[i] // top_k + 1, :C]
            xr[i, :C] = to_f16(_prologue(x, s0)).view(np.uint16)[0]
            xr[i, C : 2 * C] = to_f16(_prologue(x, s1)).view(np.uint16)[0]
    return xr


def gate_up(a, experts, sorted_tok, off, n_expert, top_k, y_cols, swap_suh=False, h_rot=0, a_rot=0):
    """moe_exl3_gate_up: h f16 [T, y_cols]. experts[e] is expert e's stacked Expert (offsets index order).
    `swap_suh` is a deliberate error (the up half's scales on the gate), for the comparison's negative control.
    h_rot = 1: h is then rotated for down -- down's own first step, with down's suh (the expert's `dn_suh`).
    a_rot = 1: `a` is prep_gu's xr (sorted rows, f16 values), whose halves ARE the rotated inputs."""
    T = sorted_tok.size
    h = np.zeros((T, y_cols), np.float16)
    a32 = a.astype(np.float32)
    for e, rows in _live_rows(T, sorted_tok, off, n_expert).items():
        ex = experts[e]
        width, K = ex.rows // 2, ex.cols
        x = a32[sorted_tok[rows] // top_k, :K] if not a_rot else None
        su_g, su_u = (ex.suh1, ex.suh) if swap_suh else (ex.suh, ex.suh1)
        halves = []
        for p, su in ((0, su_g), (1, su_u)):
            xr = a32[rows, p * K : (p + 1) * K] if a_rot else _prologue(x, su)
            acc = _gemm_rows(xr, ex.what[p * width : (p + 1) * width])
            halves.append((had128(acc) * ex.svh[p * width : (p + 1) * width].astype(np.float32)).astype(np.float32))
        hv = to_f16(silu(halves[0]) * halves[1])
        h[rows, :width] = to_f16(_prologue(hv.astype(np.float32), ex.dn_suh)) if h_rot else hv
    return h


def down(hin, experts, sorted_tok, off, n_expert, y_cols, h_rot=0):
    """moe_exl3_down: y as bf16 bits, uint16 [T, y_cols]. h_rot = 1: hin is already rotated (gate_up did step 1)."""
    T = sorted_tok.size
    y = np.zeros((T, y_cols), np.uint16)
    h32 = hin.astype(np.float32)
    for e, rows in _live_rows(T, sorted_tok, off, n_expert).items():
        ex = experts[e]
        x = h32[rows, : ex.cols]
        acc = _gemm_rows(x if h_rot else _prologue(x, ex.suh), ex.what)
        y[rows, : ex.rows] = to_bf16_bits(had128(acc) * ex.svh.astype(np.float32)[None, :])
    return y
