"""Independent evaluator for Ornith policy B projections.

This module does not call the GPU quantizer, the GPU dot, or the host
dequantizer. Binary32 and binary16 rounding are defined here. M is the
consumed packed formula summed in real arithmetic and then rounded to
binary32. F is the F64 dot of the original binary32 activations with
dyadic dequantized weights, then rounded to binary32. D sequences are the
source expressions with and without contracting a multiply into a following
add. G is supplied by the caller from a production launch.

Q8_0 and Q6_K MMVQ read the Q8_1 scale only. The Q5_1 partial below is the
one call that does read the stored auxiliary sum; it is not a layer-0
projection.
"""

from __future__ import annotations

import json
import math
import struct
from fractions import Fraction
from pathlib import Path

U = Fraction(1, 2**24)
Q8_BLOCK_BYTES = 36
Q6_BLOCK_BYTES = 210
Q8_0_BLOCK_BYTES = 34
ROW_PADDING = 512


class ContractError(ValueError):
    """The input or a derived code is outside the declared contract."""


def _finite_f32(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError("activation is not a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ContractError("nonfinite activation")
    raw = struct.pack("<f", number)
    back = struct.unpack("<f", raw)[0]
    if back != number or math.copysign(1.0, back) != math.copysign(1.0, number):
        raise ContractError("activation is not a binary32 value")
    return back


def _floor_log2(ax):
    estimate = ax.numerator.bit_length() - ax.denominator.bit_length()
    while True:
        power = Fraction(2) ** estimate
        nxt = power * 2
        if power > ax:
            estimate -= 1
            continue
        if nxt <= ax:
            estimate += 1
            continue
        return estimate


def _ties_even_int(q):
    """Nearest integer to a nonnegative Fraction, ties to even."""
    quotient, remainder = divmod(q.numerator, q.denominator)
    twice = remainder * 2
    if twice < q.denominator or (twice == q.denominator and quotient % 2 == 0):
        return quotient
    return quotient + 1


def rn32(value):
    """Round a Fraction to binary32, ties to even. Returns a Python float."""
    if not isinstance(value, Fraction):
        raise ContractError("RN32 expects an exact Fraction")
    if value == 0:
        return 0.0
    negative = value < 0
    ax = -value if negative else value
    # Midpoint between max finite binary32 and the overflow binade.
    overflow_at = Fraction(2**128) - Fraction(2**104) + Fraction(2**103)
    if ax >= overflow_at:
        return math.copysign(math.inf, -1.0 if negative else 1.0)
    if ax < Fraction(2**-126):
        mantissa = _ties_even_int(ax / Fraction(2**-149))
        if mantissa == 0:
            return math.copysign(0.0, -1.0 if negative else 1.0)
        if mantissa == 2**23:
            bits = (0x00800000)
        else:
            bits = mantissa
    else:
        exponent = _floor_log2(ax)
        mantissa = _ties_even_int(ax / Fraction(2) ** (exponent - 23))
        if mantissa == 2**24:
            mantissa = 2**23
            exponent += 1
        if exponent > 127:
            return math.copysign(math.inf, -1.0 if negative else 1.0)
        bits = ((exponent + 127) << 23) | (mantissa - 2**23)
    if negative:
        bits |= 0x80000000
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def fma32(a, b, c):
    """Correctly rounded binary32 FMA of three binary32 values."""
    return rn32(Fraction(_finite_f32(a)) * Fraction(_finite_f32(b)) +
                Fraction(_finite_f32(c)))


def rn16(value):
    """Round a binary32 value to binary16, ties to even. Returns uint16 bits.

    Overflow becomes an infinite binary16 of the same sign. The input must
    already be binary32, so this is one rounding, not a double rounding of
    a wider approximate value.
    """
    number = value if isinstance(value, float) else rn32(value)
    number = _finite_f32(number) if math.isfinite(number) else number
    if math.isnan(number):
        raise ContractError("RN16 of NaN")
    if math.isinf(number):
        return 0x7C00 if number > 0 else 0xFC00
    try:
        packed = struct.pack("<e", number)
    except OverflowError:
        return 0x7C00 if number > 0 else 0xFC00
    return struct.unpack("<H", packed)[0]


def f16_to_f32(bits):
    """Promote a binary16 bit pattern. The result is an exact binary32 value."""
    return struct.unpack("<f", struct.pack("<f", struct.unpack("<e",
                         struct.pack("<H", bits & 0xFFFF))[0]))[0]


def f32_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def same_f32(left, right):
    return f32_bits(left) == f32_bits(right)


def round_half_away(value):
    """C roundf: halfway cases go away from zero. Value is binary32."""
    number = Fraction(_finite_f32(value))
    if number >= 0:
        return math.floor(number + Fraction(1, 2))
    return -math.floor(-number + Fraction(1, 2))


def padded_row(k):
    if k <= 0:
        raise ContractError("K must be positive")
    return ((k + ROW_PADDING - 1) // ROW_PADDING) * ROW_PADDING


def gamma(k):
    if k < 0 or Fraction(k) * U >= 1:
        raise ContractError("gamma degree is outside its domain")
    return (Fraction(k) * U) / (1 - Fraction(k) * U)


def dot_limit(n, absolute_sum):
    """Sufficient |D - M_real| bound from the registered protocol."""
    g = gamma(n)
    factor = g + (1 + g) * ((1 + U) ** 3 - 1)
    return factor * absolute_sum


def within_dot_bound(result, real_sum, absolute_sum, n):
    if not math.isfinite(result):
        return False
    return abs(Fraction(result) - real_sum) <= dot_limit(n, absolute_sum)


def mmvq_degree(k, weight):
    """Published addition count. Q6_K K=2048 is 13 and K=4096 is 21.

    Q8_0 visits the same number of partials per lane when K is a multiple
    of 256. Its partial has no extra binary32 group add, so this count is
    sufficient and one addition looser than that formula's path.
    """
    if k % 256 != 0:
        raise ContractError("MMVQ degree expects K multiple of 256")
    partials = k // 256
    if weight not in ("q6_k", "q8_0"):
        raise ContractError(weight)
    return 1 + (partials - 1) + 5


def of32_degree(k):
    """Lane sum of K/32 products, then five shuffle-down adds."""
    if k % 32 != 0:
        raise ContractError("o-F32 degree expects K multiple of 32")
    return (k // 32 - 1) + 5


def _u32(value):
    return value & 0xFFFFFFFF


def _sar32(value, bits):
    value = _u32(value)
    if bits == 0:
        return value
    signed = value - 0x100000000 if value & 0x80000000 else value
    return _u32(signed >> bits)


def _i8(byte):
    return byte - 256 if byte & 0x80 else byte


def _vsubss4(a, b):
    out = 0
    for lane in range(4):
        diff = _i8((a >> (8 * lane)) & 0xFF) - _i8((b >> (8 * lane)) & 0xFF)
        diff = max(-128, min(127, diff))
        out |= (diff & 0xFF) << (8 * lane)
    return out


def _dp4a(a, b):
    total = 0
    for lane in range(4):
        total += _i8((a >> (8 * lane)) & 0xFF) * _i8((b >> (8 * lane)) & 0xFF)
    return total


class Q8Block:
    """One block_q8_1: stored scale, stored auxiliary, 32 signed codes."""

    def __init__(self, scale_bits, aux_bits, codes):
        self.scale_bits = scale_bits
        self.aux_bits = aux_bits
        self.codes = list(codes)
        self.scale_f32 = f16_to_f32(scale_bits)
        self.aux_f32 = f16_to_f32(aux_bits)

    def pack(self):
        return struct.pack("<HH32b", self.scale_bits, self.aux_bits, *self.codes)


def encode_q8_1_row(values):
    """Encode one logical row. values has length K; the tail is padded to 512.

    Returns the padded block list. Codes are created with the binary32
    scale d, not with the stored binary16 scale.
    """
    logical = [_finite_f32(value) for value in values]
    width = padded_row(len(logical))
    logical.extend([0.0] * (width - len(logical)))
    blocks = []
    for start in range(0, width, 32):
        chunk = logical[start:start + 32]
        amplitude = max(abs(item) for item in chunk)
        if amplitude == 0.0:
            blocks.append(Q8Block(0, 0, [0] * 32))
            continue
        scale = rn32(Fraction(amplitude) / 127)
        codes = []
        for item in chunk:
            code = round_half_away(rn32(Fraction(item) / Fraction(scale)))
            if code < -127 or code > 127:
                raise ContractError("Q8_1 code escaped [-127, 127]")
            codes.append(code)
        sumq = sum(codes)
        aux = rn32(Fraction(scale) * sumq)
        blocks.append(Q8Block(rn16(scale), rn16(aux), codes))
    return blocks


def encode_q8_1_matrix(rows):
    return [encode_q8_1_row(row) for row in rows]


def pack_q8_1_matrix(rows):
    return b"".join(block.pack() for row in encode_q8_1_matrix(rows) for block in row)


def q6_index_fields(index):
    half = index // 128
    within = index % 128
    segment = within // 32
    lane = within % 32
    return half, segment, lane


def q6_scale_index(index):
    half, segment, lane = q6_index_fields(index)
    return half * 8 + lane // 16 + segment * 2


def q6_code_at(ql, qh, index):
    """Logical Q6_K code in [-32, 31], from the block layout."""
    half, segment, lane = q6_index_fields(index)
    ql_base = half * 64
    high_bits = qh[half * 32 + lane]
    if segment == 0:
        low = ql[ql_base + lane] & 0x0F
        high = high_bits & 0x03
    elif segment == 1:
        low = ql[ql_base + 32 + lane] & 0x0F
        high = (high_bits >> 2) & 0x03
    elif segment == 2:
        low = ql[ql_base + lane] >> 4
        high = (high_bits >> 4) & 0x03
    else:
        low = ql[ql_base + 32 + lane] >> 4
        high = (high_bits >> 6) & 0x03
    return ((high << 4) | low) - 32


def q6_lane_bytes(lane):
    """Element groups (i=0, i=1), each four indices, consumed by one MMVQ lane."""
    half = lane // 16
    base_lane = (lane % 8) * 4
    if lane % 16 < 8:
        low = [half * 128 + base_lane + t for t in range(4)]
        high = [item + 64 for item in low]
    else:
        low = [half * 128 + 32 + base_lane + t for t in range(4)]
        high = [item + 64 for item in low]
    return low, high


def q6_kernel_codes(ql, qh, lane):
    """Four-and-four codes from the MMVQ shifts, signed arithmetic included."""
    vl = int.from_bytes(ql[4 * lane:4 * lane + 4], "little")
    qh_index = 8 * (lane // 16) + (lane % 8)
    vh_shift = 2 * ((lane % 16) // 8)
    vh = _sar32(int.from_bytes(qh[4 * qh_index:4 * qh_index + 4], "little"),
                vh_shift)
    groups = []
    for i in range(2):
        vil = _sar32(vl, 4 * i) & 0x0F0F0F0F
        vih = (_sar32(vh, 4 * i) << 4) & 0x30303030
        packed = _vsubss4(vil | vih, 0x20202020)
        groups.append([_i8((packed >> (8 * t)) & 0xFF) for t in range(4)])
    return groups


def parse_q6_block(raw):
    if len(raw) != Q6_BLOCK_BYTES:
        raise ContractError("Q6_K block is 210 bytes")
    return {
        "ql": raw[:128],
        "qh": raw[128:192],
        "scales": list(struct.unpack_from("<16b", raw, 192)),
        "d_bits": struct.unpack_from("<H", raw, 208)[0],
        "d": f16_to_f32(struct.unpack_from("<H", raw, 208)[0]),
    }


def parse_q8_0_block(raw):
    if len(raw) != Q8_0_BLOCK_BYTES:
        raise ContractError("Q8_0 block is 34 bytes")
    return {
        "d_bits": struct.unpack_from("<H", raw, 0)[0],
        "d": f16_to_f32(struct.unpack_from("<H", raw, 0)[0]),
        "qs": list(struct.unpack_from("<32b", raw, 2)),
    }


def parse_q8_1_bytes(raw):
    if len(raw) % Q8_BLOCK_BYTES != 0:
        raise ContractError("Q8_1 byte length is not a whole block")
    blocks = []
    for start in range(0, len(raw), Q8_BLOCK_BYTES):
        scale, aux = struct.unpack_from("<HH", raw, start)
        codes = list(struct.unpack_from("<32b", raw, start + 4))
        blocks.append(Q8Block(scale, aux, codes))
    return blocks


def _butterfly(lanes):
    values = list(lanes)
    for offset in (16, 8, 4, 2, 1):
        values = [rn32(Fraction(values[i]) + Fraction(values[i ^ offset]))
                  for i in range(32)]
    return values[0]


def _shfl_down(lanes):
    values = list(lanes)
    for delta in (16, 8, 4, 2, 1):
        updated = values[:]
        for i in range(32 - delta):
            updated[i] = rn32(Fraction(values[i]) + Fraction(values[i + delta]))
        values = updated
    return values[0]


def _q6_sumf(block, q8_blocks, superblock, lane, mode):
    low, high = q6_lane_bytes(lane)
    groups = (low, high)
    codes = q6_kernel_codes(block["ql"], block["qh"], lane)
    scale_offset = 8 * (lane // 16) + (lane % 16) // 4
    bq8 = 4 * (lane // 16) + (lane % 16) // 8
    sumf = 0.0
    for i in range(2):
        activation = q8_blocks[superblock * 8 + bq8 + 2 * i]
        byte_index = 4 * (lane % 8)
        act = 0
        for t in range(4):
            act |= (activation.codes[byte_index + t] & 0xFF) << (8 * t)
        weight = 0
        for t in range(4):
            weight |= (codes[i][t] & 0xFF) << (8 * t)
        exact = _dp4a(weight, act) * block["scales"][scale_offset + 4 * i]
        d8 = activation.scale_f32
        if mode == "D_mul":
            term = rn32(Fraction(d8) * exact)
            sumf = rn32(Fraction(sumf) + Fraction(term))
        else:
            sumf = fma32(d8, float(exact), sumf)
    return sumf


def q6_mmvq_sequences(weight, q8_blocks, k):
    """Three declared Q6_K MMVQ sequences for one row. weight is packed bytes."""
    if k % 256 != 0 or len(q8_blocks) < k // 32:
        raise ContractError("Q6_K MMVQ shape")
    supers = k // 256
    results = {}
    for mode in ("D_mul", "D_fma_inner", "D_fma_outer"):
        lanes = [0.0] * 32
        for superblock in range(supers):
            block = parse_q6_block(weight[superblock * Q6_BLOCK_BYTES:
                                           (superblock + 1) * Q6_BLOCK_BYTES])
            for lane in range(32):
                sumf = _q6_sumf(block, q8_blocks, superblock, lane, mode)
                if mode == "D_fma_outer":
                    lanes[lane] = fma32(block["d"], sumf, lanes[lane])
                else:
                    partial = rn32(Fraction(block["d"]) * Fraction(sumf))
                    lanes[lane] = rn32(Fraction(lanes[lane]) + Fraction(partial))
        results[mode] = _butterfly(lanes)
    return results


def q8_0_mmvq_sequences(weight, q8_blocks, k):
    """Q8_0 MMVQ sequences for one row.

    D_mul rounds (d_w * d_a) * dot and then the add. D_fma_inner matches
    D_mul: the partial has no inner addition to contract. D_fma_outer
    contracts that left-associated product into the accumulator,
    acc = fma(RN(d_w * d_a), dot, acc).

    D_compiled is the association written beside mul_mat_vec_q8_ahead: the
    inlined expression compiles as p = d_w * dot, then acc = fma(p, d_a, acc).
    That reassociates the source. It is reported separately from the three
    left-associated sequences. Butterfly adds stay separate in every sequence.
    """
    if k % 32 != 0 or len(q8_blocks) < k // 32:
        raise ContractError("Q8_0 MMVQ shape")
    blocks = k // 32
    results = {name: [0.0] * 32 for name in
               ("D_mul", "D_fma_inner", "D_fma_outer", "D_compiled")}
    for lane in range(32):
        kqs = 2 * (lane % 4)
        kbx = lane // 4
        while kbx < blocks:
            wb = parse_q8_0_block(weight[kbx * Q8_0_BLOCK_BYTES:
                                          (kbx + 1) * Q8_0_BLOCK_BYTES])
            act = q8_blocks[kbx]
            total = 0
            for i in range(2):
                for t in range(4):
                    element = 4 * (kqs + i) + t
                    total += wb["qs"][element] * act.codes[element]
            dw, da = wb["d"], act.scale_f32
            scale = rn32(Fraction(dw) * Fraction(da))
            partial = rn32(Fraction(scale) * total)
            for name in ("D_mul", "D_fma_inner"):
                results[name][lane] = rn32(Fraction(results[name][lane]) +
                                           Fraction(partial))
            results["D_fma_outer"][lane] = fma32(scale, float(total),
                                                 results["D_fma_outer"][lane])
            weight_dot = rn32(Fraction(dw) * total)
            results["D_compiled"][lane] = fma32(weight_dot, da,
                                                results["D_compiled"][lane])
            kbx += 8
    return {name: _butterfly(values) for name, values in results.items()}


def _of32_q6_lane(block, x_block, lane, mode, acc):
    for j in range(lane, 256, 32):
        code = q6_code_at(block["ql"], block["qh"], j)
        scale = block["scales"][j // 16]
        weight = rn32(Fraction(rn32(Fraction(block["d"]) * scale)) * code)
        if mode == "D_mul":
            acc = rn32(Fraction(acc) + Fraction(rn32(Fraction(weight) *
                                                      Fraction(x_block[j]))))
        else:
            acc = fma32(weight, x_block[j], acc)
    return acc


def of32_sequences(weight, values, k, weight_format):
    """o-F32 dense sequences. D_fma contracts only sum += weight * x."""
    values = [_finite_f32(item) for item in values]
    if len(values) != k:
        raise ContractError("o-F32 activation length")
    results = {}
    for mode in ("D_mul", "D_fma"):
        lanes = [0.0] * 32
        if weight_format == "q6_k":
            if k % 256 != 0:
                raise ContractError("o-F32 Q6_K K")
            for b in range(k // 256):
                block = parse_q6_block(weight[b * Q6_BLOCK_BYTES:
                                               (b + 1) * Q6_BLOCK_BYTES])
                chunk = values[b * 256:(b + 1) * 256]
                for lane in range(32):
                    lanes[lane] = _of32_q6_lane(block, chunk, lane, mode,
                                                lanes[lane])
        elif weight_format == "q8_0":
            if k % 32 != 0:
                raise ContractError("o-F32 Q8_0 K")
            for b in range(k // 32):
                block = parse_q8_0_block(weight[b * Q8_0_BLOCK_BYTES:
                                                 (b + 1) * Q8_0_BLOCK_BYTES])
                chunk = values[b * 32:(b + 1) * 32]
                for lane in range(32):
                    weight_value = rn32(Fraction(block["d"]) * block["qs"][lane])
                    if mode == "D_mul":
                        lanes[lane] = rn32(
                            Fraction(lanes[lane]) +
                            Fraction(rn32(Fraction(weight_value) *
                                           Fraction(chunk[lane]))))
                    else:
                        lanes[lane] = fma32(weight_value, chunk[lane],
                                            lanes[lane])
        else:
            raise ContractError(weight_format)
        results[mode] = _shfl_down(lanes)
    return results


def q6_consumed_terms(weight, q8_blocks, k):
    """Exact MMVQ products d6 * stored_d8 * scale * q6 * q8."""
    terms = []
    for b in range(k // 256):
        block = parse_q6_block(weight[b * Q6_BLOCK_BYTES:(b + 1) * Q6_BLOCK_BYTES])
        for index in range(256):
            code = q6_code_at(block["ql"], block["qh"], index)
            scale = block["scales"][q6_scale_index(index)]
            act = q8_blocks[(b * 256 + index) // 32]
            q8 = act.codes[index % 32]
            terms.append(Fraction(block["d"]) * Fraction(act.scale_f32) *
                         scale * code * q8)
    return terms


def q8_0_consumed_terms(weight, q8_blocks, k):
    terms = []
    for b in range(k // 32):
        block = parse_q8_0_block(weight[b * Q8_0_BLOCK_BYTES:
                                         (b + 1) * Q8_0_BLOCK_BYTES])
        act = q8_blocks[b]
        for index in range(32):
            terms.append(Fraction(block["d"]) * Fraction(act.scale_f32) *
                         block["qs"][index] * act.codes[index])
    return terms


def q6_mathematical_terms(weight, values, k):
    values = [_finite_f32(item) for item in values]
    terms = []
    for b in range(k // 256):
        block = parse_q6_block(weight[b * Q6_BLOCK_BYTES:(b + 1) * Q6_BLOCK_BYTES])
        for index in range(256):
            code = q6_code_at(block["ql"], block["qh"], index)
            scale = block["scales"][q6_scale_index(index)]
            terms.append(Fraction(block["d"]) * scale * code *
                         Fraction(values[b * 256 + index]))
    return terms


def q8_0_mathematical_terms(weight, values, k):
    values = [_finite_f32(item) for item in values]
    terms = []
    for b in range(k // 32):
        block = parse_q8_0_block(weight[b * Q8_0_BLOCK_BYTES:
                                         (b + 1) * Q8_0_BLOCK_BYTES])
        for index in range(32):
            terms.append(Fraction(block["d"]) * block["qs"][index] *
                         Fraction(values[b * 32 + index]))
    return terms


def reference_pair(terms):
    """Return M, F, P from exact consumed or mathematical terms."""
    real = sum(terms, Fraction(0))
    absolute = sum((abs(term) for term in terms), Fraction(0))
    f64_values = [float(term) for term in terms]
    f64_sum = math.fsum(f64_values)
    return rn32(real), rn32(Fraction(f64_sum)), absolute, real


def q5_1_partial(sumi, d_bits, m_bits, scale_bits, aux_bits, vdr=2):
    """One Q5_1 MMVQ call under FAST_FP16: half products, then the min term.

    The divisor QI5_1/vdr = 2 spreads the auxiliary correction across the
    lanes that each add it. A reconstructed-activation dot does not read
    aux_bits. This is the consumed partial, not a full-row reduction.
    """
    d5d8 = f16_to_f32(rn16(rn32(Fraction(f16_to_f32(d_bits)) *
                                Fraction(f16_to_f32(scale_bits)))))
    m5s8 = f16_to_f32(rn16(rn32(Fraction(f16_to_f32(m_bits)) *
                                Fraction(f16_to_f32(aux_bits)))))
    divisor = 4 // vdr
    scaled = rn32(Fraction(sumi) * Fraction(d5d8))
    correction = rn32(Fraction(m5s8) / divisor)
    return rn32(Fraction(scaled) + Fraction(correction))


def expert_rank(ids, slot):
    """Rank of slot when ordered by expert id, then by source slot.

    This is the host formula of quantize_q8_1_by_expert. Ornith projections
    do not take that kernel: it requires K == 640, one expert slot and
    Q5_1 or Q8_0.
    """
    expert = ids[slot]
    return sum(1 for other_slot, other in enumerate(ids)
               if other < expert or (other == expert and other_slot < slot))


def relative_rms(reference, other):
    """Euclidean relative RMS. Both zero is 0. A zero reference with a
    nonzero other is infinite, and the absolute RMS is still returned.
    """
    if len(reference) != len(other):
        raise ContractError("RMS length")
    ref_sq = 0.0
    err_sq = 0.0
    for left, right in zip(reference, other):
        if not math.isfinite(left) or not math.isfinite(right):
            raise ContractError("nonfinite RMS input")
        ref_sq += float(left) * float(left)
        diff = float(left) - float(right)
        err_sq += diff * diff
    absolute = math.sqrt(err_sq)
    if ref_sq == 0.0:
        return (0.0 if err_sq == 0.0 else math.inf), absolute
    return math.sqrt(err_sq / ref_sq), absolute


def classify_sequences(gpu, sequences):
    return [name for name in sequences if same_f32(gpu, sequences[name])]


def _float_token(value):
    if isinstance(value, Fraction):
        value = float(value)
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "-inf" if value < 0 else "inf"
    return {"bits": f"{f32_bits(value):08x}", "hex": float(value).hex()}


def score_projection(name, gpu, sequences, m_value, f_value, absolute, real,
                     degree):
    matches = [] if gpu is None else classify_sequences(gpu, sequences)
    bound_ok = {key: within_dot_bound(value, real, absolute, degree)
                for key, value in sequences.items()}
    return {
        "name": name,
        "matched": matches,
        "unexplained": gpu is not None and not matches,
        "bound_ok": all(bound_ok.values()),
        "bounds": bound_ok,
        "degree": degree,
        "G": None if gpu is None else _float_token(gpu),
        "M": _float_token(m_value),
        "F": _float_token(f_value),
        "D": {key: _float_token(value) for key, value in sequences.items()},
        "G_minus_M": None if gpu is None else _float_token(
            rn32(Fraction(gpu) - Fraction(m_value))),
        "M_minus_F": _float_token(rn32(Fraction(m_value) - Fraction(f_value))),
        "P_abs_hex": float(absolute).hex(),
    }


def find_auxiliary_stage_witness():
    """A scale where RN16(RN32(d*sumq)) differs from RN16(RN32(RN16(d)*sumq))."""
    for magnitude in (1, 3, 127, 255, 1000):
        scale = rn32(Fraction(float(magnitude)) / 127)
        stored = f16_to_f32(rn16(scale))
        for sumq in range(1, 65):
            direct = rn16(rn32(Fraction(scale) * sumq))
            via_stored = rn16(rn32(Fraction(stored) * sumq))
            if direct != via_stored:
                return {"magnitude": magnitude, "sumq": sumq,
                        "direct": direct, "via_stored_scale": via_stored}
    return None


# Sequences named by the P4e protocol before any GPU scoring. D_compiled is a
# later, source-derived reassociation. It can explain a row. It is not one of
# these sequences, and a compiled-only match is not a protocol pass.
PROTOCOL_SEQUENCES = {"D_mul", "D_fma_inner", "D_fma_outer", "D_fma", "encoder"}


def row_class(row):
    """protocol, supplemental, or unexplained.

    supplemental means the row matches D_compiled and no protocol sequence.
    """
    matched = set(row.get("matched") or [])
    if row.get("kind") == "q8_1":
        return "unexplained" if row.get("unexplained") else "protocol"
    if row.get("unexplained") or not row.get("bound_ok", True):
        return "unexplained"
    if matched & PROTOCOL_SEQUENCES:
        return "protocol"
    if "D_compiled" in matched:
        return "supplemental"
    return "unexplained"


def score_dump(directory):
    """Score a production-launcher dump. Missing files raise.

    original_protocol_pass is true only when every row matches a sequence
    named in the registered protocol. supplemental_explained counts rows
    that match only the later D_compiled association.
    """
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text())
    rows = []
    for case in manifest["cases"]:
        rows.extend(_score_case(root, case))
    classes = [row_class(row) for row in rows]
    return {"cases": rows,
            "original_protocol_pass": all(kind == "protocol" for kind in classes),
            "supplemental_explained": sum(kind == "supplemental" for kind in classes),
            "unexplained": sum(kind == "unexplained" for kind in classes)}


def _read(root, name):
    return (root / name).read_bytes()


def _f32s(raw):
    if len(raw) % 4 != 0:
        raise ContractError("float file length")
    return list(struct.unpack("<" + "f" * (len(raw) // 4), raw))


def _score_case(root, case):
    kind = case["kind"]
    if kind == "q8_1":
        k = case["k"]
        rows = case["rows"]
        values = _f32s(_read(root, case["x"]))
        got = parse_q8_1_bytes(_read(root, case["q8"]))
        expected = []
        for row in range(rows):
            expected.extend(encode_q8_1_row(values[row * k:(row + 1) * k]))
        if len(got) != len(expected):
            raise ContractError("Q8_1 dump shape")
        mismatches = []
        for index, (left, right) in enumerate(zip(got, expected)):
            if left.pack() != right.pack():
                mismatches.append(index)
        return [{"name": case["name"], "kind": "q8_1",
                 "mismatched_blocks": mismatches,
                 "unexplained": bool(mismatches), "bound_ok": True,
                 "matched": [] if mismatches else ["encoder"]}]
    if kind == "mmvq":
        return _score_mmvq(root, case)
    if kind == "of32":
        return _score_of32(root, case)
    raise ContractError(kind)


def _score_mmvq(root, case):
    k = case["k"]
    rows = case["rows"]
    tokens = case["tokens"]
    used = case["used"]
    ids = case["ids"]
    values = _f32s(_read(root, case["x"]))
    weight = _read(root, case["weights"])
    gpu = _f32s(_read(root, case["y"]))
    fmt = case["format"]
    block = 256 if fmt == "q6_k" else 32
    bytes_per = Q6_BLOCK_BYTES if fmt == "q6_k" else Q8_0_BLOCK_BYTES
    degree = mmvq_degree(k, fmt) if k % 256 == 0 else of32_degree(k)
    # Q8_0 with K not a multiple of 256 still has a whole number of partials
    # per lane when K/32 is a multiple of the stride 8, i.e. K multiple of 256.
    # Smaller fixtures use the lane schedule's actual partial count.
    if fmt == "q8_0" and k % 256 != 0:
        degree = _q8_small_degree(k)
    scored = []
    for token in range(tokens):
        q8 = encode_q8_1_row(values[token * k:(token + 1) * k])
        # The dot reads K elements. Drop the padding blocks from the product
        # list; they are checked by the encoder cases.
        q8 = q8[:k // 32]
        for slot in range(used):
            expert = ids[token * used + slot]
            base = (expert * rows) * (k // block) * bytes_per
            for row in range(rows):
                packed = weight[base + row * (k // block) * bytes_per:
                                base + (row + 1) * (k // block) * bytes_per]
                if fmt == "q6_k":
                    sequences = q6_mmvq_sequences(packed, q8, k)
                    terms = q6_consumed_terms(packed, q8, k)
                    math_terms = q6_mathematical_terms(
                        packed, values[token * k:(token + 1) * k], k)
                else:
                    sequences = q8_0_mmvq_sequences(packed, q8, k)
                    terms = q8_0_consumed_terms(packed, q8, k)
                    math_terms = q8_0_mathematical_terms(
                        packed, values[token * k:(token + 1) * k], k)
                m_value, _, absolute, real = reference_pair(terms)
                _, f_value, _, _ = reference_pair(math_terms)
                index = (token * used + slot) * rows + row
                scored.append(score_projection(
                    f"{case['name']}/t{token}/s{slot}/r{row}", gpu[index],
                    sequences, m_value, f_value, absolute, real, degree))
    return scored


def _q8_small_degree(k):
    blocks = k // 32
    partials = 0
    for lane in range(32):
        count = 0
        kbx = lane // 4
        while kbx < blocks:
            count += 1
            kbx += 8
        partials = max(partials, count)
    return 1 + max(partials - 1, 0) + 5


def _score_of32(root, case):
    k = case["k"]
    rows = case["rows"]
    values = _f32s(_read(root, case["x"]))
    weight = _read(root, case["weights"])
    gpu = _f32s(_read(root, case["y"]))
    fmt = case["format"]
    block = 256 if fmt == "q6_k" else 32
    bytes_per = Q6_BLOCK_BYTES if fmt == "q6_k" else Q8_0_BLOCK_BYTES
    degree = of32_degree(k)
    scored = []
    for row in range(rows):
        packed = weight[row * (k // block) * bytes_per:
                        (row + 1) * (k // block) * bytes_per]
        sequences = of32_sequences(packed, values, k, fmt)
        if fmt == "q6_k":
            terms = q6_mathematical_terms(packed, values, k)
        else:
            terms = q8_0_mathematical_terms(packed, values, k)
        m_value, f_value, absolute, real = reference_pair(terms)
        scored.append(score_projection(
            f"{case['name']}/r{row}", gpu[row], sequences, m_value, f_value,
            absolute, real, degree))
    return scored


def encode_check(directory):
    """Compare saved layer-0 norm rows with GPU Q8_1 bytes.

    Codes, stored scale and auxiliary are counted separately. Exit status is
    1 when any of the three disagree. A missing pair is an error, not a pass.
    """
    root = Path(directory)
    tokens = []
    for path in sorted(root.glob("t*_norm.f32")):
        token = int(path.name[1:].split("_", 1)[0])
        qpath = root / f"t{token}_q8.bin"
        if not qpath.is_file():
            raise ContractError("missing " + qpath.name)
        values = list(struct.unpack("<" + "f" * (path.stat().st_size // 4),
                                    path.read_bytes()))
        expected = encode_q8_1_row(values)
        got = parse_q8_1_bytes(qpath.read_bytes())
        if len(got) < len(expected):
            raise ContractError("short Q8 capture")
        code_bad = scale_bad = aux_bad = 0
        for left, right in zip(expected, got):
            if left.codes != right.codes:
                code_bad += 1
            if left.scale_bits != right.scale_bits:
                scale_bad += 1
            if left.aux_bits != right.aux_bits:
                aux_bad += 1
        tokens.append({"token": token, "code_bad": code_bad,
                       "scale_bad": scale_bad, "aux_bad": aux_bad})
    if not tokens:
        raise ContractError("no norm rows to check")
    summary = {
        "tokens": len(tokens),
        "code_bad": sum(row["code_bad"] for row in tokens),
        "scale_bad": sum(row["scale_bad"] for row in tokens),
        "aux_bad": sum(row["aux_bad"] for row in tokens),
        "rows": tokens,
    }
    (root / "q8-encode-check.json").write_text(json.dumps(summary, indent=2))
    lines = ["token\tcode_bad\tscale_bad\taux_bad"]
    for row in tokens:
        lines.append(f"{row['token']}\t{row['code_bad']}\t{row['scale_bad']}\t{row['aux_bad']}")
    (root / "q8-columns.tsv").write_text("\n".join(lines) + "\n")
    print(json.dumps({key: summary[key] for key in
                      ("tokens", "code_bad", "scale_bad", "aux_bad")},
                     sort_keys=True))
    return 0 if summary["code_bad"] == summary["scale_bad"] == summary["aux_bad"] == 0 else 1


def main(argv):
    if len(argv) == 3 and argv[1] == "score":
        report = score_dump(argv[2])
        print(json.dumps({
            "rows": len(report["cases"]),
            "original_protocol_pass": report["original_protocol_pass"],
            "supplemental_explained": report["supplemental_explained"],
            "unexplained": report["unexplained"],
        }, sort_keys=True))
        Path(argv[2], "score.json").write_text(json.dumps(report, indent=2,
                                                          sort_keys=True))
        return 0 if report["original_protocol_pass"] else 1
    if len(argv) == 3 and argv[1] == "encode-check":
        return encode_check(argv[2])
    raise SystemExit(
        "usage: representation.py score DUMP_DIR | encode-check DUMP_DIR")


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv))
