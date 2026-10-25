"""Host checks for the policy-B representation evaluator. No model or device."""

import math
import random
import struct
import sys
import unittest
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools" / "ornith15"))
import representation as rep


def f32(value):
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def pack_q6(codes, scales, d_bits):
    ql = bytearray(128)
    qh = bytearray(64)
    for index, code in enumerate(codes):
        raw = code + 32
        low, high = raw & 0x0F, (raw >> 4) & 0x03
        half, segment, lane = rep.q6_index_fields(index)
        base = half * 64
        qh_index = half * 32 + lane
        if segment == 0:
            ql[base + lane] |= low
            qh[qh_index] |= high
        elif segment == 1:
            ql[base + 32 + lane] |= low
            qh[qh_index] |= high << 2
        elif segment == 2:
            ql[base + lane] |= low << 4
            qh[qh_index] |= high << 4
        else:
            ql[base + 32 + lane] |= low << 4
            qh[qh_index] |= high << 6
    return bytes(ql) + bytes(qh) + struct.pack("<16bH", *scales, d_bits)


def pack_q8_0(codes, d_bits):
    return struct.pack("<H32b", d_bits, *codes)


class RoundingTest(unittest.TestCase):
    def test_rn32_ties_and_range(self):
        self.assertEqual(rep.f32_bits(rep.rn32(Fraction(1))),
                         rep.f32_bits(1.0))
        # Halfway between 1 and the next binary32, tie to even.
        self.assertEqual(rep.f32_bits(rep.rn32(Fraction(1) + Fraction(1, 2**24))),
                         rep.f32_bits(1.0))
        up = rep.rn32(Fraction(1) + Fraction(1, 2**24) + Fraction(1, 2**40))
        next_f32 = struct.unpack("<f", struct.pack("<I", 0x3F800001))[0]
        self.assertEqual(rep.f32_bits(up), rep.f32_bits(next_f32))
        # Smallest subnormal is exact. Half of it ties to zero.
        self.assertEqual(rep.f32_bits(rep.rn32(Fraction(1, 2**149))),
                         0x00000001)
        self.assertEqual(rep.f32_bits(rep.rn32(Fraction(1, 2**150))), 0)
        self.assertTrue(math.isinf(rep.rn32(Fraction(2**128))))
        self.assertEqual(rep.rn32(Fraction(-1, 2)), -0.5)

    def test_rn16_known_bits(self):
        self.assertEqual(rep.rn16(1.0), 0x3C00)
        self.assertEqual(rep.rn16(-0.0), 0x8000)
        midpoint = 1.0 + 2**-11
        self.assertEqual(rep.rn16(midpoint), 0x3C00)
        above = struct.unpack("<f", struct.pack("<I", 0x3F801001))[0]
        self.assertEqual(rep.rn16(above), 0x3C01)
        self.assertEqual(rep.rn16(65504.0), 0x7BFF)
        self.assertEqual(rep.rn16(65520.0), 0x7C00)
        self.assertEqual(rep.rn16(2**-25), 0)
        self.assertEqual(rep.f16_to_f32(0x3C00), 1.0)

    def test_round_half_away(self):
        for value, expected in ((0.5, 1), (-0.5, -1), (1.5, 2), (-1.5, -2),
                                (2.0, 2), (-0.0, 0)):
            self.assertEqual(rep.round_half_away(f32(value)), expected)


class Q8EncoderTest(unittest.TestCase):
    def test_zero_block_is_positive_zero(self):
        blocks = rep.encode_q8_1_row([0.0] * 32)
        self.assertEqual(len(blocks), 512 // 32)
        for block in blocks:
            self.assertEqual(block.codes, [0] * 32)
            self.assertEqual(block.scale_bits, 0)
            self.assertEqual(block.aux_bits, 0)
            self.assertEqual(block.pack()[:4], b"\x00\x00\x00\x00")

    def test_half_away_when_scale_is_one(self):
        values = [0.0] * 32
        values[0] = 127.0
        values[1] = 0.5
        values[2] = -0.5
        values[3] = 1.5
        values[4] = -1.5
        block = rep.encode_q8_1_row(values)[0]
        self.assertEqual(block.codes[0], 127)
        self.assertEqual(block.codes[1:5], [1, -1, 2, -2])
        self.assertEqual(rep.f16_to_f32(block.scale_bits), 1.0)

    def test_scale_tie_uses_binary16_even(self):
        # d = 1 + 2^-11 is the binary16 midpoint above 1.
        amplitude = rep.rn32((Fraction(1) + Fraction(1, 2**11)) * 127)
        values = [0.0] * 32
        values[0] = amplitude
        values[1] = -amplitude
        block = rep.encode_q8_1_row(values)[0]
        self.assertEqual(block.scale_bits, rep.rn16(1.0 + 2**-11))
        self.assertEqual(block.codes[0], 127)
        self.assertEqual(block.codes[1], -127)

    def test_auxiliary_keeps_the_pre_half_scale(self):
        witness = rep.find_auxiliary_stage_witness()
        self.assertIsNotNone(witness)
        self.assertNotEqual(witness["direct"], witness["via_stored_scale"])

    def test_tail_participates_and_padding_is_zero(self):
        short = [1.0] * 40
        encoded = rep.encode_q8_1_row(short)
        self.assertEqual(len(encoded), 512 // 32)
        explicit = rep.encode_q8_1_row(short + [0.0] * (512 - 40))
        self.assertEqual([block.pack() for block in encoded],
                         [block.pack() for block in explicit])
        for block in encoded[2:]:
            self.assertEqual(block.pack(), rep.Q8Block(0, 0, [0] * 32).pack())
        # The second block mixes eight real ones with a zero tail.
        self.assertEqual(encoded[1].codes[8:], [0] * 24)
        wide = rep.encode_q8_1_row([0.25] * 544)
        self.assertEqual(len(wide), 1024 // 32)
        for block in wide[544 // 32:]:
            self.assertEqual(block.codes, [0] * 32)
            self.assertEqual(block.scale_bits, 0)

    def test_rows_are_independent(self):
        rows = [[0.0] * 32, [3.0] + [0.0] * 31, [-2.0, 2.0] + [0.0] * 30]
        matrix = rep.encode_q8_1_matrix(rows)
        for row, alone in zip(matrix, rows):
            self.assertEqual([block.pack() for block in row],
                             [block.pack() for block in rep.encode_q8_1_row(alone)])

    def test_cancellation_and_rejection(self):
        values = [5.0, -5.0] + [0.0] * 30
        block = rep.encode_q8_1_row(values)[0]
        self.assertEqual(sum(block.codes), 0)
        self.assertEqual(block.aux_f32, 0.0)
        with self.assertRaises(rep.ContractError):
            rep.encode_q8_1_row([math.nan] + [0.0] * 31)
        with self.assertRaises(rep.ContractError):
            rep.encode_q8_1_row([math.inf] + [0.0] * 31)

    def test_codes_stay_in_contract_range(self):
        rng = random.Random(15)
        for _ in range(20):
            values = [f32(rng.uniform(-1e4, 1e4)) for _ in range(32)]
            for code in rep.encode_q8_1_row(values)[0].codes:
                self.assertGreaterEqual(code, -127)
                self.assertLessEqual(code, 127)


class LayoutAndDotTest(unittest.TestCase):
    def test_kernel_groups_match_q6_layout(self):
        rng = random.Random(6)
        for _ in range(30):
            codes = [rng.randrange(-32, 32) for _ in range(256)]
            raw = pack_q6(codes, [1] * 16, 0x3C00)
            block = rep.parse_q6_block(raw)
            seen = [None] * 256
            for lane in range(32):
                groups = rep.q6_kernel_codes(block["ql"], block["qh"], lane)
                elements = rep.q6_lane_bytes(lane)
                for group, idxs in zip(groups, elements):
                    for code, index in zip(group, idxs):
                        self.assertIsNone(seen[index])
                        seen[index] = code
                        self.assertEqual(code, rep.q6_code_at(block["ql"],
                                                              block["qh"], index))
            self.assertEqual(seen, codes)
        # Adversarial high bits, where a signed shift could disagree.
        distinct = pack_q6([0] * 256, list(range(-8, 8)), 0x3C00)
        block = rep.parse_q6_block(distinct)
        for lane in range(32):
            scale_offset = 8 * (lane // 16) + (lane % 16) // 4
            for i, idxs in enumerate(rep.q6_lane_bytes(lane)):
                for index in idxs:
                    self.assertEqual(block["scales"][scale_offset + 4 * i],
                                     block["scales"][rep.q6_scale_index(index)])
        raw = pack_q6([31] * 256, [-128, 127] * 8, 0x3C00)
        block = rep.parse_q6_block(raw)
        for lane in range(32):
            groups = rep.q6_kernel_codes(block["ql"], block["qh"], lane)
            for group, idxs in zip(groups, rep.q6_lane_bytes(lane)):
                for code, index in zip(group, idxs):
                    self.assertEqual(code, 31)

    def test_q6_sequences_partition_and_respect_bound(self):
        rng = random.Random(11)
        codes = [rng.randrange(-32, 32) for _ in range(256)]
        scales = [rng.randrange(-20, 21) for _ in range(16)]
        weight = pack_q6(codes, scales, rep.rn16(f32(0.015625)))
        values = [f32(rng.uniform(-1.0, 1.0)) for _ in range(256)]
        values[10] = f32(-values[3])
        q8 = rep.encode_q8_1_row(values)[:8]
        sequences = rep.q6_mmvq_sequences(weight, q8, 256)
        terms = rep.q6_consumed_terms(weight, q8, 256)
        self.assertEqual(len(terms), 256)
        m_value, _, absolute, real = rep.reference_pair(terms)
        self.assertEqual(rep.mmvq_degree(2048, "q6_k"), 13)
        self.assertEqual(rep.mmvq_degree(4096, "q6_k"), 21)
        for value in sequences.values():
            self.assertTrue(rep.within_dot_bound(value, real, absolute, 6))
        # One superblock fused into a zero accumulator does not separate the
        # outer FMA. Two awkward superblocks do; seed 1 separates on trial 2.
        rng = random.Random(1)
        separated = False
        for _ in range(6):
            chunks = []
            for _super in range(2):
                codes = [rng.randrange(-32, 32) for _ in range(256)]
                scales = [rng.randrange(-127, 128) for _ in range(16)]
                chunks.append(pack_q6(codes, scales,
                                      rep.rn16(f32(rng.uniform(0.01, 2)))))
            values = [f32(rng.uniform(-4, 4)) for _ in range(512)]
            q8 = rep.encode_q8_1_row(values)[:16]
            weight = b"".join(chunks)
            split = rep.q6_mmvq_sequences(weight, q8, 512)
            terms = rep.q6_consumed_terms(weight, q8, 512)
            real = sum(terms, Fraction(0))
            absolute = sum((abs(term) for term in terms), Fraction(0))
            for value in split.values():
                self.assertTrue(rep.within_dot_bound(
                    value, real, absolute, rep.mmvq_degree(512, "q6_k")))
            if len({rep.f32_bits(item) for item in split.values()}) == 3:
                separated = True
                break
        self.assertTrue(separated)

    def test_q8_sequences_and_duplicate_slots(self):
        weight = b"".join(pack_q8_0([(i % 17) - 8 for _ in range(32)],
                                    rep.rn16(0.125))
                          for i in range(4))
        values = [f32((i % 9) - 4) for i in range(32)]
        q8 = rep.encode_q8_1_row(values)[:1]
        sequences = rep.q8_0_mmvq_sequences(weight[:34], q8, 32)
        terms = rep.q8_0_consumed_terms(weight[:34], q8, 32)
        m_value, f_value, absolute, real = rep.reference_pair(terms)
        for value in sequences.values():
            self.assertTrue(rep.within_dot_bound(
                value, real, absolute, rep._q8_small_degree(32)))
        # Two slots that name the same expert see the same packed row.
        self.assertEqual(sequences, rep.q8_0_mmvq_sequences(weight[:34], q8, 32))
        math_terms = rep.q8_0_mathematical_terms(weight[:34], values, 32)
        _, mathematical, _, _ = rep.reference_pair(math_terms)
        # Representation cost is the Q8_1 substitution, so M and F may differ.
        self.assertTrue(math.isfinite(m_value) and math.isfinite(mathematical))
        self.assertTrue(math.isfinite(f_value))

    def test_of32_bound_and_shuffle_order(self):
        codes = [(index % 7) - 3 for index in range(256)]
        weight = pack_q6(codes, [1, -1] * 8, rep.rn16(0.5))
        values = [f32(0.25 if index % 2 == 0 else -0.5) for index in range(256)]
        sequences = rep.of32_sequences(weight, values, 256, "q6_k")
        terms = rep.q6_mathematical_terms(weight, values, 256)
        m_value, f_value, absolute, real = rep.reference_pair(terms)
        for value in sequences.values():
            self.assertTrue(rep.within_dot_bound(
                value, real, absolute, rep.of32_degree(256)))
        q8 = pack_q8_0([index - 16 for index in range(32)], rep.rn16(0.25))
        sequences = rep.of32_sequences(q8, values[:32], 32, "q8_0")
        terms = rep.q8_0_mathematical_terms(q8, values[:32], 32)
        m_value, f_value, absolute, real = rep.reference_pair(terms)
        for value in sequences.values():
            self.assertTrue(rep.within_dot_bound(
                value, real, absolute, rep.of32_degree(32)))
        self.assertTrue(math.isfinite(f_value))

    def test_long_q6_degree_holds(self):
        rng = random.Random(2048)
        chunks = []
        for _ in range(8):
            codes = [rng.randrange(-8, 9) for _ in range(256)]
            scales = [rng.randrange(-3, 4) for _ in range(16)]
            chunks.append(pack_q6(codes, scales, rep.rn16(f32(0.03125))))
        weight = b"".join(chunks)
        values = [f32(rng.uniform(-0.5, 0.5)) for _ in range(2048)]
        q8 = rep.encode_q8_1_row(values)[:64]
        sequences = rep.q6_mmvq_sequences(weight, q8, 2048)
        real = sum(rep.q6_consumed_terms(weight, q8, 2048), Fraction(0))
        absolute = sum((abs(term) for term in rep.q6_consumed_terms(weight, q8, 2048)),
                       Fraction(0))
        for value in sequences.values():
            self.assertTrue(rep.within_dot_bound(value, real, absolute, 13))

    def test_auxiliary_partial_differs_from_reconstructed_sum(self):
        values = [f32(i - 8) for i in range(32)]
        block = rep.encode_q8_1_row(values)[0]
        sumi = sum(block.codes)
        d_bits = rep.rn16(1.5)
        m_bits = rep.rn16(-0.25)
        with_aux = rep.q5_1_partial(sumi, d_bits, m_bits, block.scale_bits,
                                    block.aux_bits)
        replaced = rep.q5_1_partial(sumi, d_bits, m_bits, block.scale_bits,
                                    rep.rn16(0.0))
        self.assertFalse(rep.same_f32(with_aux, replaced))
        self.assertNotEqual(block.aux_bits, 0)

    def test_expert_rank_orders_duplicates_by_slot(self):
        ids = [1, 1, 0, 1]
        self.assertEqual([rep.expert_rank(ids, slot) for slot in range(4)],
                         [1, 2, 0, 3])
        self.assertFalse(rep.expert_rank.__doc__.find("K == 640") < 0)

    def test_relative_rms_zero_rules(self):
        self.assertEqual(rep.relative_rms([0.0, 0.0], [0.0, -0.0]), (0.0, 0.0))
        relative, absolute = rep.relative_rms([0.0, 0.0], [3.0, 4.0])
        self.assertTrue(math.isinf(relative))
        self.assertAlmostEqual(absolute, 5.0)
        with self.assertRaises(rep.ContractError):
            rep.relative_rms([math.nan], [0.0])


class ProtocolScoreTest(unittest.TestCase):
    def test_compiled_match_is_not_a_protocol_pass(self):
        protocol = {"name": "q", "matched": ["D_mul"], "bound_ok": True,
                    "unexplained": False}
        compiled = {"name": "q", "matched": ["D_compiled"], "bound_ok": True,
                    "unexplained": False}
        self.assertEqual(rep.row_class(protocol), "protocol")
        self.assertEqual(rep.row_class(compiled), "supplemental")
        self.assertFalse(rep.row_class(
            {"name": "q", "matched": [], "bound_ok": True, "unexplained": True})
            == "protocol")

    def test_compiled_q8_formula_is_hand_evaluated(self):
        # One Q8_0 block, constant codes. Each of the four active lanes dots
        # eight 3*2 products. D_compiled is p = RN(d_w * dot), then
        # acc = fma(p, d_a, 0), then the warp butterfly. This expected value
        # is written out here; it is not taken from a GPU output.
        weight = pack_q8_0([3] * 32, rep.rn16(1.5))
        activation = rep.Q8Block(rep.rn16(0.5), 0, [2] * 32)
        sequences = rep.q8_0_mmvq_sequences(weight, [activation], 32)
        dot = 8 * 3 * 2
        from fractions import Fraction
        partial = rep.fma32(
            rep.rn32(Fraction(rep.f16_to_f32(rep.rn16(1.5))) * dot),
            rep.f16_to_f32(rep.rn16(0.5)), 0.0)
        lanes = [0.0] * 32
        for lane in range(4):
            lanes[lane] = partial
        for offset in (16, 8, 4, 2, 1):
            lanes = [rep.rn32(Fraction(lanes[i]) + Fraction(lanes[i ^ offset]))
                     for i in range(32)]
        self.assertTrue(rep.same_f32(sequences["D_compiled"], lanes[0]))

    def test_pairwise_sum_needs_both_allowances(self):
        # Four binary32 terms. The one-summation gamma_3 bound does not cover
        # the gap between two association orders. Twice that allowance does:
        # each order is within gamma_3 of the exact sum.
        from fractions import Fraction

        def add(a, b):
            return f32(a + b)

        gamma3 = float((3 * Fraction(1, 2**24)) / (1 - 3 * Fraction(1, 2**24)))
        # A fixed seed is searched only to find some witness that the
        # one-summation allowance is not a pairwise bound. The allowance used
        # below is 2 * gamma_3, from the triangle inequality, not from the
        # size of the witness.
        rng = random.Random(1)
        witness = None
        for _ in range(100000):
            sample = tuple(f32(rng.uniform(-2, 2)) for _ in range(4))
            left = add(add(add(sample[0], sample[1]), sample[2]), sample[3])
            right = add(add(add(sample[3], sample[0]), sample[1]), sample[2])
            total = sum(abs(v) for v in sample)
            if total and abs(left - right) > gamma3 * total:
                witness = (sample, abs(left - right), total)
                break
        self.assertIsNotNone(witness)
        _, gap, absolute = witness
        self.assertGreater(gap, gamma3 * absolute)
        self.assertLessEqual(gap, 2 * gamma3 * absolute + 1e-15)


if __name__ == "__main__":
    unittest.main()
