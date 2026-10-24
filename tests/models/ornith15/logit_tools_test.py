"""Host-only checks for the offline logit comparator; no model/device needed."""

import array
import json
import math
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
VOCAB = 248320


class LogitToolsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="ornith-logit-tools-")
        cls.directory = Path(cls.temp.name)
        cls.binary = cls.directory / "compare_logits"
        subprocess.run(["g++", "-std=c++20", "-O2",
                        str(ROOT / "tools/ornith15/compare_logits.cpp"),
                        "-o", str(cls.binary)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def capture(self, name, first_pair, tokens=(42, 21)):
        path = self.directory / name
        row = array.array("f", [-100.0]) * VOCAB
        row[0], row[1] = first_pair
        with path.open("wb") as out:
            out.write(struct.pack("<4sIII2i", b"ORNL", 1, 2, VOCAB, *tokens))
            out.write(row.tobytes())
            row[0], row[1] = 100.0, 0.0
            out.write(row.tobytes())
        return path

    def run_comparison(self, first, second):
        return subprocess.run([str(self.binary), str(first), str(second)],
                              capture_output=True, text=True)

    def test_closed_form_kl_directions_and_average(self):
        first = self.capture("first.ologits", (0.0, 0.0))
        second = self.capture("second.ologits", (math.log(3.0), 0.0))
        result = self.run_comparison(first, second)
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertAlmostEqual(data["first_second_mean_kl"],
                               math.log(4.0 / 3.0) / 4.0, places=7)
        self.assertAlmostEqual(data["second_first_mean_kl"],
                               (0.75 * math.log(1.5) + 0.25 * math.log(0.5)) / 2.0,
                               places=7)
        self.assertEqual(data["positions"], 2)
        self.assertEqual(data["top1_agreements"], 2)

    def test_identical_capture(self):
        first = self.capture("same.ologits", (0.0, 0.0))
        data = json.loads(self.run_comparison(first, first).stdout)
        self.assertEqual(data["first_second_mean_kl"], 0.0)
        self.assertEqual(data["second_first_mean_kl"], 0.0)

    def test_reject_different_history(self):
        first = self.capture("identity-a.ologits", (0.0, 0.0))
        second = self.capture("identity-b.ologits", (0.0, 0.0), tokens=(43, 21))
        self.assertNotEqual(self.run_comparison(first, second).returncode, 0)

    def test_reject_nonfinite_logits(self):
        first = self.capture("finite.ologits", (0.0, 0.0))
        second = self.capture("nonfinite.ologits", (float("nan"), 0.0))
        self.assertNotEqual(self.run_comparison(first, second).returncode, 0)

    def test_reject_truncated_capture(self):
        first = self.capture("complete.ologits", (0.0, 0.0))
        second = self.directory / "truncated.ologits"
        second.write_bytes(first.read_bytes()[:-4])
        self.assertNotEqual(self.run_comparison(first, second).returncode, 0)


if __name__ == "__main__":
    unittest.main()
