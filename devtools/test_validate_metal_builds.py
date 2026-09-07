"""Check that the Metal matrix validator cannot count incomplete GPU runs as passed."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET


spec = importlib.util.spec_from_file_location(
    "validate_metal_builds", Path(__file__).with_name("validate-metal-builds.py"))
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


class MetalBuildResultsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.expected = list(validator.TESTS)

    def write_results(self, mode="offline", names=None, statuses=None, log=None):
        """Create CTest dashboard records without configuring or accessing a GPU."""
        names = self.expected if names is None else names
        statuses = ["passed"] * len(names) if statuses is None else statuses
        if log is None:
            path = "offline metallib" if mode == "offline" else "runtime compilation"
            log = "1: Fixed-point oracle kernels: " + path + "\n1: Done\n"
        (self.directory / "test.log").write_text(log)
        testing = self.directory / "Testing"
        tag = "20260908-0000"
        records = testing / tag
        records.mkdir(parents=True, exist_ok=True)
        (testing / "TAG").write_text(tag + "\nExperimental\n")
        site = ET.Element("Site")
        tests = ET.SubElement(site, "Testing")
        for name, status in zip(names, statuses):
            test = ET.SubElement(tests, "Test", Status=status)
            ET.SubElement(test, "Name").text = name
        ET.ElementTree(site).write(records / "Test.xml", encoding="utf-8",
                                   xml_declaration=True)

    def test_accepts_exactly_five_passed_tests_for_each_matrix_case(self):
        for mode in ("offline", "runtime"):
            for suffix in ("", "Static"):
                with self.subTest(mode=mode, suffix=suffix):
                    expected = [name + suffix for name in validator.TESTS]
                    self.write_results(mode=mode, names=expected)
                    self.assertEqual(
                        [{"name": name, "status": "passed"} for name in expected],
                        validator.check_results(self.directory, expected, mode))

    def test_rejects_success_on_gpu_skip(self):
        self.write_results(log="1: Fixed-point oracle kernels: offline metallib\n"
                               "2: Test skipped: no supported Metal device is visible\n")
        with self.assertRaisesRegex(RuntimeError, "skipped GPU execution"):
            validator.check_results(self.directory, self.expected, "offline")

    def test_rejects_missing_or_incorrect_oracle_path(self):
        for log in ("1: Done\n",
                    "1: Fixed-point oracle kernels: runtime compilation\n",
                    "1: Fixed-point oracle kernels: offline metallib\n"
                    "1: Fixed-point oracle kernels: offline metallib\n"):
            with self.subTest(log=log):
                self.write_results(log=log)
                with self.assertRaisesRegex(RuntimeError, "Expected oracle path"):
                    validator.check_results(self.directory, self.expected, "offline")

    def test_rejects_missing_extra_or_duplicate_test_records(self):
        cases = {
            "missing": self.expected[:-1],
            "extra": self.expected + ["TestUnexpectedMetalKernel"],
            "duplicate": self.expected[:-1] + [self.expected[0]],
        }
        for name, records in cases.items():
            with self.subTest(case=name):
                self.write_results(names=records)
                with self.assertRaisesRegex(RuntimeError, "five expected Metal tests"):
                    validator.check_results(self.directory, self.expected, "offline")

    def test_rejects_failed_or_notrun_test_records(self):
        for status in ("failed", "notrun"):
            with self.subTest(status=status):
                statuses = ["passed"] * len(self.expected)
                statuses[-1] = status
                self.write_results(statuses=statuses)
                with self.assertRaisesRegex(RuntimeError, "Not all GPU tests passed"):
                    validator.check_results(self.directory, self.expected, "offline")


if __name__ == "__main__":
    unittest.main()
