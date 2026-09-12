import os
import unittest
from raschlab.control import parse_control

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "cfile_mini.CON")


class TestControl(unittest.TestCase):
    def test_parse_control(self):
        ctrl = parse_control(FIXTURE_PATH)
        self.assertIsInstance(ctrl["NI"], int)
        self.assertEqual(ctrl["NI"], 5)
        self.assertIsInstance(ctrl["ITEM1"], int)
        self.assertEqual(ctrl["ITEM1"], 19)
        self.assertIsInstance(ctrl["NAMLEN"], int)
        self.assertEqual(ctrl["NAMLEN"], 16)
        self.assertIsInstance(ctrl["KEY1"], str)
        self.assertEqual(ctrl["KEY1"], "ABCDE")
        # Check comment stripping
        self.assertEqual(ctrl["CODES"], "ABCDE")
        self.assertEqual(ctrl["MISSCORE"], -1)


if __name__ == "__main__":
    unittest.main()
