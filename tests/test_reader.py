import os
import unittest
from raschlab.reader import read_matrix

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "mini_data.prn")


class TestReader(unittest.TestCase):
    def test_read_matrix(self):
        labels, rows = read_matrix(FIXTURE_PATH, item1=19, ni=5, namlen=16)
        self.assertEqual(len(labels), 4)
        for row in rows:
            self.assertEqual(len(row), 5)
        # Short line was padded with spaces
        self.assertEqual(rows[3], "AB   ")


if __name__ == "__main__":
    unittest.main()
