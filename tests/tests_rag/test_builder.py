import unittest
from ...src.raglight.rag.builder import Builder


class TestRAGBuilder(unittest.TestCase):
    def setUp(self):
        self.builder = Builder()
        self.assertEqual(True, True, "Null test")


if __name__ == "__main__":
    unittest.main()
