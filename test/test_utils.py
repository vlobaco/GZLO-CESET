import unittest
from gzlo_ceset.utils import (
    mine_text_from_docx, 
    mine_text_from_pdf,
    preprocess_text,
    )

class TestUtils(unittest.TestCase):

    def test_mine_pdf(self):
        file_path = "test/test_file.pdf"
        text = mine_text_from_pdf(file_path)
        self.assertTrue(text.startswith("Hello World!"))

    def test_mine_docx(self):
        file_path = "test/test_file.docx"
        text = mine_text_from_docx(file_path)
        self.assertTrue(text.startswith("Hello World!"))

    def test_preprocess_text(self):
        text = "Hola, mamá.\n¿Cómo estás?"
        self.assertEqual(preprocess_text(
            text,
            remove_accents= True,
            remove_punctuation= True,
            remove_stop_words= True,
            ), ["hola", "mama"])
