import os
import tempfile
import unittest

from reportlab.pdfgen import canvas

from app.core.pdf_processor import chunk_text, extract_text_from_pdf


class PdfProcessorTests(unittest.TestCase):
    def test_extract_text_from_pdf(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = os.path.join(temp_dir, "sample.pdf")

            doc = canvas.Canvas(pdf_path)
            doc.drawString(100, 700, "This is a sample PDF page.")
            doc.save()

            text = extract_text_from_pdf(pdf_path)
            self.assertIn("sample pdf page", text.lower())

    def test_chunk_text_creates_overlapping_chunks(self):
        text = " ".join(f"sentence_{index}" for index in range(1, 80))
        chunks = chunk_text(text, chunk_size=50, overlap=10)

        self.assertTrue(len(chunks) > 1)
        self.assertTrue(all(len(chunk) <= 50 for chunk in chunks))
        self.assertTrue(chunks[0] != chunks[1])


if __name__ == "__main__":
    unittest.main()
