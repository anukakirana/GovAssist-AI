import os
import tempfile
import unittest

from reportlab.pdfgen import canvas
from unittest.mock import MagicMock

from app.routes.documents import index_pdf_document


class DocumentUploadTests(unittest.TestCase):
    def test_index_pdf_document_splits_and_stores_chunks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = os.path.join(temp_dir, "sample.pdf")

            doc = canvas.Canvas(pdf_path)
            doc.drawString(100, 700, "This document explains eligibility rules for government assistance.")
            doc.save()

            qdrant_service = MagicMock()
            qdrant_service.ensure_collection.return_value = True
            qdrant_service.add_document.return_value = {"status": "stored"}

            embedding_service = MagicMock()
            embedding_service.generate_embedding.side_effect = lambda text: [0.1, 0.2, 0.3] if text else []

            result = index_pdf_document(
                file_path=pdf_path,
                filename="sample.pdf",
                qdrant_service=qdrant_service,
                embedding_service=embedding_service,
            )

            self.assertIn("chunks", result)
            self.assertGreaterEqual(result["chunks"], 1)
            self.assertTrue(qdrant_service.ensure_collection.called)


if __name__ == "__main__":
    unittest.main()
