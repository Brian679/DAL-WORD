from django.test import TestCase

from documents.models import Document, DocumentVersion

from . import autonomous


AI_LIKE = (
    "It is important to note that this study plays a crucial role in the field. "
    "Furthermore, it is essential to consider various factors that shape the outcome."
)


class IntegrityGuardrailTests(TestCase):
    """Detection and plagiarism tools report problems; they never disguise text."""

    def setUp(self):
        self.document = Document.objects.create(
            title="Test", content={"sections": [{"title": "Intro", "content": AI_LIKE}]}
        )

    def test_humanise_request_reports_without_rewriting(self):
        reply, updated = autonomous._humanise_ai_sections(self.document, "topic", [])
        self.document.refresh_from_db()
        self.assertFalse(updated)
        self.assertEqual(self.document.content["sections"][0]["content"], AI_LIKE)
        self.assertFalse(DocumentVersion.objects.exists())
        self.assertIn("doesn't rewrite text", reply)

    def test_reduce_similarity_request_reports_without_rewriting(self):
        reply, updated = autonomous._reduce_plagiarism_similarity(self.document, "topic", [])
        self.document.refresh_from_db()
        self.assertFalse(updated)
        self.assertEqual(self.document.content["sections"][0]["content"], AI_LIKE)
        self.assertIn("cite", reply)

    def test_evasion_intents_do_not_need_edit_confirmation(self):
        self.assertNotIn("humanise_ai_sections", autonomous.DOCUMENT_MODIFYING_INTENTS)
        self.assertNotIn("reduce_plagiarism_similarity", autonomous.DOCUMENT_MODIFYING_INTENTS)
