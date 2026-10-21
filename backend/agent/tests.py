from unittest import mock

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


class NoFullDraftTests(TestCase):
    """Requests to write a whole dissertation get an outline, not a finished draft."""

    def test_empty_document_gets_outline(self):
        document = Document.objects.create(title="Thesis", content={})
        with mock.patch.object(
            autonomous, "generate_outline_sections",
            return_value=[{"title": "Introduction"}, {"title": "Literature Review"}],
        ):
            reply, updated = autonomous._outline_instead_of_full_draft(document, "Remote work", [])
        document.refresh_from_db()
        self.assertTrue(updated)
        self.assertIn("doesn't write whole dissertations", reply)
        self.assertEqual(
            [s["title"] for s in document.content["sections"]], ["Introduction", "Literature Review"]
        )
        self.assertTrue(all(s["content"] == "" for s in document.content["sections"]))

    def test_existing_work_is_left_alone(self):
        document = Document.objects.create(
            title="Thesis", content={"sections": [{"title": "Intro", "content": AI_LIKE}]}
        )
        with mock.patch.object(autonomous, "generate_outline_sections") as outline:
            reply, updated = autonomous._outline_instead_of_full_draft(document, "Remote work", [])
        outline.assert_not_called()
        document.refresh_from_db()
        self.assertFalse(updated)
        self.assertEqual(document.content["sections"][0]["content"], AI_LIKE)


class SampleDataLabelTests(TestCase):
    """Generated numbers are labelled as sample data; the writer's own numbers are not."""

    def _chart(self, instruction):
        document = Document.objects.create(
            title="Thesis", content={"sections": [{"title": "Results", "content": "Text."}]}
        )
        with mock.patch.object(autonomous, "generate_chart", return_value="/media/c.png") as chart, \
                mock.patch.object(
                    autonomous, "_ai_chart_series",
                    return_value={"series": [1, 2, 3], "x_labels": ["a", "b", "c"], "chart_type": "bar"},
                ):
            autonomous._add_chart(document, None, [], instruction)
        document.refresh_from_db()
        section = document.content["sections"][0]
        return section, chart.call_args.kwargs["title"]

    def test_generated_chart_is_labelled(self):
        section, title = self._chart("add a chart")
        self.assertIn("SAMPLE DATA", section["blocks"][0]["caption"])
        self.assertIn("SAMPLE DATA", title)
        self.assertIn("not real survey/experimental results", section["content"])

    def test_user_data_chart_is_not_labelled(self):
        section, title = self._chart("plot Agree: 42, Neutral: 10, Disagree: 8")
        self.assertNotIn("SAMPLE DATA", section["blocks"][0]["caption"])
        self.assertNotIn("not real survey/experimental results", section["content"])
