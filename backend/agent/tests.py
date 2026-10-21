from unittest import mock

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from rest_framework.test import APIClient

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


@override_settings(GEMINI_API_KEY="", GROK_API_KEY="")
class ErrorHandlingTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user("writer", password="Str0ng-pass-123")
        self.client = APIClient()
        self.client.force_authenticate(self.user)
        self.document = Document.objects.create(
            user=self.user,
            title="Thesis",
            content={"topic": "Remote work", "sections": [{"title": "Introduction", "content": "My own words."}]},
        )

    def test_writing_without_ai_leaves_text_alone(self):
        for message in ["write the introduction", "enhance the introduction", "fix grammar"]:
            result = autonomous.run_agent(self.document, message, model_choice="gemini")
            self.document.refresh_from_db()
            self.assertFalse(result["document_updated"], message)
            self.assertEqual(self.document.content["sections"][0]["content"], "My own words.")

    def test_bad_actions_return_client_errors(self):
        url = f"/api/agent/{self.document.id}/action/"
        cases = [
            ({"action": "bogus"}, 400),
            ({"action": "insert_section", "payload": {}}, 400),
            ({"action": "generate_chart", "payload": {"query": "nowhere"}}, 404),
            ({"action": "enhance_section", "payload": {"query": "Introduction", "instruction": "tighten"}}, 503),
        ]
        for body, expected in cases:
            response = self.client.post(url, body, format="json")
            self.assertEqual(response.status_code, expected, body)
            self.assertIn("error", response.json())
        self.document.refresh_from_db()
        self.assertEqual(self.document.content["sections"][0]["content"], "My own words.")

    def test_malformed_content_is_rejected(self):
        url = f"/api/documents/{self.document.id}/"
        for content in ["text", {"sections": "x"}, {"sections": [{"content": 5}]}]:
            response = self.client.patch(url, {"content": content}, format="json")
            self.assertEqual(response.status_code, 400, content)

    def test_grammar_without_section_means_whole_document(self):
        self.assertEqual(autonomous._heuristic_intent("fix grammar")["intent"], "enhance_document")
        self.assertEqual(
            autonomous._heuristic_intent("fix grammar in the introduction")["intent"], "enhance_section"
        )

    def test_image_caption_is_the_subject_not_the_command(self):
        self.assertEqual(autonomous._image_caption_subject("add an image of a city to introduction"), "a city")
        self.assertEqual(autonomous._image_caption_subject("add image"), "")
