import io
import logging

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from documents.models import Document, ChatMessage
from documents.serializers import DocumentSerializer
from tasks.dissertation_tasks import generate_dissertation_sections

from .autonomous import run_agent, generate_dissertation_plan_llm, llm_chapters_to_flat_steps, _research_design
from .agents_v2 import run_multi_agent_supervision
from .academic_runtime import ClaimGraphBuilder, CoherenceChecker, EvaluationEngine, WorkflowEngine
from .executor import run_action

logger = logging.getLogger(__name__)


def _doc_context(document: Document) -> str:
    """Flatten document content to plain text for Gemini context."""
    import re as _re
    content = document.content or {}
    parts = [document.title]
    comment_list: list[str] = []
    _comment_re = _re.compile(r'\[Comment:\s*([^\]]+)\]', _re.IGNORECASE)
    for section in content.get("sections", []):
        title = section.get("title", "")
        body = section.get("content", "")
        if title:
            wc = len(body.split()) if body else 0
            parts.append(f"\n## {title}  (~{wc} words)")
        if body:
            parts.append(body)
            for m in _comment_re.finditer(body):
                comment_list.append(f"  - In \"{title}\": {m.group(1).strip()}")
    if comment_list:
        parts.append(
            "\n\n## REVIEWER COMMENTS (inline annotations from the document):\n"
            + "\n".join(comment_list)
        )
    return "\n".join(parts)


class AgentActionView(APIView):
    def post(self, request, document_id: int):
        action = request.data.get("action")
        payload = request.data.get("payload", {})

        document = Document.objects.get(pk=document_id)
        updated = run_action(document, action, payload)
        return Response(DocumentSerializer(updated).data)


def _extract_file_text(uploaded_file) -> str:
    """Extract plain text from an uploaded PDF or DOCX file."""
    name = (uploaded_file.name or "").lower()
    raw = uploaded_file.read()
    try:
        if name.endswith(".pdf"):
            try:
                import pypdf
                reader = pypdf.PdfReader(io.BytesIO(raw))
                return "\n".join(page.extract_text() or "" for page in reader.pages)
            except ImportError:
                pass
            try:
                import PyPDF2
                reader = PyPDF2.PdfReader(io.BytesIO(raw))
                return "\n".join(page.extract_text() or "" for page in reader.pages)
            except ImportError:
                pass
        elif name.endswith(".docx"):
            try:
                import docx
                doc = docx.Document(io.BytesIO(raw))
                return "\n".join(p.text for p in doc.paragraphs)
            except ImportError:
                pass
        elif name.endswith(".txt"):
            return raw.decode("utf-8", errors="replace")
    except Exception as exc:
        logger.warning("File text extraction failed: %s", exc)
    return ""


class ChatView(APIView):
    def post(self, request, document_id: int):
        # Support both JSON and multipart (file upload)
        message = (request.data.get("message") or request.POST.get("message", "")).strip()
        model_choice = request.data.get("model") or request.POST.get("model", "grok")
        # preview_only=True → classify intent + build plan, but do NOT execute or persist messages.
        # Used by the frontend to show the user a confirmation card before the agent acts.
        preview_only_raw = request.data.get("preview_only") or request.POST.get("preview_only", "")
        preview_only = str(preview_only_raw).lower() in {"true", "1", "yes"}
        grounded_research_raw = request.data.get("grounded_research") or request.POST.get("grounded_research", "")
        verify_citations_raw = request.data.get("verify_citations") or request.POST.get("verify_citations", "")
        synthetic_mode_raw = request.data.get("synthetic_mode") or request.POST.get("synthetic_mode", "")
        grounded_research = str(grounded_research_raw).lower() in {"true", "1", "yes"}
        verify_citations = str(verify_citations_raw).lower() in {"true", "1", "yes"}
        synthetic_mode = str(synthetic_mode_raw).lower() in {"true", "1", "yes"}
        if not message:
            return Response({"error": "message is required"}, status=status.HTTP_400_BAD_REQUEST)

        # Optional file attachment — keep filename in chat, keep extracted text private for model context
        uploaded_file = request.FILES.get("file")
        attachment_text = ""
        attachment_note = ""
        if uploaded_file:
            attachment_note = f"[Attached file: {uploaded_file.name}]"
            file_text = _extract_file_text(uploaded_file)
            if file_text:
                attachment_text = file_text[:8000]

        try:
            document = Document.objects.get(pk=document_id)
        except Document.DoesNotExist:
            return Response({"error": "document not found"}, status=status.HTTP_404_NOT_FOUND)

        user_chat_content = message
        if attachment_note:
            user_chat_content = f"{message}\n\n{attachment_note}" if message else attachment_note

        # For normal (non-preview) requests, persist the user message up-front so it appears
        # in recent_history for the agent.  Preview requests don't persist anything yet.
        if not preview_only:
            ChatMessage.objects.create(document=document, role="user", content=user_chat_content)

        recent_history = list(
            ChatMessage.objects.filter(document=document)
            .order_by("-created_at")
            .values("role", "content")[:14]
        )
        recent_history.reverse()

        try:
            result = run_agent(
                document,
                message,
                model_choice=model_choice,
                recent_history=recent_history,
                attachment_text=attachment_text,
                preview_only=preview_only,
                grounded_research=grounded_research,
                verify_citations=verify_citations,
                synthetic_mode=synthetic_mode,
            )
        except Exception as exc:
            logger.error("Agent error for doc %d: %s", document_id, exc, exc_info=True)
            if not preview_only:
                ChatMessage.objects.create(document=document, role="assistant", content=f"Error: {exc}")
            return Response({"error": str(exc)}, status=status.HTTP_502_BAD_GATEWAY)

        awaiting_confirmation = result.get("awaiting_confirmation", False)

        if awaiting_confirmation:
            # Preview for a document-modifying intent: don't persist anything yet.
            pass
        elif preview_only:
            # preview_only request but a non-modifying intent (chat/summarize) executed fully.
            # Persist both messages now so they appear in history.
            ChatMessage.objects.create(document=document, role="user", content=user_chat_content)
            ChatMessage.objects.create(document=document, role="assistant", content=result["reply"])
        else:
            # Normal (confirmed) execution: persist assistant message.
            ChatMessage.objects.create(document=document, role="assistant", content=result["reply"])

        response_data = {
            "reply": result["reply"],
            "plan": result["plan"],
            "chat_summary": result.get("chat_summary", {}),
            "orchestration": result.get("orchestration", {}),
            "document_updated": result["document_updated"],
            "intent": result["intent"],
            "model": result.get("model", "Unknown Model"),
            "awaiting_confirmation": awaiting_confirmation,
            "confirmation": result.get("confirmation"),
            "research": result.get("research", {}),
            "citation_verification": result.get("citation_verification"),
        }
        if result["document_updated"]:
            document.refresh_from_db()
            response_data["document"] = DocumentSerializer(document).data

        return Response(response_data)


class GenerateDissertationView(APIView):
    def post(self, request, document_id: int):
        topic = request.data.get("topic", "Untitled Topic")
        task = generate_dissertation_sections.delay(document_id=document_id, topic=topic)
        return Response({"task_id": task.id}, status=status.HTTP_202_ACCEPTED)


class ResearchWorkflowView(APIView):
    """Run retrieval + specialist-agent supervision for research-grounded writing."""

    def post(self, request, document_id: int):
        message = (request.data.get("message") or "").strip()
        topic = (request.data.get("topic") or message or "").strip()
        if not topic:
            return Response({"error": "topic or message required"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            document = Document.objects.get(pk=document_id)
        except Document.DoesNotExist:
            return Response({"error": "document not found"}, status=status.HTTP_404_NOT_FOUND)

        supervision = run_multi_agent_supervision(topic=topic, instruction=message or topic, document_id=document.id)
        checker = CoherenceChecker()
        evaluator = EvaluationEngine()
        workflow = WorkflowEngine()

        doc_text = _doc_context(document)
        coherence = checker.check(doc_text)
        metrics = evaluator.score(doc_text)
        claim_graph = ClaimGraphBuilder().build(
            text=doc_text,
            citations=[c for c in supervision.get("retrieval", {}).get("top_papers", []) if isinstance(c, str)],
        )

        return Response(
            {
                "phase": workflow.choose_phase(message or topic),
                "supervision": supervision,
                "coherence": coherence,
                "metrics": {
                    "citation_density": metrics.citation_density,
                    "coherence_score": metrics.coherence_score,
                    "redundancy_score": metrics.redundancy_score,
                    "argument_consistency": metrics.argument_consistency,
                    "methodology_alignment": metrics.methodology_alignment,
                },
                "claim_graph": [
                    {
                        "claim": n.claim,
                        "citation": n.citation,
                        "confidence": n.confidence,
                        "source_passage": n.source_passage,
                        "source_location": n.source_location,
                    }
                    for n in claim_graph[:80]
                ],
            }
        )


class DissertationPlanView(APIView):
    """Call the LLM to generate a tailored dissertation chapter plan.

    The plan is returned to the frontend so the user sees a topic-specific
    todo list immediately, before writing begins. The generated chapter
    structure is also cached inside the document so the write step reuses it.
    """

    def post(self, request, document_id: int):
        message = (request.data.get("message") or "").strip()
        topic = (request.data.get("topic") or message[:300]).strip()
        if not topic and not message:
            return Response({"error": "message or topic required"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            document = Document.objects.get(pk=document_id)
        except Document.DoesNotExist:
            return Response({"error": "document not found"}, status=status.HTTP_404_NOT_FOUND)

        # Detect research design from the message so the plan is appropriate
        research_design = _research_design(message, topic, document)

        # Ask the LLM to generate the chapter plan
        try:
            llm_chapters = generate_dissertation_plan_llm(topic, message, research_design)
        except Exception as exc:
            logger.error("DissertationPlanView LLM call failed: %s", exc, exc_info=True)
            return Response({"error": "Plan generation failed"}, status=status.HTTP_502_BAD_GATEWAY)

        # Cache the plan in the document so _write_dissertation can reuse it
        existing = document.content or {}
        existing["_dissertation_plan_chapters"] = llm_chapters
        document.content = existing
        document.save(update_fields=["content"])

        # Convert to the flat step format the frontend uses
        flat_steps = llm_chapters_to_flat_steps(llm_chapters)
        return Response({"plan": flat_steps, "chapters": llm_chapters})


class AIDetectView(APIView):
    """
    Detect AI-generated content in a document.

    Uses perplexity approximation + burstiness analysis, mirroring the
    two-signal methodology documented in Turnitin's AI writing detector.
    """

    def post(self, request, document_id: int):
        try:
            document = Document.objects.get(pk=document_id)
        except Document.DoesNotExist:
            return Response({"error": "document not found"}, status=status.HTTP_404_NOT_FOUND)

        from .ai_detector import detect_ai_content

        # Caller may pass raw text directly (e.g. selected passage).
        # Otherwise we flatten all document sections.
        custom_text = (request.data.get("text") or "").strip()
        if custom_text:
            full_text = custom_text
        else:
            content = document.content or {}
            parts = []
            for section in content.get("sections", []):
                body = (section.get("content") or "").strip()
                if body:
                    parts.append(body)
            full_text = "\n\n".join(parts)

        result = detect_ai_content(full_text)
        return Response(result)


class AcademicQualityView(APIView):
    """
    Rule-based academic writing quality check per section.
    Returns per-section scores and actionable issue list.
    """

    def post(self, request, document_id: int):
        try:
            document = Document.objects.get(pk=document_id)
        except Document.DoesNotExist:
            return Response({"error": "document not found"}, status=status.HTTP_404_NOT_FOUND)

        from .ai_detector import academic_quality_check

        custom_text = (request.data.get("text") or "").strip()
        if custom_text:
            result = academic_quality_check(custom_text)
            return Response({"sections": [{"title": "Selection", "result": result}], "overall": result})

        content = document.content or {}
        sections_out = []
        all_text_parts = []
        for section in content.get("sections", []):
            body = (section.get("content") or "").strip()
            if not body:
                continue
            all_text_parts.append(body)
            result = academic_quality_check(body)
            sections_out.append({
                "title": section.get("title") or "Untitled",
                "result": result,
            })

        overall = academic_quality_check("\n\n".join(all_text_parts)) if all_text_parts else {
            "quality_score": 0, "verdict": "insufficient_text", "issues": [], "word_count": 0
        }
        return Response({"sections": sections_out, "overall": overall})


class PlagiarismCheckView(APIView):
    """
    Check a document for plagiarism against every other document in the
    workspace, using shingled n-gram fingerprint matching — the same
    technique Turnitin's OriginalityCheck index is built on — and, unless
    disabled, against the open scholarly web (Crossref, arXiv, PubMed, SSRN,
    Semantic Scholar, and Google Scholar if ALLOW_GOOGLE_SCHOLAR_SCRAPE=1).
    """

    def post(self, request, document_id: int):
        try:
            document = Document.objects.get(pk=document_id)
        except Document.DoesNotExist:
            return Response({"error": "document not found"}, status=status.HTTP_404_NOT_FOUND)

        from .plagiarism_detector import check_plagiarism
        from .web_plagiarism import (
            check_external_plagiarism,
            external_check_enabled,
            merge_into_check_result,
        )

        custom_text = (request.data.get("text") or "").strip()
        if custom_text:
            full_text = custom_text
        else:
            content = document.content or {}
            parts = []
            for section in content.get("sections", []):
                body = (section.get("content") or "").strip()
                if body:
                    parts.append(body)
            full_text = "\n\n".join(parts)

        source_docs = []
        for other in Document.objects.exclude(pk=document_id).only("id", "title", "content"):
            other_content = other.content or {}
            parts = []
            for section in other_content.get("sections", []):
                body = (section.get("content") or "").strip()
                if body:
                    parts.append(body)
            other_text = "\n\n".join(parts)
            if other_text.strip():
                source_docs.append((other.id, other.title or f"Document {other.id}", other_text))

        result = check_plagiarism(full_text, source_docs)

        include_external = str(request.data.get("include_external", "")).strip().lower()
        run_external = external_check_enabled() if include_external == "" else include_external not in ("0", "false")
        if run_external and full_text.strip():
            external = check_external_plagiarism(full_text)
            result = merge_into_check_result(result, external)
        else:
            result["external_checked"] = False
            result["external_papers_checked"] = 0

        return Response(result)
