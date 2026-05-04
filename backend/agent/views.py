import io
import logging

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from documents.models import Document
from documents.serializers import DocumentSerializer
from tasks.dissertation_tasks import generate_dissertation_sections

from .autonomous import run_agent
from .executor import run_action

logger = logging.getLogger(__name__)


def _doc_context(document: Document) -> str:
    """Flatten document content to plain text for Gemini context."""
    content = document.content or {}
    parts = [document.title]
    for section in content.get("sections", []):
        title = section.get("title", "")
        body = section.get("content", "")
        if title:
            parts.append(f"\n## {title}")
        if body:
            parts.append(body)
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
        if not message:
            return Response({"error": "message is required"}, status=status.HTTP_400_BAD_REQUEST)

        # Optional file attachment — extract its text and append as context
        uploaded_file = request.FILES.get("file")
        if uploaded_file:
            file_text = _extract_file_text(uploaded_file)
            if file_text:
                message = f"{message}\n\n[Attached file: {uploaded_file.name}]\n{file_text[:8000]}"

        try:
            document = Document.objects.get(pk=document_id)
        except Document.DoesNotExist:
            return Response({"error": "document not found"}, status=status.HTTP_404_NOT_FOUND)

        try:
            result = run_agent(document, message, model_choice=model_choice)
        except Exception as exc:
            logger.error("Agent error for doc %d: %s", document_id, exc, exc_info=True)
            return Response({"error": str(exc)}, status=status.HTTP_502_BAD_GATEWAY)

        response_data = {
            "reply": result["reply"],
            "plan": result["plan"],
            "chat_summary": result.get("chat_summary", {}),
            "orchestration": result.get("orchestration", {}),
            "document_updated": result["document_updated"],
            "intent": result["intent"],
            "model": result.get("model", "Unknown Model"),
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
