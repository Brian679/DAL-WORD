import re

from django.http import FileResponse, HttpResponse
from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from .docx_export import build_docx
from .models import Document, DocumentVersion
from .serializers import DocumentSerializer

# Import the extraction helper from agent.views
from agent.views import _extract_file_text


class DocumentViewSet(viewsets.ModelViewSet):
    serializer_class = DocumentSerializer

    def get_queryset(self):
        return Document.objects.filter(user=self.request.user).order_by("-updated_at")

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

    def perform_update(self, serializer):
        instance = serializer.save()
        DocumentVersion.objects.create(document=instance, content=instance.content, note="manual-save")

    @action(detail=False, methods=["post"])
    def extract(self, request):
        uploaded_file = request.FILES.get("file")
        if not uploaded_file:
            return Response({"error": "no file provided"}, status=400)
        text = _extract_file_text(uploaded_file)
        return Response({"text": text})

    @action(detail=True, methods=["post"])
    def snapshot(self, request, pk=None):
        document = self.get_object()
        note = request.data.get("note", "snapshot")
        version = DocumentVersion.objects.create(document=document, content=document.content, note=note)
        return Response({"version_id": version.id, "note": version.note})

    @action(detail=True, methods=["get"])
    def export(self, request, pk=None):
        # Param is deliberately NOT named "format" — DRF's content negotiation
        # reserves that query param for its own renderer-suffix matching and
        # raises Http404 before this method body ever runs if a registered
        # renderer doesn't exist for the given value (there's no "docx" renderer).
        document = self.get_object()
        fmt = (request.query_params.get("as_format") or "docx").lower()
        safe_title = re.sub(r"[^A-Za-z0-9 _-]+", "", document.title or "document").strip() or "document"
        if fmt == "bib":
            bibtex_text = ((document.content or {}).get("bibliography_bibtex") or "").strip()
            if not bibtex_text:
                return Response({"error": "this document has no citation library to export"}, status=404)
            response = HttpResponse(bibtex_text, content_type="application/x-bibtex")
            response["Content-Disposition"] = f'attachment; filename="{safe_title}.bib"'
            return response
        if fmt != "docx":
            return Response({"error": f"unsupported export format: {fmt}"}, status=400)
        buffer = build_docx(document)
        return FileResponse(
            buffer,
            as_attachment=True,
            filename=f"{safe_title}.docx",
            content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
