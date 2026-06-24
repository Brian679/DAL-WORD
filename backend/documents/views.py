from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

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
