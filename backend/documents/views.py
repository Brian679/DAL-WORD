from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from .models import Document, DocumentVersion
from .serializers import DocumentSerializer


class DocumentViewSet(viewsets.ModelViewSet):
    queryset = Document.objects.all().order_by("-updated_at")
    serializer_class = DocumentSerializer

    def perform_update(self, serializer):
        instance = serializer.save()
        DocumentVersion.objects.create(document=instance, content=instance.content, note="manual-save")

    @action(detail=True, methods=["post"])
    def snapshot(self, request, pk=None):
        document = self.get_object()
        note = request.data.get("note", "snapshot")
        version = DocumentVersion.objects.create(document=document, content=document.content, note=note)
        return Response({"version_id": version.id, "note": version.note})
