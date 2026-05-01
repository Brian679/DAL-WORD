from rest_framework import serializers

from .models import Document, DocumentVersion


class DocumentVersionSerializer(serializers.ModelSerializer):
    class Meta:
        model = DocumentVersion
        fields = ["id", "note", "created_at"]


class DocumentSerializer(serializers.ModelSerializer):
    versions = DocumentVersionSerializer(many=True, read_only=True)

    class Meta:
        model = Document
        fields = [
            "id",
            "title",
            "content",
            "created_at",
            "updated_at",
            "versions",
        ]
