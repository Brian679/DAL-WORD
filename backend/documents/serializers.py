from rest_framework import serializers

from .models import Document, DocumentVersion, ChatMessage


class ChatMessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChatMessage
        fields = ["id", "role", "content", "created_at"]


class DocumentVersionSerializer(serializers.ModelSerializer):
    class Meta:
        model = DocumentVersion
        fields = ["id", "note", "created_at"]


class DocumentSerializer(serializers.ModelSerializer):
    versions = DocumentVersionSerializer(many=True, read_only=True)
    chat_messages = ChatMessageSerializer(many=True, read_only=True)

    class Meta:
        model = Document
        fields = [
            "id",
            "title",
            "content",
            "created_at",
            "updated_at",
            "versions",
            "chat_messages",
        ]

    def validate_content(self, value):
        if not isinstance(value, dict):
            raise serializers.ValidationError("content must be an object.")
        sections = value.get("sections", [])
        if not isinstance(sections, list) or not all(isinstance(s, dict) for s in sections):
            raise serializers.ValidationError("content.sections must be a list of objects.")
        for section in sections:
            if not isinstance(section.get("content") or "", str) or not isinstance(section.get("title") or "", str):
                raise serializers.ValidationError("each section's title and content must be text.")
        return value
