from celery import shared_task
from django.utils import timezone

from documents.models import Document, DocumentVersion

from agent.planner import create_dissertation_outline


@shared_task
def generate_dissertation_sections(document_id: int, topic: str) -> dict[str, str | int]:
    document = Document.objects.get(pk=document_id)
    outline = create_dissertation_outline(topic)

    sections = []
    for item in outline:
        sections.append(
            {
                "title": item.title,
                "content": (
                    f"Draft section for {item.title} in topic '{topic}'. "
                    "Expand this with citations, arguments, and evidence."
                ),
                "status": "generated",
                "generated_at": timezone.now().isoformat(),
            }
        )

    document.content = {"topic": topic, "sections": sections}
    document.save(update_fields=["content", "updated_at"])
    DocumentVersion.objects.create(document=document, content=document.content, note="full-dissertation-generated")

    return {"document_id": document.id, "sections": len(sections), "status": "done"}
