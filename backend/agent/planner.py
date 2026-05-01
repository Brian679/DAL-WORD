from dataclasses import dataclass


@dataclass
class PlanItem:
    order: int
    title: str


def create_dissertation_outline(topic: str) -> list[PlanItem]:
    defaults = [
        "Introduction",
        "Literature Review",
        "Methodology",
        "Results",
        "Discussion",
        "Conclusion",
    ]
    return [PlanItem(order=i + 1, title=f"{i + 1}. {name}") for i, name in enumerate(defaults)]


def to_json(items: list[PlanItem]) -> list[dict[str, str | int]]:
    return [{"order": item.order, "title": item.title} for item in items]
