from django.conf import settings
from django.db import transaction
from django.db.models import F
from django.utils import timezone

from .models import AIUsage, Subscription


def current_month() -> str:
    return timezone.now().strftime("%Y-%m")


def subscription_for(user) -> Subscription:
    subscription, _ = Subscription.objects.get_or_create(user=user)
    return subscription


def monthly_limit(user) -> int:
    if subscription_for(user).is_pro():
        return settings.BILLING_PRO_MONTHLY_AI_LIMIT
    return settings.BILLING_FREE_MONTHLY_AI_LIMIT


def used_this_month(user) -> int:
    row = AIUsage.objects.filter(user=user, month=current_month()).first()
    return row.count if row else 0


def try_consume_ai_request(user) -> bool:
    """Count one AI request against the user's plan; False when the limit is reached."""
    limit = monthly_limit(user)
    month = current_month()
    with transaction.atomic():
        AIUsage.objects.get_or_create(user=user, month=month)
        updated = AIUsage.objects.filter(user=user, month=month, count__lt=limit).update(count=F("count") + 1)
    return updated == 1


def limit_reached_payload(user) -> dict:
    return {
        "error": (
            f"You've used all {monthly_limit(user)} AI requests on your plan this month. "
            "Upgrade to Pro for more, or wait until next month."
        ),
        "code": "ai_limit_reached",
        "upgrade_required": not subscription_for(user).is_pro(),
    }
