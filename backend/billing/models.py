from django.conf import settings
from django.db import models
from django.utils import timezone


class Subscription(models.Model):
    """One row per user: their Stripe customer and current paid access."""

    PLAN_FREE = "free"
    PLAN_PRO = "pro"
    PLAN_CHOICES = [(PLAN_FREE, "Free"), (PLAN_PRO, "Pro")]

    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="subscription")
    plan = models.CharField(max_length=20, choices=PLAN_CHOICES, default=PLAN_FREE)
    # Stripe's subscription status ("active", "past_due", "canceled", ...), or
    # "pass" for a one-off Thesis Pass that simply runs until access_until.
    status = models.CharField(max_length=30, blank=True)
    stripe_customer_id = models.CharField(max_length=255, blank=True, db_index=True)
    stripe_subscription_id = models.CharField(max_length=255, blank=True, db_index=True)
    access_until = models.DateTimeField(null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    def is_pro(self) -> bool:
        if self.plan != self.PLAN_PRO:
            return False
        if self.status in {"active", "trialing"}:
            return True
        # Passes, and subscriptions that were cancelled but are paid up to period end.
        return bool(self.access_until and self.access_until > timezone.now())

    def __str__(self) -> str:
        return f"{self.user} ({self.plan}, {self.status or 'none'})"


class AIUsage(models.Model):
    """AI requests per user per calendar month, for plan limits."""

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="ai_usage")
    month = models.CharField(max_length=7)  # "2026-10"
    count = models.PositiveIntegerField(default=0)

    class Meta:
        constraints = [models.UniqueConstraint(fields=["user", "month"], name="unique_ai_usage_per_month")]


class ProcessedStripeEvent(models.Model):
    """Stripe retries webhooks, so each event id is applied once."""

    event_id = models.CharField(max_length=255, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
