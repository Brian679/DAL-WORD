"""Apply Stripe webhook events to local Subscription rows."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone as dt_timezone
from typing import Any

from django.conf import settings
from django.contrib.auth import get_user_model
from django.db import IntegrityError, transaction
from django.utils import timezone

from .models import ProcessedStripeEvent, Subscription

logger = logging.getLogger(__name__)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    try:
        value = obj[key]
    except (KeyError, TypeError, IndexError):
        return default
    return default if value is None else value


def _from_timestamp(value: Any) -> datetime | None:
    return datetime.fromtimestamp(int(value), tz=dt_timezone.utc) if value else None


def _period_end(stripe_subscription: Any) -> datetime | None:
    # Newer Stripe API versions moved current_period_end onto the subscription items.
    end = _get(stripe_subscription, "current_period_end")
    if not end:
        items = _get(_get(stripe_subscription, "items", {}), "data", [])
        end = _get(items[0], "current_period_end") if items else None
    return _from_timestamp(end)


def _subscription_for_customer(customer_id: str) -> Subscription | None:
    return Subscription.objects.filter(stripe_customer_id=customer_id).first() if customer_id else None


def apply_subscription(subscription: Subscription, stripe_subscription: Any) -> None:
    status = _get(stripe_subscription, "status", "")
    subscription.stripe_subscription_id = _get(stripe_subscription, "id", "")
    subscription.status = status
    subscription.plan = Subscription.PLAN_FREE if status in {"canceled", "incomplete_expired"} else Subscription.PLAN_PRO
    period_end = _period_end(stripe_subscription)
    # Keep a longer Thesis Pass if the user already has one.
    if period_end and (not subscription.access_until or period_end > subscription.access_until):
        subscription.access_until = period_end
    if subscription.access_until and subscription.access_until > timezone.now():
        subscription.plan = Subscription.PLAN_PRO
    subscription.save()


def _grant_pass(subscription: Subscription, session_id: str) -> None:
    # Stripe can send separate events for the same session, so fulfil each session once.
    try:
        with transaction.atomic():
            ProcessedStripeEvent.objects.create(event_id=f"checkout:{session_id}")
    except IntegrityError:
        return
    start = max(timezone.now(), subscription.access_until or timezone.now())
    subscription.access_until = start + timedelta(days=settings.BILLING_THESIS_PASS_DAYS)
    subscription.plan = Subscription.PLAN_PRO
    if subscription.status not in {"active", "trialing"}:
        subscription.status = "pass"
    subscription.save()


def _on_checkout_completed(session: Any, retrieve_subscription) -> None:
    user_id = _get(session, "client_reference_id")
    user = get_user_model().objects.filter(pk=user_id).first() if user_id else None
    if user is None:
        logger.warning("Stripe checkout %s has no matching user", _get(session, "id"))
        return
    subscription, _ = Subscription.objects.get_or_create(user=user)
    customer_id = _get(session, "customer", "")
    if customer_id:
        subscription.stripe_customer_id = customer_id
    if _get(session, "mode") == "subscription" and _get(session, "subscription"):
        apply_subscription(subscription, retrieve_subscription(_get(session, "subscription")))
    elif _get(session, "mode") == "payment" and _get(session, "payment_status") == "paid":
        _grant_pass(subscription, _get(session, "id", ""))
    else:
        subscription.save()


def handle_event(event: Any, retrieve_subscription) -> bool:
    """Apply one verified Stripe event. Returns False if it was already processed."""
    try:
        with transaction.atomic():
            ProcessedStripeEvent.objects.create(event_id=_get(event, "id"))
            event_type = _get(event, "type", "")
            obj = _get(_get(event, "data", {}), "object", {})
            # Delayed payment methods complete checkout unpaid, then send async_payment_succeeded.
            if event_type in {"checkout.session.completed", "checkout.session.async_payment_succeeded"}:
                _on_checkout_completed(obj, retrieve_subscription)
            elif event_type in {
                "customer.subscription.created",
                "customer.subscription.updated",
                "customer.subscription.deleted",
            }:
                subscription = _subscription_for_customer(_get(obj, "customer", ""))
                if subscription is not None:
                    # Stripe doesn't guarantee event order, so apply the subscription as it
                    # is now rather than the possibly stale snapshot in this event.
                    apply_subscription(subscription, retrieve_subscription(_get(obj, "id")))
    except IntegrityError:
        return False
    return True
