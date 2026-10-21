import logging

import stripe
from django.conf import settings
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .stripe_sync import handle_event
from .usage import monthly_limit, subscription_for, used_this_month

logger = logging.getLogger(__name__)

# Checkout options the frontend can ask for: price setting name and Stripe mode.
_PRICES = {
    "pro_monthly": ("STRIPE_PRICE_PRO_MONTHLY", "subscription"),
    "pro_yearly": ("STRIPE_PRICE_PRO_YEARLY", "subscription"),
    "thesis_pass": ("STRIPE_PRICE_THESIS_PASS", "payment"),
}


def _stripe_ready() -> bool:
    return bool(settings.STRIPE_SECRET_KEY)


def _not_configured() -> Response:
    return Response(
        {"error": "Payments aren't set up yet. Please try again later."},
        status=status.HTTP_503_SERVICE_UNAVAILABLE,
    )


class BillingStatusView(APIView):
    def get(self, request):
        subscription = subscription_for(request.user)
        return Response(
            {
                "plan": "pro" if subscription.is_pro() else "free",
                "status": subscription.status,
                "access_until": subscription.access_until,
                "ai_requests_used": used_this_month(request.user),
                "ai_requests_limit": monthly_limit(request.user),
                "can_manage": bool(subscription.stripe_customer_id),
                "payments_enabled": _stripe_ready(),
                "options": [key for key, (setting, _) in _PRICES.items() if getattr(settings, setting, "")],
            }
        )


class CheckoutView(APIView):
    def post(self, request):
        if not _stripe_ready():
            return _not_configured()
        option = request.data.get("option")
        if option not in _PRICES:
            return Response({"error": f"Unknown plan option: {option!r}."}, status=status.HTTP_400_BAD_REQUEST)
        price_setting, mode = _PRICES[option]
        price_id = getattr(settings, price_setting, "")
        if not price_id:
            return _not_configured()

        subscription = subscription_for(request.user)
        if (
            mode == "subscription"
            and subscription.stripe_subscription_id
            and subscription.status in {"active", "trialing", "past_due", "unpaid"}
        ):
            return Response(
                {"error": "You already have a Pro subscription. Use Manage billing to change it."},
                status=status.HTTP_409_CONFLICT,
            )
        stripe.api_key = settings.STRIPE_SECRET_KEY
        try:
            if not subscription.stripe_customer_id:
                customer = stripe.Customer.create(
                    email=request.user.email or None,
                    metadata={"user_id": str(request.user.pk)},
                )
                subscription.stripe_customer_id = customer["id"]
                subscription.save(update_fields=["stripe_customer_id", "updated_at"])
            session = stripe.checkout.Session.create(
                mode=mode,
                customer=subscription.stripe_customer_id,
                client_reference_id=str(request.user.pk),
                line_items=[{"price": price_id, "quantity": 1}],
                allow_promotion_codes=True,
                success_url=f"{settings.FRONTEND_URL}/?billing=success",
                cancel_url=f"{settings.FRONTEND_URL}/?billing=cancelled",
                metadata={"user_id": str(request.user.pk), "option": option},
            )
        except stripe.error.StripeError as exc:
            logger.error("Stripe checkout failed for user %s: %s", request.user.pk, exc)
            return Response(
                {"error": "We couldn't start checkout. Please try again."},
                status=status.HTTP_502_BAD_GATEWAY,
            )
        return Response({"url": session["url"]})


class PortalView(APIView):
    """Stripe-hosted page where the user updates their card or cancels."""

    def post(self, request):
        if not _stripe_ready():
            return _not_configured()
        subscription = subscription_for(request.user)
        if not subscription.stripe_customer_id:
            return Response({"error": "You don't have a paid plan yet."}, status=status.HTTP_400_BAD_REQUEST)
        stripe.api_key = settings.STRIPE_SECRET_KEY
        try:
            session = stripe.billing_portal.Session.create(
                customer=subscription.stripe_customer_id,
                return_url=f"{settings.FRONTEND_URL}/",
            )
        except stripe.error.StripeError as exc:
            logger.error("Stripe portal failed for user %s: %s", request.user.pk, exc)
            return Response(
                {"error": "We couldn't open billing settings. Please try again."},
                status=status.HTTP_502_BAD_GATEWAY,
            )
        return Response({"url": session["url"]})


@csrf_exempt
@require_POST
def stripe_webhook(request):
    if not settings.STRIPE_WEBHOOK_SECRET:
        return HttpResponse(status=503)
    try:
        event = stripe.Webhook.construct_event(
            request.body,
            request.META.get("HTTP_STRIPE_SIGNATURE", ""),
            settings.STRIPE_WEBHOOK_SECRET,
        )
    except (ValueError, stripe.error.SignatureVerificationError):
        return HttpResponse(status=400)

    stripe.api_key = settings.STRIPE_SECRET_KEY
    handle_event(event, stripe.Subscription.retrieve)
    return HttpResponse(status=200)
