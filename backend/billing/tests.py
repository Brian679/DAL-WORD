import hashlib
import hmac
import json
import time
from datetime import timedelta
from unittest import mock

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.utils import timezone
from rest_framework.test import APIClient

from documents.models import Document

from .models import Subscription
from .stripe_sync import handle_event
from .usage import subscription_for, try_consume_ai_request, used_this_month


def _event(event_id, event_type, obj):
    return {"id": event_id, "type": event_type, "data": {"object": obj}}


def _stripe_subscription(status="active", days=30):
    end = int((timezone.now() + timedelta(days=days)).timestamp())
    return {"id": "sub_1", "customer": "cus_1", "status": status, "items": {"data": [{"current_period_end": end}]}}


class WebhookSyncTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user("buyer", password="Str0ng-pass-123")

    def test_subscription_checkout_makes_user_pro(self):
        session = {"id": "cs_1", "client_reference_id": str(self.user.pk), "customer": "cus_1",
                   "mode": "subscription", "subscription": "sub_1"}
        handle_event(_event("evt_1", "checkout.session.completed", session), lambda _id: _stripe_subscription())
        subscription = subscription_for(self.user)
        self.assertTrue(subscription.is_pro())
        self.assertEqual(subscription.stripe_customer_id, "cus_1")

    def test_cancellation_ends_pro_after_period(self):
        Subscription.objects.create(user=self.user, stripe_customer_id="cus_1", plan="pro", status="active")
        canceled = _stripe_subscription("canceled", days=-1)
        handle_event(_event("evt_2", "customer.subscription.deleted", canceled), lambda _id: canceled)
        self.assertFalse(subscription_for(self.user).is_pro())

    def test_late_stale_update_does_not_restore_pro(self):
        Subscription.objects.create(user=self.user, stripe_customer_id="cus_1", plan="pro", status="active")
        canceled = _stripe_subscription("canceled", days=-1)
        handle_event(_event("evt_5", "customer.subscription.deleted", canceled), lambda _id: canceled)
        # An older "active" snapshot arrives late; the current state from Stripe wins.
        handle_event(_event("evt_6", "customer.subscription.updated", _stripe_subscription("active")),
                     lambda _id: canceled)
        self.assertFalse(subscription_for(self.user).is_pro())

    def test_same_session_in_two_events_grants_one_pass(self):
        session = {"id": "cs_7", "client_reference_id": str(self.user.pk), "customer": "cus_1",
                   "mode": "payment", "payment_status": "paid"}
        handle_event(_event("evt_7", "checkout.session.completed", session), None)
        first = subscription_for(self.user).access_until
        handle_event(_event("evt_8", "checkout.session.async_payment_succeeded", session), None)
        self.assertEqual(subscription_for(self.user).access_until, first)

    def test_delayed_payment_is_fulfilled_when_it_succeeds(self):
        session = {"id": "cs_8", "client_reference_id": str(self.user.pk), "customer": "cus_1",
                   "mode": "payment", "payment_status": "unpaid"}
        handle_event(_event("evt_9", "checkout.session.completed", session), None)
        self.assertFalse(subscription_for(self.user).is_pro())
        handle_event(_event("evt_10", "checkout.session.async_payment_succeeded",
                            {**session, "payment_status": "paid"}), None)
        self.assertTrue(subscription_for(self.user).is_pro())

    def test_active_subscription_past_its_period_lapses(self):
        Subscription.objects.create(user=self.user, plan="pro", status="active",
                                    access_until=timezone.now() - timedelta(days=4))
        self.assertFalse(subscription_for(self.user).is_pro())

    def test_thesis_pass_grants_thirty_days(self):
        session = {"id": "cs_2", "client_reference_id": str(self.user.pk), "customer": "cus_1",
                   "mode": "payment", "payment_status": "paid"}
        handle_event(_event("evt_3", "checkout.session.completed", session), None)
        subscription = subscription_for(self.user)
        self.assertTrue(subscription.is_pro())
        self.assertGreater(subscription.access_until, timezone.now() + timedelta(days=29))

    def test_duplicate_events_apply_once(self):
        session = {"id": "cs_3", "client_reference_id": str(self.user.pk), "customer": "cus_1",
                   "mode": "payment", "payment_status": "paid"}
        self.assertTrue(handle_event(_event("evt_4", "checkout.session.completed", session), None))
        first = subscription_for(self.user).access_until
        self.assertFalse(handle_event(_event("evt_4", "checkout.session.completed", session), None))
        self.assertEqual(subscription_for(self.user).access_until, first)


@override_settings(STRIPE_WEBHOOK_SECRET="whsec_test", STRIPE_SECRET_KEY="sk_test")
class WebhookEndpointTests(TestCase):
    def test_signed_pass_purchase_upgrades_user(self):
        user = get_user_model().objects.create_user("payer", password="Str0ng-pass-123")
        payload = json.dumps(_event("evt_9", "checkout.session.completed", {
            "id": "cs_9", "object": "checkout.session", "client_reference_id": str(user.pk),
            "customer": "cus_9", "mode": "payment", "payment_status": "paid",
        })).encode()
        timestamp = int(time.time())
        signature = hmac.new(b"whsec_test", f"{timestamp}.".encode() + payload, hashlib.sha256).hexdigest()
        response = self.client.post("/api/billing/webhook/", data=payload, content_type="application/json",
                                    HTTP_STRIPE_SIGNATURE=f"t={timestamp},v1={signature}")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(subscription_for(user).is_pro())

    def test_rejects_bad_signature(self):
        response = self.client.post("/api/billing/webhook/", data=b"{}", content_type="application/json",
                                    HTTP_STRIPE_SIGNATURE="t=1,v1=bad")
        self.assertEqual(response.status_code, 400)


@override_settings(BILLING_FREE_MONTHLY_AI_LIMIT=2, BILLING_PRO_MONTHLY_AI_LIMIT=5,
                   GEMINI_API_KEY="", GROK_API_KEY="", STRIPE_SECRET_KEY="")
class UsageLimitTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user("student", password="Str0ng-pass-123")
        self.client = APIClient()
        self.client.force_authenticate(self.user)
        self.document = Document.objects.create(user=self.user, title="T", content={"sections": []})

    def test_free_plan_stops_at_limit(self):
        url = f"/api/agent/{self.document.id}/chat/"
        for _ in range(2):
            self.assertEqual(self.client.post(url, {"message": "check academic quality"}, format="json").status_code, 200)
        response = self.client.post(url, {"message": "check academic quality"}, format="json")
        self.assertEqual(response.status_code, 402)
        self.assertEqual(response.json()["code"], "ai_limit_reached")
        self.assertEqual(used_this_month(self.user), 2)

    def test_pro_plan_gets_higher_limit(self):
        Subscription.objects.create(user=self.user, plan="pro", status="active")
        self.assertEqual(sum(try_consume_ai_request(self.user) for _ in range(7)), 5)

    def test_status_and_checkout_without_stripe(self):
        status_response = self.client.get("/api/billing/status/").json()
        self.assertEqual(status_response["plan"], "free")
        self.assertFalse(status_response["payments_enabled"])
        self.assertEqual(self.client.post("/api/billing/checkout/", {"option": "pro_monthly"}, format="json").status_code, 503)

    @override_settings(STRIPE_SECRET_KEY="sk_test", STRIPE_PRICE_PRO_MONTHLY="price_123")
    def test_checkout_creates_customer_and_session(self):
        with mock.patch("stripe.Customer.create", return_value={"id": "cus_9"}) as customer, \
                mock.patch("stripe.checkout.Session.create", return_value={"url": "https://checkout.stripe.com/x"}) as session:
            response = self.client.post("/api/billing/checkout/", {"option": "pro_monthly"}, format="json")
        self.assertEqual(response.json()["url"], "https://checkout.stripe.com/x")
        customer.assert_called_once()
        kwargs = session.call_args.kwargs
        self.assertEqual(kwargs["mode"], "subscription")
        self.assertEqual(kwargs["customer"], "cus_9")
        self.assertEqual(kwargs["client_reference_id"], str(self.user.pk))
        self.assertEqual(self.client.post("/api/billing/checkout/", {"option": "gold"}, format="json").status_code, 400)

    @override_settings(STRIPE_SECRET_KEY="sk_test", STRIPE_PRICE_PRO_MONTHLY="price_123")
    def test_active_subscriber_cannot_start_second_subscription(self):
        Subscription.objects.create(user=self.user, plan="pro", status="active",
                                    stripe_customer_id="cus_1", stripe_subscription_id="sub_1")
        with mock.patch("stripe.checkout.Session.create") as session:
            response = self.client.post("/api/billing/checkout/", {"option": "pro_monthly"}, format="json")
        self.assertEqual(response.status_code, 409)
        session.assert_not_called()

    def test_invalid_action_is_not_charged(self):
        url = f"/api/agent/{self.document.id}/action/"
        response = self.client.post(url, {"action": "enhance_section", "payload": {"query": "Nowhere",
                                                                                  "instruction": "x"}}, format="json")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(used_this_month(self.user), 0)

    def test_preview_that_waits_for_confirmation_is_refunded(self):
        url = f"/api/agent/{self.document.id}/chat/"
        response = self.client.post(url, {"message": "create outline for remote work", "preview_only": True},
                                    format="json")
        self.assertTrue(response.json()["awaiting_confirmation"])
        self.assertEqual(used_this_month(self.user), 0)
