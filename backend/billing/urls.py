from django.urls import path

from .views import BillingStatusView, CheckoutView, PortalView, stripe_webhook

urlpatterns = [
    path("status/", BillingStatusView.as_view(), name="billing-status"),
    path("checkout/", CheckoutView.as_view(), name="billing-checkout"),
    path("portal/", PortalView.as_view(), name="billing-portal"),
    path("webhook/", stripe_webhook, name="billing-webhook"),
]
