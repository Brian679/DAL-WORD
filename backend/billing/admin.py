from django.contrib import admin

from .models import AIUsage, Subscription


@admin.register(Subscription)
class SubscriptionAdmin(admin.ModelAdmin):
    list_display = ("user", "plan", "status", "access_until", "updated_at")
    list_filter = ("plan", "status")
    search_fields = ("user__username", "user__email", "stripe_customer_id")


@admin.register(AIUsage)
class AIUsageAdmin(admin.ModelAdmin):
    list_display = ("user", "month", "count")
    list_filter = ("month",)
