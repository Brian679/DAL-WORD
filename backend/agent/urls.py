from django.urls import path

from .views import AgentActionView, ChatView, GenerateDissertationView, DissertationPlanView

urlpatterns = [
    path("<int:document_id>/action/", AgentActionView.as_view(), name="agent-action"),
    path("<int:document_id>/chat/", ChatView.as_view(), name="agent-chat"),
    path("<int:document_id>/generate-dissertation/", GenerateDissertationView.as_view(), name="generate-dissertation"),
    path("<int:document_id>/dissertation-plan/", DissertationPlanView.as_view(), name="dissertation-plan"),
]
