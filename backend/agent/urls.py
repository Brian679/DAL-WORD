from django.urls import path

from .views import AgentActionView, ChatView, GenerateDissertationView

urlpatterns = [
    path("<int:document_id>/action/", AgentActionView.as_view(), name="agent-action"),
    path("<int:document_id>/chat/", ChatView.as_view(), name="agent-chat"),
    path("<int:document_id>/generate-dissertation/", GenerateDissertationView.as_view(), name="generate-dissertation"),
]
