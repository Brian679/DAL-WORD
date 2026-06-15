from django.urls import path

from .views import AgentActionView, ChatView, GenerateDissertationView, DissertationPlanView, ResearchWorkflowView, AIDetectView, AcademicQualityView

urlpatterns = [
    path("<int:document_id>/action/", AgentActionView.as_view(), name="agent-action"),
    path("<int:document_id>/chat/", ChatView.as_view(), name="agent-chat"),
    path("<int:document_id>/research-workflow/", ResearchWorkflowView.as_view(), name="agent-research-workflow"),
    path("<int:document_id>/generate-dissertation/", GenerateDissertationView.as_view(), name="generate-dissertation"),
    path("<int:document_id>/dissertation-plan/", DissertationPlanView.as_view(), name="dissertation-plan"),
    path("<int:document_id>/ai-detect/", AIDetectView.as_view(), name="agent-ai-detect"),
    path("<int:document_id>/academic-quality/", AcademicQualityView.as_view(), name="agent-academic-quality"),
]
