from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.http import FileResponse
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/auth/", include("accounts.urls")),
    path("api/documents/", include("documents.urls")),
    path("api/agent/", include("agent.urls")),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
else:
    # In production the built React app is served from this same domain (see
    # deployment guide); /assets/, /static/ and /media/ are handled by static
    # file mappings on the host, so Django only needs to hand back index.html
    # for the root path.
    def frontend_index(request):
        index_path = settings.BASE_DIR.parent / "frontend" / "dist" / "index.html"
        return FileResponse(open(index_path, "rb"))

    urlpatterns += [path("", frontend_index)]
