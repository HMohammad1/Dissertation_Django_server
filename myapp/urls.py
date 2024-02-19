from django.urls import path
from myapp.views import Create, Distance, NearbyTransport, PredictTransport
from . import views

urlpatterns = [
    # path("home/<lat1>/<lon1>/<lat2>/<lon2>/", views.home, name="home"),
    path('', Create.as_view()),
    path('distance/', Distance.as_view()),
    path('transport/', NearbyTransport.as_view()),
    path('predict/', PredictTransport.as_view())
]