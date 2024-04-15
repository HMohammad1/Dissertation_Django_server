from django.urls import path
from myapp.views import Create, Distance, NearbyTransport, PredictTransport, OnBusRoute, FetchBusData
from . import views

# The paths that are appended to the NGROK url and what code to call
urlpatterns = [
    # path("home/<lat1>/<lon1>/<lat2>/<lon2>/", views.home, name="home"),
    path('', Create.as_view()),
    path('distance/', Distance.as_view()),
    path('transport/', NearbyTransport.as_view()),
    path('predict/', PredictTransport.as_view()),
    path('busRoute/', OnBusRoute.as_view()),
    path('busData/', FetchBusData.as_view())
]