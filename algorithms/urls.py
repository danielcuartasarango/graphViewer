from django.conf.urls import url
from algorithms import views

urlpatterns = [
    url(r'^api/root$', views.root),
]