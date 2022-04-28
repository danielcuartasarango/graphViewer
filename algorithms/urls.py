
from algorithms import views
from django.urls import  re_path

urlpatterns = [
    re_path(r'^api/root$', views.root),
]