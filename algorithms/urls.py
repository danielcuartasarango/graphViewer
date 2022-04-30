
from algorithms import views
from django.urls import  re_path

urlpatterns = [
    re_path(r'^api/root$', views.root),
    re_path(r'^api/root/(?P<ide>[0-9]+)$', views.root_detail)
   # url(r'^api/tutorials/published$', views.tutorial_list_published)
]