from django.conf.urls import url
from algorithms import views


urlpatterns = [
    url(r'^graph$',views.graphApi),
    url(r'^graph/([0-9]+)$', views.graphApi)


]
