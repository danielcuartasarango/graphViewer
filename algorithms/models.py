
from cProfile import label
from django.db import models

class LinkedTo(models.Model):
    nodeId = models.IntegerField()
    distance = models.IntegerField()

class Coordenates(models.Model):
    x = models.IntegerField()
    y = models.IntegerField()

class Data(models.Model):
    ide = models.IntegerField()
    label = models.CharField(max_length=100)
    type = models.CharField(max_length=100)
    radius = models.FloatField()
    
    linkedTo = models.ManyToManyField(LinkedTo)
    coordenates = models.ForeignKey(Coordenates, null=True, on_delete=models.CASCADE)
   

class Graph(models.Model):
    name = models.CharField(max_length=100)
    data = models.ManyToManyField(Data)
   # root = models.OneToOneField(Root, on_delete=models.CASCADE)

class Root(models.Model):
    generalData1 = models.IntegerField()
    generalData2 = models.CharField(max_length=100)
    generalData3 = models.IntegerField()
    graph = models.OneToOneField(Graph, on_delete=models.CASCADE)










"""
class Site(models.Model):
    url = models.CharField(max_length=100)

class User(models.Model):
    username = models.CharField(max_length=100)

class AccessKey(models.Model):
    key = models.CharField(max_length=100)

class Profile(models.Model):
    sites = models.ManyToManyField(Site)
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    access_key = models.ForeignKey(AccessKey, null=True, on_delete=models.CASCADE)

class Avatar(models.Model):
    image = models.CharField(max_length=100)
    profile = models.ForeignKey(Profile, related_name='avatars', on_delete=models.CASCADE)"""