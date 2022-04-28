from django.db import models

class Graph(models.Model):
    source = models.IntegerField()
    target = models.IntegerField()
    label = models.CharField(max_length=200)
    ide = models.IntegerField()
    name = models.CharField(max_length=200)
    x = models.IntegerField()
    y = models.IntegerField()



