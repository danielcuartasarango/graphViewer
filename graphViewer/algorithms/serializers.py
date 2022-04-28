from dataclasses import fields
from rest_framework import serializers
from algorithms.models import Graph

class GraphSerializer(serializers.ModelSerializer):
    class Meta:
        model= Graph
        fields=('source', 'target', 'label', 'ide','name','x','y')