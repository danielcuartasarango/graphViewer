from django.shortcuts import render

from django.http.response import JsonResponse
from rest_framework.parsers import JSONParser
from rest_framework import status

from .models import    Root
from .serializers import  RootSerializer
from rest_framework.decorators import api_view

@api_view(['GET', 'POST', 'DELETE'])
def root(request):
    if request.method == 'GET':
        root = Root.objects.all()
        root_serializer = RootSerializer(root, many=True)
        return JsonResponse(root_serializer.data, safe=False)
    elif request.method == 'POST':
        root_data = JSONParser().parse(request)
        root_serializer = RootSerializer( data=root_data)
        if root_serializer.is_valid():
            root_serializer.save()
            return JsonResponse(root_serializer.data, status=status.HTTP_201_CREATED)
        return JsonResponse(root_serializer.errors, status=status.HTTP_400_BAD_REQUEST)




