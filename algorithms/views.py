import json
from tokenize import String
from django.shortcuts import render

from django.http.response import JsonResponse
from rest_framework.parsers import JSONParser
from rest_framework import status
from sklearn.metrics import mutual_info_score

from .models import    Root
from .serializers import  RootSerializer
from .algoritmos import *
from .procedures import *
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

@api_view(['GET', 'PUT', 'DELETE'])
def root_detail(request, ide):
    try: 
        root = Root.objects.get(id=ide) 
    except Root.DoesNotExist: 
        return JsonResponse({'message': 'The graph does not exist'}, status=status.HTTP_404_NOT_FOUND) 
   
    if request.method == 'GET': 
        root_serializer = RootSerializer(root) 
        return JsonResponse(root_serializer.data)
    elif request.method == 'GETI': 
        root_serializer = RootSerializer(root) 
        
        m = matriz_ad(root_serializer.data)
        
        x= m
        y= np.zeros((len(m),len(m)))
        
        f = lambda x, params : mutual_info_score(np.shape(x),np.shape(y))

        subset_opt, partition_value, cluster_max = QUEYRANNE(m,f)
        print(subset_opt, partition_value, cluster_max)
        a = root_serializer.data
        cut = {'generalData1':subset_opt}
        a.update(cut)
        
        print(a)

        return JsonResponse(a)  
    elif request.method == 'PUT': 
        root_data = JSONParser().parse(request) 
        root_serializer = RootSerializer(root, data=root_data) 
        if root_serializer.is_valid(): 
            root_serializer.save() 
            return JsonResponse(root_serializer.data) 
        return JsonResponse(root_serializer.errors, status=status.HTTP_400_BAD_REQUEST) 
    elif request.method == 'DELETE': 
        root.delete() 
        return JsonResponse({'message': 'Graph was deleted successfully!'}, status=status.HTTP_204_NO_CONTENT)
@api_view(['GET'])
def root_detail_pruebas(request, ide):
    try: 
        root = Root.objects.get(id=ide) 
    except Root.DoesNotExist: 
        return JsonResponse({'message': 'The graph does not exist'}, status=status.HTTP_404_NOT_FOUND) 
   
    if request.method == 'GET': 
        
        root_serializer = RootSerializer(root) 
        
        m = matriz_ad(root_serializer.data)
        
        x= m
        y= np.zeros((len(m),len(m)))
        
        f = lambda x, params : mutual_info_score(np.shape(x),np.shape(y))

        subset_opt, partition_value, cluster_max = QUEYRANNE(m,f)
        print(subset_opt, partition_value, cluster_max)
        a = root_serializer.data
        cut = {'generalData1':subset_opt}
        a.update(cut)
        
        return JsonResponse(a) 