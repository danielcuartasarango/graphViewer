from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from rest_framework.parsers import JSONParser
from django.http.response import JsonResponse

from algorithms.models import Graph
from algorithms.serializers import GraphSerializer


@csrf_exempt
def graphApi(request, id=0):
    if request.method=='GET':
        graph = Graph.objects.all()
        graph_serializer = GraphSerializer(graph,many=True)
        return JsonResponse(graph_serializer.data, safe=False)
    elif request.method=='POST':
        graph_data=JSONParser().parse(request)
        graph_serializer=GraphSerializer(data=graph_data)
        if graph_serializer.is_valid():
            graph_serializer.save()
            return JsonResponse("Added Successfully", safe= False)
        return JsonResponse("Failed to Add")
    elif request.method=='PUT':
        graph_data = JSONParser().parse(request)
        graph = Graph.objects.get(ide=graph_data['ide'])
        graph_serializer=GraphSerializer(graph, data=graph_data)
        if graph_serializer.is_valid():
            graph_serializer.save()
            return JsonResponse("Update Successfully", safe=False)
        return JsonResponse("Failded to Update")
    elif request.method=='DELETE':
        graph = Graph.objects.get(id=id)
        graph.delete()
        return JsonResponse("Deleted Successfully", safe=False)





