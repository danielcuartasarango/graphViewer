from rest_framework import serializers
from drf_writable_nested.serializers import WritableNestedModelSerializer
from .models import  Coordenates, Data, Graph, LinkedTo, Root


class LinkedToSerializer(serializers.ModelSerializer):
   
    class Meta:
        model = LinkedTo
        fields = ('nodeId', 'distance')

class CoordenatesSerializer(serializers.ModelSerializer):

    class Meta:
        model = Coordenates
        fields = ( 'x','y')


class DataSerializer(WritableNestedModelSerializer):
    linkedTo = LinkedToSerializer(many=True)
    coordenates = CoordenatesSerializer()

    

    class Meta:
        model = Data
        fields = ('ide', 'label', 'type', 'radius','linkedTo', 'coordenates')





class GraphSerializer(WritableNestedModelSerializer):
    # Direct ManyToMany relation
    data = DataSerializer(many=True)

    # Reverse FK relation
    #avatars = AvatarSerializer(many=True)

    # Direct FK relation
    #access_key = AccessKeySerializer(allow_null=True)

    class Meta:
        model = Graph
        fields = ('data', 'name')
        # ('data')


class RootSerializer(WritableNestedModelSerializer):
    # Reverse OneToOne relation
    graph = GraphSerializer()

    class Meta:
        model = Root
        fields = ( 'graph', 'generalData1', 'generalData2', 'generalData3')