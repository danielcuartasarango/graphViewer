import numpy as np

def matriz_ad(data):


    graph = data.get("graph")
    data2 = graph.get("data")

    states = [nodes.get("ide") for nodes in data2]
    numNodes = len(states)
    matriz = np.zeros((numNodes, numNodes))
    
    
    x = 1
    for key in data2:
       
        link = key.get("linkedTo")
        ye = [nodes.get("nodeId") for nodes in link]
        if len(ye)>0:
            for k in ye:
                print()
                y = k
                matriz[x-1][y-1] = 1
                #matriz[y-1][x-1]=1
        x = x+1
       
    return matriz
