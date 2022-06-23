import pyphi
import numpy as np
tpm = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0]    
])
cm = np.array([
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
])

labels = ('A', 'B', 'C')
network = pyphi.Network(tpm, cm=cm, node_labels=labels)
state = (1, 0, 0)
node_indices = (0, 1, 2)
subsystem = pyphi.Subsystem(network, state, node_indices)


print(pyphi.connectivity.block_cm(cm))
