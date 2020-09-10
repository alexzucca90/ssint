from neal import SimulatedAnnealingSampler
import networkx as nx
import dimod
import numpy as np


# ------- Set up our graph -------

# Create empty graph
G = nx.Graph()

# Add edges to the graph (also adds nodes)
G.add_edges_from([(1, 2),
                  (1, 3),
                  (2, 4),
                  (3, 4),
                  (3, 5),
                  (4, 5)])

# ------- Set up our QUBO -------

# Initialize our Q matrix
Q = np.zeros((len(G.nodes), len(G.nodes)))

# Update Q matrix for every edge in the graph
for i, j in G.edges:
    Q[i, i] += 1
    Q[j, j] += 1
    Q[i, j] += 2


qubo = dimod.BinaryQuadraticModel(Q)

# ------- Run our QUBO on SA -------
# Set up SA parameters
numruns = 1

# Run the QUBO on the solver from your config file
sampler = SimulatedAnnealingSampler()
response = sampler.sample_qubo(qubo, num_reads=numruns)
energies = iter(response.data())

print(response)
