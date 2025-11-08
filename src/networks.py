import networkx as nx
import numpy as np
from typing import Dict, Tuple

"""
Network topology creation and management functions
"""

def create_ring_graph(n: int = 8) -> nx.Graph:
    """
    Create a ring (cycle) graph where each node connects to 2 neighbors.
    
    Args:
        n: Number of nodes
        
    Returns:
        NetworkX graph object
    """
    G = nx.cycle_graph(n)
    G.name = f"Ring Graph (n={n})"
    return G


def create_star_graph(n: int = 8) -> nx.Graph:
    """
    Create a star graph with one central hub and n-1 leaf nodes.
    
    Args:
        n: Total number of nodes (hub + leaves)
        
    Returns:
        NetworkX graph object
    """
    G = nx.star_graph(n - 1)  # star_graph(k) creates k+1 nodes
    G.name = f"Star Graph (n={n})"
    return G


def create_random_graph(n: int = 8, p: float = 0.3, seed: int = 42) -> nx.Graph:
    """
    Create an Erdős-Rényi random graph.
    
    Args:
        n: Number of nodes
        p: Probability of edge creation
        seed: Random seed for reproducibility
        
    Returns:
        NetworkX graph object
    """
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    
    # Ensure connectivity
    if not nx.is_connected(G):
        # Add edges to make connected
        components = list(nx.connected_components(G))
        for i in range(len(components) - 1):
            node1 = list(components[i])[0]
            node2 = list(components[i + 1])[0]
            G.add_edge(node1, node2)
    
    G.name = f"Random Graph (n={n}, p={p})"
    return G


def get_adjacency_matrix(G: nx.Graph) -> np.ndarray:
    """
    Convert NetworkX graph to adjacency matrix.
    
    Args:
        G: NetworkX graph
        
    Returns:
        Adjacency matrix as numpy array
    """
    return nx.to_numpy_array(G)


def get_transition_matrix(G: nx.Graph) -> np.ndarray:
    """
    Create stochastic transition matrix for classical random walk.
    Each row sums to 1 (probabilities of moving to neighbors).
    
    Args:
        G: NetworkX graph
        
    Returns:
        Transition matrix as numpy array
    """
    A = get_adjacency_matrix(G)
    
    # Degree matrix (diagonal with node degrees)
    degrees = A.sum(axis=1)
    
    # Avoid division by zero for isolated nodes
    degrees[degrees == 0] = 1
    
    # Create stochastic matrix: P[i,j] = A[i,j] / degree(i)
    D_inv = np.diag(1.0 / degrees)
    P = D_inv @ A
    
    return P


def get_graph_properties(G: nx.Graph) -> Dict[str, any]:
    """
    Compute basic properties of the graph.
    
    Args:
        G: NetworkX graph
        
    Returns:
        Dictionary with graph properties
    """
    return {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'avg_degree': np.mean([d for n, d in G.degree()]),
        'diameter': nx.diameter(G) if nx.is_connected(G) else float('inf'),
        'avg_clustering': nx.average_clustering(G),
        'is_connected': nx.is_connected(G)
    }


def map_to_line_positions(G: nx.Graph) -> Dict[int, int]:
    """
    Map graph nodes to positions on a 1D line for quantum walk.
    For ring graphs, this is a natural circular mapping.
    
    Args:
        G: NetworkX graph
        
    Returns:
        Dictionary mapping node IDs to line positions
    """
    n = G.number_of_nodes()
    return {node: idx for idx, node in enumerate(sorted(G.nodes()))}


# Preset network configurations for experiments
PRESET_NETWORKS = {
    'ring_8': lambda: create_ring_graph(8),
    'star_8': lambda: create_star_graph(8),
    'random_8': lambda: create_random_graph(8, 0.3),
    'ring_16': lambda: create_ring_graph(16),
}


def get_preset_network(name: str) -> nx.Graph:
    """
    Get a preset network configuration.
    
    Args:
        name: Name of preset ('ring_8', 'star_8', 'random_8', 'ring_16')
        
    Returns:
        NetworkX graph object
    """
    if name not in PRESET_NETWORKS:
        raise ValueError(f"Unknown preset: {name}. Available: {list(PRESET_NETWORKS.keys())}")
    
    return PRESET_NETWORKS[name]()