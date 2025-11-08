"""
Classical random walk implementation for baseline comparison
"""

import numpy as np
import networkx as nx
from typing import Tuple, Optional
from .networks import get_transition_matrix


def classical_diffusion(
    G: nx.Graph,
    steps: int = 10,
    start_node: int = 0
) -> np.ndarray:
    """
    Perform classical random walk diffusion on a graph.
    
    At each step, probability spreads equally to all neighbors.
    This is equivalent to repeated matrix multiplication with
    the transition matrix.
    
    Args:
        G: NetworkX graph
        steps: Number of diffusion steps
        start_node: Starting node for the walker
        
    Returns:
        Array of shape (steps+1, num_nodes) containing probability
        distribution at each timestep
    """
    n = G.number_of_nodes()
    
    # Get transition matrix
    P = get_transition_matrix(G)
    
    # Initialize probability distribution
    prob = np.zeros(n)
    prob[start_node] = 1.0
    
    # Store history
    history = [prob.copy()]
    
    # Evolve the distribution
    for step in range(steps):
        prob = P @ prob
        
        # Normalize to handle numerical errors
        prob = prob / np.sum(prob)
        
        history.append(prob.copy())
    
    return np.array(history)


def classical_diffusion_with_damping(
    G: nx.Graph,
    steps: int = 10,
    start_node: int = 0,
    damping: float = 0.85
) -> np.ndarray:
    """
    Classical random walk with damping (similar to PageRank).
    
    At each step, with probability (1-damping), the walker teleports
    to a random node. This prevents getting stuck and speeds convergence.
    
    Args:
        G: NetworkX graph
        steps: Number of steps
        start_node: Starting node
        damping: Damping factor (0 to 1). Higher = more random walk behavior
        
    Returns:
        Array of shape (steps+1, num_nodes)
    """
    n = G.number_of_nodes()
    P = get_transition_matrix(G)
    
    # Damped transition matrix
    # P_damped = damping * P + (1-damping) * (1/n) * ones_matrix
    teleport = (1 - damping) / n * np.ones((n, n))
    P_damped = damping * P + teleport
    
    # Initialize
    prob = np.zeros(n)
    prob[start_node] = 1.0
    history = [prob.copy()]
    
    # Evolve
    for step in range(steps):
        prob = P_damped @ prob
        prob = prob / np.sum(prob)
        history.append(prob.copy())
    
    return np.array(history)


def run_classical_simulation(
    G: nx.Graph,
    steps: int = 10,
    start_node: int = 0,
    damping: Optional[float] = None
) -> Tuple[np.ndarray, dict]:
    """
    Run classical random walk and return results with metadata.
    
    Args:
        G: NetworkX graph
        steps: Number of steps
        start_node: Starting node
        damping: Optional damping factor
        
    Returns:
        Tuple of (probability_history, metadata_dict)
    """
    if damping is not None:
        history = classical_diffusion_with_damping(G, steps, start_node, damping)
        method = f"classical_damped_{damping}"
    else:
        history = classical_diffusion(G, steps, start_node)
        method = "classical"
    
    metadata = {
        'method': method,
        'num_nodes': G.number_of_nodes(),
        'num_steps': steps,
        'start_node': start_node,
        'final_distribution': history[-1],
        'max_probability_node': np.argmax(history[-1])
    }
    
    return history, metadata


def hitting_time_classical(
    G: nx.Graph,
    start_node: int,
    target_node: int,
    max_steps: int = 1000,
    threshold: float = 0.1
) -> int:
    """
    Estimate hitting time: steps until target node probability exceeds threshold.
    
    Args:
        G: NetworkX graph
        start_node: Starting node
        target_node: Target node to reach
        max_steps: Maximum steps to simulate
        threshold: Probability threshold to consider "reached"
        
    Returns:
        Number of steps to reach target (or max_steps if not reached)
    """
    history = classical_diffusion(G, max_steps, start_node)
    
    for step, prob_dist in enumerate(history):
        if prob_dist[target_node] >= threshold:
            return step
    
    return max_steps


def mixing_time_classical(
    G: nx.Graph,
    start_node: int = 0,
    max_steps: int = 1000,
    epsilon: float = 0.01
) -> int:
    """
    Estimate mixing time: steps until distribution is close to uniform.
    
    Args:
        G: NetworkX graph
        start_node: Starting node
        max_steps: Maximum steps to simulate
        epsilon: Distance threshold from uniform distribution
        
    Returns:
        Number of steps to reach mixing (or max_steps if not reached)
    """
    n = G.number_of_nodes()
    uniform = np.ones(n) / n
    
    history = classical_diffusion(G, max_steps, start_node)
    
    for step, prob_dist in enumerate(history):
        # Total variation distance from uniform
        tv_distance = 0.5 * np.sum(np.abs(prob_dist - uniform))
        
        if tv_distance <= epsilon:
            return step
    
    return max_steps


# Example usage and testing
if __name__ == "__main__":
    # Test with a simple ring graph
    from .networks import create_ring_graph
    
    G = create_ring_graph(8)
    history = classical_diffusion(G, steps=10, start_node=0)
    
    print(f"Classical walk on {G.name}")
    print(f"Initial: {history[0]}")
    print(f"Step 5: {history[5]}")
    print(f"Step 10: {history[10]}")
    print(f"Sum check: {np.sum(history[10]):.6f}")