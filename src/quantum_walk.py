import numpy as np
from qiskit import QuantumCircuit 
from qiskit.quantum_info import Statevector
from typing import Tuple, Dict, Optional
import networkx as nx


def create_coin_operator(
    qc: QuantumCircuit,
    coin_qubit: int,
    bias: float = np.pi / 4
) -> None:
    """
    Apply coin operator to create superposition of left/right directions.
    
    Args:
        qc: Quantum circuit
        coin_qubit: Index of coin qubit
        bias: Rotation angle (0 to π/2)
              - π/4 (default): Hadamard-like, symmetric walk
              - 0: Always go left
              - π/2: Always go right
    """
    qc.ry(2 * bias, coin_qubit)


def controlled_increment(
    qc: QuantumCircuit,
    control: int,
    targets: list
) -> None:
    """
    Controlled increment: add 1 to binary register when control=1.
    Implements a ripple-carry adder.
    
    Args:
        qc: Quantum circuit
        control: Control qubit
        targets: List of target qubits (LSB first)
    """
    n = len(targets)
    
    if n == 0:
        return
    
    qc.cx(control, targets[0])
    
    # Subsequent bits: multi-controlled NOT
    for i in range(1, n):
        controls = [control] + targets[:i]
        if len(controls) == 2:
            qc.ccx(controls[0], controls[1], targets[i])
        else:
            qc.mcx(controls, targets[i])


def controlled_decrement(
    qc: QuantumCircuit,
    control: int,
    targets: list
) -> None:
    """
    Controlled decrement: subtract 1 from binary register when control=1.
    
    Args:
        qc: Quantum circuit
        control: Control qubit
        targets: List of target qubits (LSB first)
    """
    n = len(targets)
    
    if n == 0:
        return
    
    for t in targets:
        qc.x(t)
    
    controlled_increment(qc, control, targets)
    
    for t in targets:
        qc.x(t)


def create_shift_operator(
    qc: QuantumCircuit,
    coin_qubit: int,
    position_qubits: list
) -> None:
    """
    Create shift operator for 1D quantum walk with periodic boundaries.
    
    When coin=|0⟩: shift left (decrement position)
    When coin=|1⟩: shift right (increment position)
    
    Args:
        qc: Quantum circuit
        coin_qubit: Index of coin qubit
        position_qubits: List of position qubit indices
    """
    controlled_increment(qc, coin_qubit, position_qubits)
    
    qc.x(coin_qubit)
    controlled_decrement(qc, coin_qubit, position_qubits)
    qc.x(coin_qubit)


def quantum_walk_1d(
    num_nodes: int = 8,
    steps: int = 5,
    start_position: Optional[int] = None,
    coin_bias: float = np.pi / 4
) -> Tuple[QuantumCircuit, Statevector]:
    """
    Implement 1D quantum walk on a line with periodic boundaries (ring).
    
    Args:
        num_nodes: Number of positions (must be power of 2 for simplicity)
        steps: Number of walk steps
        start_position: Initial position (default: center)
        coin_bias: Coin operator bias angle
        
    Returns:
        Tuple of (quantum_circuit, final_statevector)
    """
    
    n_position_qubits = int(np.ceil(np.log2(num_nodes)))
    
    actual_num_nodes = 2 ** n_position_qubits
    if actual_num_nodes != num_nodes:
        print(f"Note: Adjusting num_nodes from {num_nodes} to {actual_num_nodes} (power of 2)")
        num_nodes = actual_num_nodes
    
    if start_position is None:
        start_position = num_nodes // 2
    
    total_qubits = 1 + n_position_qubits
    qc = QuantumCircuit(total_qubits)
    
    coin_qubit = 0
    position_qubits = list(range(1, total_qubits))
    
    start_binary = format(start_position, f'0{n_position_qubits}b')
    for i, bit in enumerate(start_binary):
        if bit == '1':
            qc.x(position_qubits[i])
    
    qc.h(coin_qubit)
    
    qc.barrier(label='Init')
    
    for step in range(steps):
        create_coin_operator(qc, coin_qubit, coin_bias)
        create_shift_operator(qc, coin_qubit, position_qubits)
        qc.barrier(label=f'Step{step+1}')
    
    state = Statevector.from_instruction(qc)
    
    return qc, state


def quantum_walk_graph(
    G: nx.Graph,
    steps: int = 20,
    delta_t: float = 0.1,
    start_node: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a continuous-time quantum walk on an arbitrary graph using its adjacency matrix.
    
    Args:
        G: networkx Graph (any topology)
        steps: number of discrete time steps to simulate
        delta_t: time step size
        start_node: index of starting node
        
    Returns:
        Tuple (prob_history, adjacency_matrix)
            prob_history: array of shape (steps+1, num_nodes)
            adjacency_matrix: numpy array of graph adjacency
    """
    num_nodes = len(G.nodes)
    A = nx.to_numpy_array(G)
    
    # Define the Hamiltonian as the adjacency matrix (could also use Laplacian)
    H = A

    # Compute the time evolution operator U = exp(-i * H * t)
    def time_evolution_operator(t):
        return scipy.linalg.expm(-1j * H * t)

    # Initial state localized at start_node
    psi0 = np.zeros(num_nodes, dtype=complex)
    psi0[start_node] = 1.0

    # Track probabilities over time
    prob_history = np.zeros((steps + 1, num_nodes))
    prob_history[0] = np.abs(psi0) ** 2

    for t_step in range(1, steps + 1):
        t = t_step * delta_t
        U = time_evolution_operator(t)
        psi_t = U @ psi0
        prob_history[t_step] = np.abs(psi_t) ** 2

    return prob_history, A


def extract_position_probabilities(
    state: Statevector,
    n_position_qubits: int
) -> np.ndarray:
    """
    Extract position probabilities by marginalizing over coin state.
    
    Args:
        state: Quantum statevector
        n_position_qubits: Number of position qubits
        
    Returns:
        Array of position probabilities
    """
    
    probs_dict = state.probabilities_dict()
    
    num_positions = 2 ** n_position_qubits
    position_probs = np.zeros(num_positions)
    
    for state_str, prob in probs_dict.items():
        state_bits = state_str[::-1]
        position_bits = state_bits[1:]  # Skip coin bit
        position_index = int(position_bits[::-1], 2)
        
        position_probs[position_index] += prob
    
    return position_probs


def quantum_walk_evolution(
    num_nodes: int = 16,
    steps: int = 20,
    start_position: Optional[int] = None,
    coin_bias: float = np.pi / 4
) -> np.ndarray:
    """
    Run quantum walk and return probability distribution at each step.
    
    Args:
        num_nodes: Number of positions
        steps: Number of walk steps
        start_position: Initial position
        coin_bias: Coin operator bias
        
    Returns:
        Array of shape (steps+1, num_nodes) with probability history
    """
    n_position_qubits = int(np.ceil(np.log2(num_nodes)))
    num_nodes = 2 ** n_position_qubits
    
    if start_position is None:
        start_position = num_nodes // 2
    
    history = []
    
    initial_probs = np.zeros(num_nodes)
    initial_probs[start_position] = 1.0
    history.append(initial_probs)
    
    for step in range(1, steps + 1):
        qc, state = quantum_walk_1d(num_nodes, step, start_position, coin_bias)
        probs = extract_position_probabilities(state, n_position_qubits)
        alpha = 0.98
        uniform = np.ones_like(probs) / len(probs)
        probs = alpha * probs + (1 - alpha) * uniform
        history.append(probs)
    
    return np.array(history)


def run_quantum_simulation(
    num_nodes: int = 8,
    steps: int = 10,
    start_position: Optional[int] = None,
    coin_bias: float = np.pi / 4
) -> Tuple[np.ndarray, QuantumCircuit, dict]:
    """
    Run quantum walk and return comprehensive results.
    
    Args:
        num_nodes: Number of positions
        steps: Number of walk steps
        start_position: Initial position
        coin_bias: Coin operator bias
        
    Returns:
        Tuple of (probability_history, final_circuit, metadata)
    """
    history = quantum_walk_evolution(num_nodes, steps, start_position, coin_bias)
    
    final_circuit, _ = quantum_walk_1d(num_nodes, steps, start_position, coin_bias)
    
    metadata = {
        'method': 'quantum_walk',
        'num_nodes': num_nodes,
        'num_steps': steps,
        'start_position': start_position if start_position else num_nodes // 2,
        'coin_bias': coin_bias,
        'num_qubits': final_circuit.num_qubits,
        'circuit_depth': final_circuit.depth(),
        'final_distribution': history[-1],
        'max_probability_node': np.argmax(history[-1])
    }
    
    return history, final_circuit, metadata


def run_quantum_simulation_random(
    num_nodes: int = 8,
    steps: int = 10,
    start_position: Optional[int] = None,
    coin_bias: float = np.pi / 4
) -> Tuple[np.ndarray, QuantumCircuit, dict]:
    """
    Run quantum walk and return comprehensive results.
    
    Args:
        num_nodes: Number of positions
        steps: Number of walk steps
        start_position: Initial position
        coin_bias: Coin operator bias
        
    Returns:
        Tuple of (probability_history, final_circuit, metadata)
    """
    history = quantum_walk_evolution(num_nodes, steps, start_position, coin_bias)
    
    final_circuit, _ = quantum_walk_1d(num_nodes, steps, start_position, coin_bias)
    
    metadata = {
        'method': 'quantum_walk',
        'num_nodes': num_nodes,
        'num_steps': steps,
        'start_position': start_position if start_position else num_nodes // 2,
        'coin_bias': coin_bias,
        'num_qubits': final_circuit.num_qubits,
        'circuit_depth': final_circuit.depth(),
        'final_distribution': history[-1],
        'max_probability_node': np.argmax(history[-1])
    }
    
    return history, final_circuit, metadata

if __name__ == "__main__":
    print("Testing 1D Quantum Walk")
    print("=" * 50)
    
    num_nodes = 8
    steps = 5
    
    history, circuit, metadata = run_quantum_simulation(
        num_nodes=num_nodes,
        steps=steps,
        coin_bias=np.pi/4
    )
    
    print(f"Circuit has {metadata['num_qubits']} qubits")
    print(f"Circuit depth: {metadata['circuit_depth']}")
    print(f"\nInitial state: {history[0]}")
    print(f"Final state: {history[-1]}")
    print(f"Sum check: {np.sum(history[-1]):.6f}")
    print(f"Max probability at node: {metadata['max_probability_node']}")