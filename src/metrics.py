import numpy as np
from typing import Dict, List, Tuple
from scipy.stats import entropy as scipy_entropy


def compute_entropy(prob_distribution: np.ndarray) -> float:
    """
    Compute Shannon entropy of probability distribution.
    Higher entropy = more spread out distribution.
    
    H = -Σ p_i * log(p_i)
    
    Args:
        prob_distribution: Probability distribution over nodes
        
    Returns:
        Shannon entropy value
    """
    probs = prob_distribution[prob_distribution > 1e-10]
    
    if len(probs) == 0:
        return 0.0
    
    return -np.sum(probs * np.log2(probs))


def compute_coverage(
    prob_distribution: np.ndarray,
    threshold: float = 0.01
) -> float:
    """
    Compute fraction of nodes with probability above threshold.
    
    Args:
        prob_distribution: Probability distribution over nodes
        threshold: Minimum probability to count as "covered"
        
    Returns:
        Coverage fraction (0 to 1)
    """
    return np.sum(prob_distribution > threshold) / len(prob_distribution)


def compute_mean_position(prob_distribution: np.ndarray) -> float:
    """
    Compute expected position (mean) of the walker.
    
    Args:
        prob_distribution: Probability distribution over nodes
        
    Returns:
        Mean position
    """
    positions = np.arange(len(prob_distribution))
    return np.sum(positions * prob_distribution)


def compute_variance(prob_distribution: np.ndarray) -> float:
    """
    Compute variance of position distribution.
    Higher variance = more spread out.
    
    Args:
        prob_distribution: Probability distribution over nodes
        
    Returns:
        Variance value
    """
    mean = compute_mean_position(prob_distribution)
    positions = np.arange(len(prob_distribution))
    return np.sum(prob_distribution * (positions - mean) ** 2)


def compute_standard_deviation(prob_distribution: np.ndarray) -> float:
    """
    Compute standard deviation of position distribution.
    
    Args:
        prob_distribution: Probability distribution over nodes
        
    Returns:
        Standard deviation
    """
    return np.sqrt(compute_variance(prob_distribution))


def compute_max_probability(prob_distribution: np.ndarray) -> Tuple[float, int]:
    """
    Find maximum probability and its position.
    
    Args:
        prob_distribution: Probability distribution over nodes
        
    Returns:
        Tuple of (max_probability, position)
    """
    max_idx = np.argmax(prob_distribution)
    return prob_distribution[max_idx], max_idx


def compute_total_variation_distance(
    dist1: np.ndarray,
    dist2: np.ndarray
) -> float:
    """
    Compute total variation distance between two distributions.
    
    TV(P,Q) = 0.5 * Σ |P_i - Q_i|
    
    Args:
        dist1: First probability distribution
        dist2: Second probability distribution
        
    Returns:
        Total variation distance (0 to 1)
    """
    return 0.5 * np.sum(np.abs(dist1 - dist2))


def compute_kl_divergence(
    dist1: np.ndarray,
    dist2: np.ndarray
) -> float:
    """
    Compute KL divergence from dist2 to dist1.
    
    KL(P||Q) = Σ P_i * log(P_i / Q_i)
    
    Args:
        dist1: Target distribution
        dist2: Reference distribution
        
    Returns:
        KL divergence
    """
    
    epsilon = 1e-10
    dist1_safe = np.clip(dist1, epsilon, 1.0)
    dist2_safe = np.clip(dist2, epsilon, 1.0)
    
    return scipy_entropy(dist1_safe, dist2_safe)


def analyze_distribution_history(
    history: np.ndarray
) -> Dict[str, List[float]]:
    """
    Compute metrics for entire history of probability distributions.
    
    Args:
        history: Array of shape (num_steps, num_nodes)
        
    Returns:
        Dictionary with metric time series
    """
    metrics = {
        'entropy': [],
        'coverage': [],
        'mean_position': [],
        'std_dev': [],
        'max_probability': []
    }
    
    for prob_dist in history:
        metrics['entropy'].append(compute_entropy(prob_dist))
        metrics['coverage'].append(compute_coverage(prob_dist))
        metrics['mean_position'].append(compute_mean_position(prob_dist))
        metrics['std_dev'].append(compute_standard_deviation(prob_dist))
        max_prob, _ = compute_max_probability(prob_dist)
        metrics['max_probability'].append(max_prob)
    
    return metrics


def compare_walks(
    classical_history: np.ndarray,
    quantum_history: np.ndarray
) -> Dict[str, any]:
    """
    Comprehensive comparison between classical and quantum walks.
    
    Args:
        classical_history: Classical probability history
        quantum_history: Quantum probability history
        
    Returns:
        Dictionary with comparison metrics
    """
    
    classical_metrics = analyze_distribution_history(classical_history)
    quantum_metrics = analyze_distribution_history(quantum_history)
    
    final_classical = classical_history[-1]
    final_quantum = quantum_history[-1]
    
    comparison = {
        'classical_metrics': classical_metrics,
        'quantum_metrics': quantum_metrics,
        
        'final_entropy_difference': (
            quantum_metrics['entropy'][-1] - classical_metrics['entropy'][-1]
        ),
        'final_coverage_difference': (
            quantum_metrics['coverage'][-1] - classical_metrics['coverage'][-1]
        ),
        'final_tv_distance': compute_total_variation_distance(
            final_quantum, final_classical
        ),
        
        'classical_entropy_growth': (
            classical_metrics['entropy'][-1] - classical_metrics['entropy'][0]
        ),
        'quantum_entropy_growth': (
            quantum_metrics['entropy'][-1] - quantum_metrics['entropy'][0]
        ),
        
        'classical_steps_to_50_coverage': _steps_to_threshold(
            classical_metrics['coverage'], 0.5
        ),
        'quantum_steps_to_50_coverage': _steps_to_threshold(
            quantum_metrics['coverage'], 0.5
        ),
    }
    
    return comparison


def _steps_to_threshold(metric_series: List[float], threshold: float) -> int:
    """Helper function to find first step reaching threshold."""
    for step, value in enumerate(metric_series):
        if value >= threshold:
            return step
    return len(metric_series)


def compute_interference_visibility(
    quantum_history: np.ndarray,
    classical_history: np.ndarray
) -> float:
    """
    Compute a measure of quantum interference visibility.
    
    High visibility = quantum distribution differs significantly from classical.
    
    Args:
        quantum_history: Quantum probability history
        classical_history: Classical probability history
        
    Returns:
        Interference visibility measure
    """
    final_quantum = quantum_history[-1]
    final_classical = classical_history[-1]
    
    visibility = compute_total_variation_distance(final_quantum, final_classical)
    
    return visibility


def generate_summary_statistics(comparison: Dict) -> str:
    """
    Generate human-readable summary of comparison.
    
    Args:
        comparison: Output from compare_walks()
        
    Returns:
        Formatted string with key findings
    """
    summary = []
    summary.append("Quantum vs Classical")
    
    # Coverage
    classical_50_cov = comparison['classical_steps_to_50_coverage']
    quantum_50_cov = comparison['quantum_steps_to_50_coverage']
    
    if quantum_50_cov < classical_50_cov:
        speedup = classical_50_cov / max(quantum_50_cov, 1)
        summary.append(f"\n✓ Quantum reaches 50% coverage {speedup:.1f}x faster")
        summary.append(f"  Classical: {classical_50_cov} steps")
        summary.append(f"  Quantum:   {quantum_50_cov} steps")
    else:
        summary.append(f"\nCoverage rates similar")
    
    # Entropy
    entropy_diff = comparison['final_entropy_difference']
    summary.append(f"\nFinal entropy difference: {entropy_diff:+.3f}")
    
    # Distribution difference
    tv_dist = comparison['final_tv_distance']
    summary.append(f"\nTotal variation distance: {tv_dist:.3f}")
    
    return "\n".join(summary)


if __name__ == "__main__":
    uniform = np.ones(8) / 8
    peaked = np.array([0.0, 0.1, 0.4, 0.4, 0.1, 0.0, 0.0, 0.0])
    
    print(f"Uniform distribution entropy: {compute_entropy(uniform):.3f}")
    print(f"Peaked distribution entropy: {compute_entropy(peaked):.3f}")
    print(f"Coverage (peaked, threshold=0.1): {compute_coverage(peaked, 0.1):.2f}")