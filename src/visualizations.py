"""
Visualization functions for quantum walk analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import networkx as nx
from typing import Optional, Tuple, Dict
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_network(
    G: nx.Graph,
    probabilities: np.ndarray,
    title: str = "Network State",
    ax: Optional[plt.Axes] = None,
    cmap: str = 'Blues',
    vmin: float = 0.0,
    vmax: float = 1.0
) -> plt.Figure:
    """
    Plot network with nodes colored by probability.
    
    Args:
        G: NetworkX graph
        probabilities: Probability at each node
        title: Plot title
        ax: Matplotlib axes (creates new if None)
        cmap: Colormap name
        vmin, vmax: Color scale limits
        
    Returns:
        Figure object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()
    
    # Layout
    pos = nx.spring_layout(G, seed=42)
    
    # Draw network
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, width=2)
    
    # Draw nodes with probability colors
    nodes = nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=probabilities,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        node_size=800,
        edgecolors='black',
        linewidths=2
    )
    
    # Labels
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=12, font_weight='bold')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Colorbar
    plt.colorbar(nodes, ax=ax, label='Probability')
    
    return fig


def plot_probability_bars(
    classical_probs: np.ndarray,
    quantum_probs: np.ndarray,
    step: int,
    figsize: Tuple[int, int] = (14, 5)
) -> plt.Figure:
    """
    Side-by-side bar chart comparison of probability distributions.
    
    Args:
        classical_probs: Classical probabilities
        quantum_probs: Quantum probabilities
        step: Current step number
        figsize: Figure size
        
    Returns:
        Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    nodes = np.arange(len(classical_probs))
    
    # Classical
    ax1.bar(nodes, classical_probs, color='steelblue', alpha=0.8, edgecolor='black')
    ax1.set_xlabel('Node', fontsize=12)
    ax1.set_ylabel('Probability', fontsize=12)
    ax1.set_title(f'Classical Walk - Step {step}', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, max(1.0, np.max(classical_probs) * 1.1)])
    ax1.grid(axis='y', alpha=0.3)
    
    # Quantum
    ax2.bar(nodes, quantum_probs, color='crimson', alpha=0.8, edgecolor='black')
    ax2.set_xlabel('Node', fontsize=12)
    ax2.set_ylabel('Probability', fontsize=12)
    ax2.set_title(f'Quantum Walk - Step {step}', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, max(1.0, np.max(quantum_probs) * 1.1)])
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def animate_diffusion(
    classical_history: np.ndarray,
    quantum_history: np.ndarray,
    G: nx.Graph,
    filename: Optional[str] = None,
    interval: int = 500
) -> animation.FuncAnimation:
    """
    Create animated comparison of classical vs quantum diffusion.
    
    Args:
        classical_history: Array of shape (steps, nodes)
        quantum_history: Array of shape (steps, nodes)
        G: NetworkX graph
        filename: If provided, save animation as GIF
        interval: Milliseconds between frames
        
    Returns:
        Animation object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Layout (fixed for all frames)
    pos = nx.spring_layout(G, seed=42)
    
    # Initialize empty plots
    nodes1 = None
    nodes2 = None
    
    def init():
        ax1.clear()
        ax2.clear()
        return []
    
    def update(frame):
        nonlocal nodes1, nodes2
        
        ax1.clear()
        ax2.clear()
        
        # Classical
        nx.draw_networkx_edges(G, pos, ax=ax1, alpha=0.3, width=2)
        nodes1 = nx.draw_networkx_nodes(
            G, pos, ax=ax1,
            node_color=classical_history[frame],
            cmap='Blues',
            vmin=0, vmax=1,
            node_size=800,
            edgecolors='black',
            linewidths=2
        )
        nx.draw_networkx_labels(G, pos, ax=ax1, font_size=10, font_weight='bold')
        ax1.set_title(f'Classical Walk - Step {frame}', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Quantum
        nx.draw_networkx_edges(G, pos, ax=ax2, alpha=0.3, width=2)
        nodes2 = nx.draw_networkx_nodes(
            G, pos, ax=ax2,
            node_color=quantum_history[frame],
            cmap='Reds',
            vmin=0, vmax=1,
            node_size=800,
            edgecolors='black',
            linewidths=2
        )
        nx.draw_networkx_labels(G, pos, ax=ax2, font_size=10, font_weight='bold')
        ax2.set_title(f'Quantum Walk - Step {frame}', fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        return [nodes1, nodes2]
    
    anim = animation.FuncAnimation(
        fig, update,
        init_func=init,
        frames=len(classical_history),
        interval=interval,
        blit=False
    )
    
    if filename:
        try:
            anim.save(filename, writer='pillow', fps=2)
            print(f"Animation saved to {filename}")
        except Exception as e:
            print(f"Could not save animation: {e}")
    
    return anim


def plot_metrics_over_time(
    classical_metrics: Dict,
    quantum_metrics: Dict,
    figsize: Tuple[int, int] = (14, 5)
) -> plt.Figure:
    """
    Plot entropy and coverage evolution over time.
    
    Args:
        classical_metrics: Classical metrics dictionary
        quantum_metrics: Quantum metrics dictionary
        figsize: Figure size
        
    Returns:
        Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    steps_c = range(len(classical_metrics['entropy']))
    steps_q = range(len(quantum_metrics['entropy']))
    
    # Entropy
    ax1.plot(steps_c, classical_metrics['entropy'], 
             'o-', linewidth=2, markersize=6, label='Classical', color='steelblue')
    ax1.plot(steps_q, quantum_metrics['entropy'],
             's-', linewidth=2, markersize=6, label='Quantum', color='crimson')
    ax1.set_xlabel('Steps', fontsize=12)
    ax1.set_ylabel('Shannon Entropy', fontsize=12)
    ax1.set_title('Entropy Evolution', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Coverage
    ax2.plot(steps_c, classical_metrics['coverage'],
             'o-', linewidth=2, markersize=6, label='Classical', color='steelblue')
    ax2.plot(steps_q, quantum_metrics['coverage'],
             's-', linewidth=2, markersize=6, label='Quantum', color='crimson')
    ax2.set_xlabel('Steps', fontsize=12)
    ax2.set_ylabel('Network Coverage', fontsize=12)
    ax2.set_title('Coverage Evolution', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    
    plt.tight_layout()
    return fig


def create_heatmap_comparison(
    classical_history: np.ndarray,
    quantum_history: np.ndarray,
    figsize: Tuple[int, int] = (14, 6)
) -> plt.Figure:
    """
    Create heatmaps showing evolution of probability over time.
    
    Args:
        classical_history: Classical probability history
        quantum_history: Quantum probability history
        figsize: Figure size
        
    Returns:
        Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Classical heatmap
    im1 = ax1.imshow(classical_history.T, aspect='auto', cmap='Blues',
                     origin='lower', interpolation='nearest')
    ax1.set_xlabel('Time Steps', fontsize=12)
    ax1.set_ylabel('Node Index', fontsize=12)
    ax1.set_title('Classical Diffusion Heatmap', fontsize=14, fontweight='bold')
    plt.colorbar(im1, ax=ax1, label='Probability')
    
    # Quantum heatmap
    im2 = ax2.imshow(quantum_history.T, aspect='auto', cmap='Reds',
                     origin='lower', interpolation='nearest')
    ax2.set_xlabel('Time Steps', fontsize=12)
    ax2.set_ylabel('Node Index', fontsize=12)
    ax2.set_title('Quantum Diffusion Heatmap', fontsize=14, fontweight='bold')
    plt.colorbar(im2, ax=ax2, label='Probability')
    
    plt.tight_layout()
    return fig


def plot_bias_experiment(
    bias_results: Dict[float, np.ndarray],
    figsize: Tuple[int, int] = (15, 4)
) -> plt.Figure:
    """
    Plot results of coin bias experiments.
    
    Args:
        bias_results: Dictionary mapping bias values to final probability distributions
        figsize: Figure size
        
    Returns:
        Figure object
    """
    n_biases = len(bias_results)
    fig, axes = plt.subplots(1, n_biases, figsize=figsize)
    
    if n_biases == 1:
        axes = [axes]
    
    for ax, (bias, probs) in zip(axes, bias_results.items()):
        nodes = np.arange(len(probs))
        ax.bar(nodes, probs, color='purple', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Node', fontsize=11)
        ax.set_ylabel('Probability', fontsize=11)
        ax.set_title(f'Bias = {bias:.2f} rad', fontsize=12, fontweight='bold')
        ax.set_ylim([0, max(probs) * 1.1])
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_summary_figure(
    classical_history: np.ndarray,
    quantum_history: np.ndarray,
    G: nx.Graph,
    classical_metrics: Dict,
    quantum_metrics: Dict,
    circuit: Optional[any] = None,
    figsize: Tuple[int, int] = (18, 10)
) -> plt.Figure:
    """
    Create comprehensive summary figure with all key visualizations.
    
    Args:
        classical_history: Classical probability history
        quantum_history: Quantum probability history
        G: NetworkX graph
        classical_metrics: Classical metrics
        quantum_metrics: Quantum metrics
        circuit: Optional quantum circuit to display
        figsize: Figure size
        
    Returns:
        Figure object
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Top row: Network visualizations
    ax1 = fig.add_subplot(gs[0, 0])
    plot_network(G, classical_history[-1], "Classical Final State", ax=ax1, cmap='Blues')
    
    ax2 = fig.add_subplot(gs[0, 1])
    plot_network(G, quantum_history[-1], "Quantum Final State", ax=ax2, cmap='Reds')
    
    # Probability bars
    ax3 = fig.add_subplot(gs[0, 2])
    nodes = np.arange(len(classical_history[-1]))
    width = 0.35
    ax3.bar(nodes - width/2, classical_history[-1], width, 
            label='Classical', color='steelblue', alpha=0.8)
    ax3.bar(nodes + width/2, quantum_history[-1], width,
            label='Quantum', color='crimson', alpha=0.8)
    ax3.set_xlabel('Node')
    ax3.set_ylabel('Probability')
    ax3.set_title('Final Distribution Comparison', fontweight='bold')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # Middle row: Metrics
    ax4 = fig.add_subplot(gs[1, 0])
    steps = range(len(classical_metrics['entropy']))
    ax4.plot(steps, classical_metrics['entropy'], 'o-', label='Classical', color='steelblue')
    ax4.plot(steps, quantum_metrics['entropy'], 's-', label='Quantum', color='crimson')
    ax4.set_xlabel('Steps')
    ax4.set_ylabel('Entropy')
    ax4.set_title('Entropy Evolution', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(steps, classical_metrics['coverage'], 'o-', label='Classical', color='steelblue')
    ax5.plot(steps, quantum_metrics['coverage'], 's-', label='Quantum', color='crimson')
    ax5.set_xlabel('Steps')
    ax5.set_ylabel('Coverage')
    ax5.set_title('Network Coverage', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Heatmap
    ax6 = fig.add_subplot(gs[1, 2])
    diff = quantum_history - classical_history
    im = ax6.imshow(diff.T, aspect='auto', cmap='RdBu_r', 
                    origin='lower', vmin=-0.3, vmax=0.3)
    ax6.set_xlabel('Time Steps')
    ax6.set_ylabel('Node')
    ax6.set_title('Probability Difference\n(Quantum - Classical)', fontweight='bold')
    plt.colorbar(im, ax=ax6)
    
    # Bottom row: Circuit and summary
    if circuit is not None:
        ax7 = fig.add_subplot(gs[2, :2])
        try:
            circuit.draw('mpl', ax=ax7)
            ax7.set_title('Quantum Circuit', fontweight='bold')
        except:
            ax7.text(0.5, 0.5, 'Circuit visualization unavailable',
                    ha='center', va='center', fontsize=12)
            ax7.axis('off')
    else:
        ax7 = fig.add_subplot(gs[2, :2])
        ax7.axis('off')
    
    # Key findings text
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    
    # Calculate key stats
    final_tv = np.sum(np.abs(quantum_history[-1] - classical_history[-1])) / 2
    q_max = np.max(quantum_history[-1])
    c_max = np.max(classical_history[-1])
    
    findings_text = f"""
    KEY FINDINGS
    {'='*30}
    
    Final TV Distance: {final_tv:.3f}
    
    Max Probability:
    • Classical: {c_max:.3f}
    • Quantum: {q_max:.3f}
    
    Final Entropy:
    • Classical: {classical_metrics['entropy'][-1]:.3f}
    • Quantum: {quantum_metrics['entropy'][-1]:.3f}
    
    Final Coverage:
    • Classical: {classical_metrics['coverage'][-1]:.2%}
    • Quantum: {quantum_metrics['coverage'][-1]:.2%}
    """
    
    ax8.text(0.1, 0.9, findings_text, fontsize=10, verticalalignment='top',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.suptitle('Quantum Diffusion Simulator - Complete Analysis',
                fontsize=16, fontweight='bold', y=0.98)
    
    return fig


# Quick plotting functions for notebook use
def quick_comparison(classical_history, quantum_history, step=-1):
    """Quick side-by-side comparison at specific step."""
    fig = plot_probability_bars(
        classical_history[step],
        quantum_history[step],
        step if step >= 0 else len(classical_history) + step
    )
    plt.show()
    return fig


def quick_animation(classical_history, quantum_history, G):
    """Quick animation without saving."""
    anim = animate_diffusion(classical_history, quantum_history, G)
    plt.show()
    return anim