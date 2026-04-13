"""
ChemNeuron Phase 1: Reaction Network Generator + Physics Simulator + Visualization
Demonstrates chemical reaction networks with spiking dynamics
"""

import numpy as np
import matplotlib.pyplot as plt
from backend.physics_engine import ReactionNetwork
from backend.visualizer import ChemNeuronVisualizer
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))


def create_demo_network_simple() -> ReactionNetwork:
    """Create a simple oscillatory network"""
    net = ReactionNetwork(num_species=3, species_names=['M0', 'M1', 'M2'])

    # A -> B -> C -> A (oscillatory loop)
    net.add_reaction(0, 1, 0.4)
    net.add_reaction(1, 2, 0.35)
    net.add_reaction(2, 0, 0.3)

    net.set_decay_rates(np.array([0.15, 0.15, 0.15]))
    net.set_spike_thresholds(np.array([0.4, 0.4, 0.4]))

    return net


def create_demo_network_complex() -> ReactionNetwork:
    """Create a more complex network with multiple feedback loops"""
    net = ReactionNetwork(num_species=5, species_names=['M0', 'M1', 'M2', 'M3', 'M4'])

    # Main pathway
    net.add_reaction(0, 1, 0.35)
    net.add_reaction(1, 2, 0.30)
    net.add_reaction(2, 3, 0.25)
    net.add_reaction(3, 4, 0.20)

    # Feedback loops
    net.add_reaction(3, 0, 0.15)  # Negative feedback
    net.add_reaction(2, 1, 0.1)   # Cross inhibition
    net.add_reaction(4, 2, 0.12)  # Stabilization

    net.set_decay_rates(np.array([0.12, 0.13, 0.14, 0.15, 0.16]))
    net.set_spike_thresholds(np.array([0.45, 0.40, 0.50, 0.35, 0.40]))

    return net


def simulate_with_learning(network: ReactionNetwork, num_iterations: int = 3):
    """
    Simulate network learning through STDP

    Show how reaction rates evolve with learning
    """
    print("\n" + "="*70)
    print("CHEMNEURON PHASE 1: Reaction Network Learning Simulation")
    print("="*70)

    initial_reactions = [r.copy() for r in network.reactions]
    learning_rate = 0.02

    for iteration in range(num_iterations):
        print(f"\n--- Learning Iteration {iteration + 1}/{num_iterations} ---")

        # Simulate
        t = np.linspace(0, 60, 2000)
        t_sim, concentrations = network.simulate(t)

        # Detect spikes
        spikes = network.detect_spikes(concentrations, threshold_multiplier=0.8)

        # Count spikes
        spike_counts = {i: len(spikes[i]) for i in range(network.num_species)}
        print(f"Spike counts: {spike_counts}")

        # Compute learning
        learning_matrix = network.compute_learning_matrix(spikes, tau_stdp=50)

        # Show learning matrix
        print(f"Learning matrix norm: {np.linalg.norm(learning_matrix):.4f}")

        # Apply learning
        network.apply_learning(learning_matrix, learning_rate=learning_rate)

        # Show updated rates
        print(f"Updated reaction rates:")
        for i, rxn in enumerate(network.reactions):
            delta = rxn['rate'] - initial_reactions[i]['rate']
            print(f"  {network.species_names[rxn['source']]} → "
                  f"{network.species_names[rxn['target']]}: "
                  f"{rxn['rate']:.4f} (Δ={delta:+.4f})")

    return t_sim, concentrations, spikes


def main():
    print("\n" + "▓"*70)
    print("▓" + " "*68 + "▓")
    print("▓  ChemNeuron: Neural Networks Through Chemistry".center(70) + "▓")
    print("▓  Phase 1: Reaction Network Generation & Spiking Dynamics".center(70) + "▓")
    print("▓" + " "*68 + "▓")
    print("▓"*70)

    # Create networks
    print("\n[1] Creating demo networks...")
    net_simple = create_demo_network_simple()
    net_complex = create_demo_network_complex()

    # Simulate simple network
    print("\n[2] Simulating simple oscillatory network...")
    t_simple = np.linspace(0, 80, 3000)
    t_sim_s, conc_s = net_simple.simulate(t_simple)
    spikes_s = net_simple.detect_spikes(conc_s)

    # Simulate complex network with learning
    print("\n[3] Simulating complex network with STDP learning...")
    t_complex, conc_complex, spikes_c = simulate_with_learning(net_complex, num_iterations=2)

    # Visualize
    print("\n[4] Creating visualizations...")
    visualizer = ChemNeuronVisualizer(style='dark')

    # Figure 1: Simple network analysis
    print("    - Simple network analysis...")
    fig1 = plt.figure(figsize=(18, 12))
    fig1.patch.set_facecolor('#0a0e27')
    fig1.suptitle('Simple Oscillatory Network - ChemNeuron Phase 1',
                  fontsize=18, fontweight='bold', y=0.98)

    ax1 = plt.subplot(2, 3, 1)
    visualizer.plot_network_graph(net_simple, ax1)

    ax2 = plt.subplot(2, 3, 2)
    visualizer.plot_dynamics(t_sim_s, conc_s, net_simple, ax2)

    ax3 = plt.subplot(2, 3, 3)
    visualizer.plot_spikes(t_sim_s, conc_s, spikes_s, net_simple, ax3)

    learning_mat_s = net_simple.compute_learning_matrix(spikes_s)
    ax4 = plt.subplot(2, 3, 4)
    visualizer.plot_learning_matrix(learning_mat_s, net_simple, ax4)

    ax5 = plt.subplot(2, 3, 5)
    visualizer.plot_phase_space(conc_s, 0, 1, net_simple, ax5)

    # Power spectrum
    ax6 = plt.subplot(2, 3, 6)
    from scipy import signal
    freqs, power = signal.welch(conc_s[:, 0], fs=3000/80)
    ax6.semilogy(freqs[1:100], power[1:100], linewidth=2, color='#00d9ff')
    ax6.set_xlabel('Frequency (Hz)', fontweight='bold')
    ax6.set_ylabel('Power Spectral Density', fontweight='bold')
    ax6.set_title('Power Spectrum (M0)', fontweight='bold')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('phase1_simple_network.png', dpi=150, facecolor='#0a0e27', bbox_inches='tight')
    print("    - Saved: phase1_simple_network.png")

    # Figure 2: Complex network with learning
    print("    - Complex network with learning...")
    fig2 = plt.figure(figsize=(18, 12))
    fig2.patch.set_facecolor('#0a0e27')
    fig2.suptitle('Complex Network with STDP Learning - ChemNeuron Phase 1',
                  fontsize=18, fontweight='bold', y=0.98)

    ax1 = plt.subplot(2, 3, 1)
    visualizer.plot_network_graph(net_complex, ax1)

    ax2 = plt.subplot(2, 3, 2)
    visualizer.plot_dynamics(t_complex, conc_complex, net_complex, ax2)

    ax3 = plt.subplot(2, 3, 3)
    visualizer.plot_spikes(t_complex, conc_complex, spikes_c, net_complex, ax3)

    learning_mat_c = net_complex.compute_learning_matrix(spikes_c)
    ax4 = plt.subplot(2, 3, 4)
    visualizer.plot_learning_matrix(learning_mat_c, net_complex, ax4)

    ax5 = plt.subplot(2, 3, 5)
    visualizer.plot_phase_space(conc_complex, 0, 2, net_complex, ax5)

    # Reaction rate changes
    ax6 = plt.subplot(2, 3, 6)
    rates = [r['rate'] for r in net_complex.reactions]
    ax6.bar(range(len(rates)), rates, color='#ff006e', alpha=0.8, edgecolor='#00d9ff', linewidth=2)
    ax6.set_xlabel('Reaction Index', fontweight='bold')
    ax6.set_ylabel('Rate Constant', fontweight='bold')
    ax6.set_title('Current Reaction Rates', fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('phase1_complex_network.png', dpi=150, facecolor='#0a0e27', bbox_inches='tight')
    print("    - Saved: phase1_complex_network.png")

    # Interactive dashboard
    print("    - Creating interactive dashboard...")
    visualizer.create_interactive_dashboard(
        t_sim_s, conc_s, spikes_s, net_simple,
        save_path='phase1_dashboard.html'
    )

    # Statistics
    print("\n[5] Network Statistics:")
    print(f"    Simple network: {net_simple.get_stats()}")
    print(f"    Complex network: {net_complex.get_stats()}")

    print("\n" + "="*70)
    print("PHASE 1 COMPLETE ✓")
    print("="*70)
    print("\nOutputs generated:")
    print("  • phase1_simple_network.png - Simple network visualization")
    print("  • phase1_complex_network.png - Complex network with learning")
    print("  • phase1_dashboard.html - Interactive Plotly dashboard")
    print("\nNext: Phase 2 - Spiking Neural Network Integration")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
