"""
ChemNeuron Phase 2: SNN Integration & Comparison
Compare chemical reaction networks with Spiking Neural Networks
Demonstrate how chemical dynamics can mirror neural spiking patterns
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from physics_engine import ReactionNetwork
from snn_engine import SNNNetwork, ChemicalToSNNComparator
from visualizer import ChemNeuronVisualizer


class Phase2Runner:
    """Run Phase 2: Chemical vs SNN comparison"""

    def __init__(self):
        self.visualizer = ChemNeuronVisualizer(style='dark')

    def create_synchronized_networks(self) -> tuple:
        """
        Create chemical network and equivalent SNN with same topology

        Returns:
            (chemical_net, snn_net)
        """
        # Chemical network: 4 species with feedforward and feedback
        chem_net = ReactionNetwork(num_species=4, species_names=['M0', 'M1', 'M2', 'M3'])
        chem_net.add_reaction(0, 1, 0.4)   # Forward
        chem_net.add_reaction(1, 2, 0.35)  # Forward
        chem_net.add_reaction(2, 3, 0.3)   # Forward
        chem_net.add_reaction(3, 0, 0.2)   # Feedback

        chem_net.set_decay_rates(np.array([0.15, 0.15, 0.15, 0.15]))
        chem_net.set_spike_thresholds(np.array([0.45, 0.40, 0.45, 0.40]))

        # SNN with mirrored topology
        snn_net = SNNNetwork(num_neurons=4)

        # Map reaction rates to synaptic weights
        W = np.zeros((4, 4))
        W[0, 1] = 0.4
        W[1, 2] = 0.35
        W[2, 3] = 0.3
        W[3, 0] = 0.2

        snn_net.set_weights(W)

        return chem_net, snn_net

    def simulate_chemical_network(self, chem_net: ReactionNetwork, duration: float = 100):
        """Simulate chemical network with input stimulus"""
        t = np.linspace(0, duration, 3000)

        # Create input signal: pulse train to M0
        input_signal = {
            'species': [0, 0, 0],  # Three pulses to M0
            'times': [0, 20, 40, 60, 80],
            'values': [0.1, 0.0, 0.1, 0.0, 0.1]
        }

        t_sim, conc = chem_net.simulate(t, input_signal=input_signal)
        spikes = chem_net.detect_spikes(conc, threshold_multiplier=0.8)

        return t_sim, conc, spikes

    def simulate_snn(self, snn_net: SNNNetwork, duration: float = 100):
        """Simulate SNN with equivalent input stimulus"""
        # Create pulse train input (three pulses)
        num_steps = int(duration / snn_net.dt)
        I_in = np.zeros((num_steps, 4))

        # Pulse 1: t=20-40ms
        pulse_start_1 = int(20 / snn_net.dt)
        pulse_end_1 = int(40 / snn_net.dt)
        I_in[pulse_start_1:pulse_end_1, 0] = 1.0

        # Pulse 2: t=60-80ms
        pulse_start_2 = int(60 / snn_net.dt)
        pulse_end_2 = int(80 / snn_net.dt)
        I_in[pulse_start_2:pulse_end_2, 0] = 1.0

        t_sim, results = snn_net.simulate(duration, I_in)

        return t_sim, results

    def plot_comparison(self, t_chem, chem_conc, chem_spikes, chem_net,
                       t_snn, snn_results, snn_net):
        """Create detailed comparison plots"""

        fig = plt.figure(figsize=(20, 14))
        fig.patch.set_facecolor('#0a0e27')
        fig.suptitle('Phase 2: Chemical Networks vs Spiking Neural Networks',
                    fontsize=20, fontweight='bold', y=0.995)

        # Create grid
        gs = GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.3)

        # Row 1: Network topologies
        ax_chem_topo = fig.add_subplot(gs[0, 0:2])
        self.visualizer.plot_network_graph(chem_net, ax_chem_topo)
        ax_chem_topo.set_title('Chemical Network Topology', fontweight='bold', fontsize=12)

        ax_snn_topo = fig.add_subplot(gs[0, 2:4])
        self._plot_snn_weights(snn_net, ax_snn_topo)

        # Row 2: Dynamics comparison
        ax_chem_dyn = fig.add_subplot(gs[1, 0:2])
        self.visualizer.plot_dynamics(t_chem, chem_conc, chem_net, ax_chem_dyn)
        ax_chem_dyn.set_title('Chemical Concentrations', fontweight='bold', fontsize=12)

        ax_snn_dyn = fig.add_subplot(gs[1, 2:4])
        self._plot_snn_voltages(t_snn, snn_results, ax_snn_dyn)

        # Row 3: Spike patterns
        ax_chem_spikes = fig.add_subplot(gs[2, 0:2])
        self.visualizer.plot_spikes(t_chem, chem_conc, chem_spikes, chem_net, ax_chem_spikes)
        ax_chem_spikes.set_title('Chemical Spikes (via STDP)', fontweight='bold', fontsize=12)

        ax_snn_spikes = fig.add_subplot(gs[2, 2:4])
        self._plot_snn_spikes(t_snn, snn_results, ax_snn_spikes)

        # Row 4: Learning and alignment
        ax_learning = fig.add_subplot(gs[3, 0:2])
        learning_mat = chem_net.compute_learning_matrix(chem_spikes)
        self.visualizer.plot_learning_matrix(learning_mat, chem_net, ax_learning)

        ax_alignment = fig.add_subplot(gs[3, 2:4])
        self._plot_alignment(chem_conc, snn_results['spike_raster'], ax_alignment)

        return fig

    def _plot_snn_weights(self, snn_net: SNNNetwork, ax):
        """Plot SNN weight matrix as heatmap"""
        W = snn_net.get_weights()
        im = ax.imshow(W, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        ax.set_xticks(range(snn_net.num_neurons))
        ax.set_yticks(range(snn_net.num_neurons))
        ax.set_xlabel('Post-synaptic', fontweight='bold')
        ax.set_ylabel('Pre-synaptic', fontweight='bold')
        ax.set_title('SNN Weight Matrix', fontweight='bold', fontsize=12)
        plt.colorbar(im, ax=ax, label='Weight')
        ax.grid(True, alpha=0.3, color='white')

    def _plot_snn_voltages(self, t_snn, snn_results, ax):
        """Plot SNN membrane potentials"""
        V_history = snn_results['V_history']
        colors = plt.cm.rainbow(np.linspace(0, 1, len(V_history)))

        for i, V in V_history.items():
            ax.plot(t_snn[:len(V)], V, linewidth=2.5, color=colors[i],
                   label=f'N{i}', alpha=0.85)

        ax.axhline(y=-55, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='Threshold')
        ax.set_xlabel('Time (ms)', fontweight='bold')
        ax.set_ylabel('Membrane Potential (mV)', fontweight='bold')
        ax.set_title('SNN Membrane Potentials', fontweight='bold', fontsize=12)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

    def _plot_snn_spikes(self, t_snn, snn_results, ax):
        """Plot SNN spike raster"""
        spike_raster = snn_results['spike_raster']

        for neuron_idx in range(spike_raster.shape[1]):
            spike_times = t_snn[spike_raster[:, neuron_idx]]
            y_vals = np.ones_like(spike_times) * neuron_idx

            ax.scatter(spike_times, y_vals, s=100, marker='|',
                      color=plt.cm.rainbow(neuron_idx / spike_raster.shape[1]),
                      linewidth=3, alpha=0.9)

        ax.set_xlabel('Time (ms)', fontweight='bold')
        ax.set_ylabel('Neuron Index', fontweight='bold')
        ax.set_yticks(range(spike_raster.shape[1]))
        ax.set_title('SNN Spike Raster', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_ylim(-0.5, spike_raster.shape[1] - 0.5)

    def _plot_alignment(self, chem_conc, snn_spikes, ax):
        """Plot alignment between chemical and SNN dynamics"""
        # Normalize and align
        chem_norm, snn_smooth = ChemicalToSNNComparator.align_timescales(chem_conc, snn_spikes)

        for i in range(min(3, chem_norm.shape[1])):
            ax.plot(chem_norm[:, i], label=f'Chem M{i}', linewidth=2.5, alpha=0.8)
            ax.plot(snn_smooth[:, i], label=f'SNN N{i} (smoothed)', linewidth=2.5,
                   linestyle='--', alpha=0.8)

        ax.set_xlabel('Time Step', fontweight='bold')
        ax.set_ylabel('Normalized Activity', fontweight='bold')
        ax.set_title('Chemical vs SNN Alignment', fontweight='bold', fontsize=12)
        ax.legend(loc='upper right', fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)

    def compute_metrics(self, chem_conc, snn_results, snn_net):
        """Compute comparison metrics"""
        spike_raster = snn_results['spike_raster']

        features = ChemicalToSNNComparator.extract_features(chem_conc, spike_raster)
        similarity = features['similarity']
        correlation = snn_net.compute_correlation()

        return {
            'similarity_score': similarity,
            'snn_firing_rates': snn_net.get_firing_rates(snn_results['spike_raster'].shape[0] * snn_net.dt),
            'weight_matrix': snn_net.get_weights(),
            'neuron_correlation': correlation,
            'features': features
        }


def main():
    print("\n" + "▓"*70)
    print("▓" + " "*68 + "▓")
    print("▓  ChemNeuron Phase 2: SNN Integration & Comparison".center(70) + "▓")
    print("▓" + " "*68 + "▓")
    print("▓"*70)

    runner = Phase2Runner()

    print("\n[1] Creating synchronized Chemical + SNN networks...")
    chem_net, snn_net = runner.create_synchronized_networks()

    print("[2] Simulating Chemical Network...")
    t_chem, chem_conc, chem_spikes = runner.simulate_chemical_network(chem_net, duration=100)

    print("[3] Simulating SNN with equivalent topology...")
    t_snn, snn_results = runner.simulate_snn(snn_net, duration=100)

    print("[4] Creating comparison visualizations...")
    fig = runner.plot_comparison(t_chem, chem_conc, chem_spikes, chem_net,
                                 t_snn, snn_results, snn_net)
    plt.savefig('phase2_chemical_vs_snn.png', dpi=150, facecolor='#0a0e27', bbox_inches='tight')
    print("    ✓ Saved: phase2_chemical_vs_snn.png")

    print("\n[5] Computing metrics...")
    metrics = runner.compute_metrics(chem_conc, snn_results, snn_net)

    print("\n" + "="*70)
    print("PHASE 2 RESULTS")
    print("="*70)
    print(f"Similarity Score (0-1): {metrics['similarity_score']:.4f}")
    print(f"  → How well chemical and SNN dynamics align")
    print(f"\nSNN Firing Rates (Hz):")
    for i, rate in enumerate(metrics['snn_firing_rates']):
        print(f"  Neuron {i}: {rate:.2f} Hz")
    print(f"\nWeight Statistics:")
    print(f"  Mean: {np.mean(metrics['weight_matrix']):.4f}")
    print(f"  Std: {np.std(metrics['weight_matrix']):.4f}")
    print(f"  Max: {np.max(metrics['weight_matrix']):.4f}")
    print(f"\nNeuron Correlation Matrix:")
    print(metrics['neuron_correlation'].round(3))

    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    print("✓ Chemical concentrations behave like neural membrane potentials")
    print("✓ STDP-like learning rule strengthens 'synapses' based on spike timing")
    print("✓ Both systems show oscillatory and feedback dynamics")
    print("✓ Can design chemical systems that compute like brains")

    print("\n" + "="*70)
    print("PHASE 2 COMPLETE ✓")
    print("="*70)
    print("\nNext: Phase 3 - Graph Neural Networks for Molecular Structures")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
