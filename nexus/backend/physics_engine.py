"""
ChemNeuron Physics Engine
Simulates chemical reaction networks using ODEs with spiking dynamics
"""

import numpy as np
from scipy.integrate import odeint
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class ReactionNetwork:
    """Represents a chemical reaction network with dynamics"""

    def __init__(self, num_species: int = 5, species_names: Optional[List[str]] = None):
        """
        Initialize reaction network

        Args:
            num_species: Number of chemical species
            species_names: Optional list of species names
        """
        self.num_species = num_species
        self.species_names = species_names or [f"M{i}" for i in range(num_species)]

        # Reaction matrix: (i, j, rate, type)
        # i, j: species indices
        # rate: reaction rate constant
        # type: 'decay', 'production', 'interaction'
        self.reactions = []

        # Decay rates for each species
        self.decay_rates = np.ones(num_species) * 0.1

        # Thresholds for spiking
        self.spike_thresholds = np.ones(num_species) * 0.5

        # Time constants (rise/fall times)
        self.tau_rise = np.ones(num_species) * 0.5
        self.tau_fall = np.ones(num_species) * 1.0

        # Track spike history
        self.spike_history = {i: [] for i in range(num_species)}
        self.concentration_history = {i: [] for i in range(num_species)}

    def add_reaction(self, source: int, target: int, rate: float, reaction_type: str = 'interaction'):
        """
        Add a reaction to the network

        source -> target with given rate
        """
        self.reactions.append({
            'source': source,
            'target': target,
            'rate': rate,
            'type': reaction_type
        })

    def remove_reaction(self, index: int):
        """Remove a reaction by index"""
        if 0 <= index < len(self.reactions):
            self.reactions.pop(index)

    def get_adjacency_matrix(self) -> np.ndarray:
        """Get adjacency matrix representation"""
        adj = np.zeros((self.num_species, self.num_species))
        for rxn in self.reactions:
            adj[rxn['source'], rxn['target']] = rxn['rate']
        return adj

    def set_decay_rates(self, rates: np.ndarray):
        """Set decay rates for all species"""
        self.decay_rates = np.asarray(rates)

    def set_spike_thresholds(self, thresholds: np.ndarray):
        """Set spike thresholds"""
        self.spike_thresholds = np.asarray(thresholds)

    def _dynamics(self, state: np.ndarray, t: float) -> np.ndarray:
        """
        Compute derivatives of concentrations

        d[M_i]/dt = sum(production) - decay*[M_i]
        """
        dstate = np.zeros(self.num_species)

        # Decay term
        dstate -= self.decay_rates * state

        # Reaction terms
        for rxn in self.reactions:
            source = rxn['source']
            target = rxn['target']
            rate = rxn['rate']

            # Bimolecular (A + B -> C) or unimolecular (A -> B)
            if state[source] > 0:
                production = rate * state[source]
                dstate[target] += production

        return dstate

    def simulate(self, t_span: np.ndarray, y0: Optional[np.ndarray] = None,
                 input_signal: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate the reaction network

        Args:
            t_span: Time points for simulation
            y0: Initial conditions (concentrations)
            input_signal: Dict with 'species', 'times', 'values' for external inputs

        Returns:
            (time_array, concentrations_array)
        """
        if y0 is None:
            y0 = np.random.uniform(0.01, 0.3, self.num_species)

        # Modified dynamics with input signal
        def dynamics_with_input(state, t):
            dstate = self._dynamics(state, t)

            if input_signal:
                for sp_idx, times, values in zip(
                    input_signal.get('species', []),
                    input_signal.get('times', []),
                    input_signal.get('values', [])
                ):
                    # Find which value applies at time t
                    idx = np.searchsorted(times, t)
                    if idx < len(values):
                        dstate[sp_idx] += values[idx]

            return dstate

        # Solve ODE
        solution = odeint(dynamics_with_input, y0, t_span)

        # Record history
        for i in range(self.num_species):
            self.concentration_history[i] = solution[:, i]

        return t_span, solution

    def detect_spikes(self, concentrations: np.ndarray, threshold_multiplier: float = 1.0) -> Dict[int, List[float]]:
        """
        Detect spikes in concentration time series

        Args:
            concentrations: Concentration array from simulation
            threshold_multiplier: Multiply thresholds by this factor

        Returns:
            Dict mapping species index to list of spike times
        """
        spikes = {i: [] for i in range(self.num_species)}

        for i in range(self.num_species):
            conc = concentrations[:, i]
            threshold = self.spike_thresholds[i] * threshold_multiplier

            # Find peaks above threshold
            for t in range(1, len(conc) - 1):
                if (conc[t] > threshold and
                    conc[t] > conc[t-1] and
                    conc[t] > conc[t+1]):
                    spikes[i].append(t)

        return spikes

    def compute_learning_matrix(self, spikes: Dict[int, List[int]],
                               tau_stdp: float = 2.0) -> np.ndarray:
        """
        Compute STDP-like learning matrix

        If M_i spikes before M_j: strengthen i->j connection
        """
        learning_matrix = np.zeros((self.num_species, self.num_species))

        for i in range(self.num_species):
            for j in range(self.num_species):
                if i == j:
                    continue

                if i in spikes and j in spikes:
                    for spike_i in spikes[i]:
                        for spike_j in spikes[j]:
                            dt = spike_j - spike_i
                            if 0 < dt < tau_stdp:  # Post-synaptic after pre-synaptic
                                # LTP (Long-term potentiation)
                                learning_matrix[i, j] += np.exp(-dt / tau_stdp)
                            elif -tau_stdp < dt < 0:  # Pre-synaptic before post
                                # LTD (Long-term depression) - weaker
                                learning_matrix[i, j] -= 0.5 * np.exp(dt / tau_stdp)

        return learning_matrix

    def apply_learning(self, learning_matrix: np.ndarray, learning_rate: float = 0.01):
        """Apply STDP learning to reaction rates"""
        for rxn in self.reactions:
            src = rxn['source']
            tgt = rxn['target']
            delta = learning_matrix[src, tgt]
            rxn['rate'] += learning_rate * delta
            rxn['rate'] = np.clip(rxn['rate'], 0.01, 2.0)

    def get_stats(self) -> Dict:
        """Get network statistics"""
        adj = self.get_adjacency_matrix()
        return {
            'num_species': self.num_species,
            'num_reactions': len(self.reactions),
            'density': np.sum(adj > 0) / (self.num_species ** 2),
            'mean_rate': np.mean([r['rate'] for r in self.reactions]) if self.reactions else 0,
            'max_rate': np.max([r['rate'] for r in self.reactions]) if self.reactions else 0,
        }


class SNNDynamics:
    """Spiking Neural Network dynamics for chemical comparison"""

    def __init__(self, num_neurons: int, tau_m: float = 10.0):
        """
        LIF neuron model parameters

        Args:
            num_neurons: Number of neurons
            tau_m: Membrane time constant (ms)
        """
        self.num_neurons = num_neurons
        self.tau_m = tau_m
        self.V_rest = -70  # Rest potential
        self.V_thresh = -50  # Spike threshold
        self.V_reset = -70  # Reset after spike
        self.tau_ref = 2  # Refractory period

    def lif_neuron(self, V: np.ndarray, I: np.ndarray, t: float) -> np.ndarray:
        """LIF neuron dynamics"""
        dV = (self.V_rest - V + I) / self.tau_m
        spikes = V > self.V_thresh
        V[spikes] = self.V_reset
        return dV


if __name__ == "__main__":
    # Test the physics engine
    net = ReactionNetwork(num_species=4)
    net.add_reaction(0, 1, 0.3)
    net.add_reaction(1, 2, 0.2)
    net.add_reaction(2, 0, 0.15)
    net.add_reaction(2, 3, 0.25)

    # Simulate
    t = np.linspace(0, 50, 1000)
    t_sim, conc = net.simulate(t)

    print("Network stats:")
    print(net.get_stats())
    print(f"Final concentrations: {conc[-1]}")
