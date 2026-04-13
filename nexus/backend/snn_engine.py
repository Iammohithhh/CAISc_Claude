"""
ChemNeuron Phase 2: Spiking Neural Network Engine
Implements LIF (Leaky Integrate-and-Fire) neurons and STDP learning
Uses Brian2 for efficient neuromorphic simulation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class LIFNeuron:
    """
    Leaky Integrate-and-Fire (LIF) Neuron Model

    Membrane potential dynamics:
    dV/dt = (-V + R*I) / tau_m
    """

    def __init__(self, tau_m: float = 20.0, V_rest: float = -70.0,
                 V_thresh: float = -55.0, V_reset: float = -70.0,
                 tau_ref: float = 2.0):
        """
        Initialize LIF neuron parameters

        Args:
            tau_m: Membrane time constant (ms)
            V_rest: Resting potential (mV)
            V_thresh: Spike threshold (mV)
            V_reset: Reset potential after spike (mV)
            tau_ref: Refractory period (ms)
        """
        self.tau_m = tau_m
        self.V_rest = V_rest
        self.V_thresh = V_thresh
        self.V_reset = V_reset
        self.tau_ref = tau_ref
        self.R = 1.0  # Resistance (arbitrary units)

        # State variables
        self.V = V_rest
        self.t_last_spike = -np.inf
        self.spike_times = []

    def step(self, I_in: float, dt: float) -> bool:
        """
        Simulate one timestep of the LIF neuron

        Args:
            I_in: Input current
            dt: Timestep (ms)

        Returns:
            bool: Whether neuron spiked this timestep
        """
        # Check refractory period
        if (self.t_last_spike + self.tau_ref) > 0:
            return False

        # Leaky integration
        dV = (-self.V + self.R * I_in) / self.tau_m
        self.V += dV * dt

        # Spiking
        if self.V >= self.V_thresh:
            self.spike_times.append(len(self.spike_times))
            self.V = self.V_reset
            self.t_last_spike = 0  # Start refractory period
            return True

        # Update refractory timer
        if self.t_last_spike > -np.inf:
            self.t_last_spike -= dt

        return False


class SNNNetwork:
    """Spiking Neural Network with STDP learning"""

    def __init__(self, num_neurons: int, tau_m: float = 20.0):
        """
        Initialize SNN

        Args:
            num_neurons: Number of neurons
            tau_m: Membrane time constant
        """
        self.num_neurons = num_neurons
        self.tau_m = tau_m
        self.dt = 0.1  # Timestep (ms)

        # Neurons
        self.neurons = [LIFNeuron(tau_m=tau_m) for _ in range(num_neurons)]

        # Synaptic weights (excitatory/inhibitory)
        self.W = np.random.uniform(-0.5, 0.5, (num_neurons, num_neurons))
        np.fill_diagonal(self.W, 0)  # No self-connections

        # Synaptic eligibility traces for STDP
        self.eligibility = np.zeros((num_neurons, num_neurons))

        # STDP parameters
        self.A_plus = 0.01   # LTP amplitude
        self.A_minus = 0.01  # LTD amplitude
        self.tau_stdp = 20.0  # STDP time constant (ms)

        # History tracking
        self.V_history = {i: [] for i in range(num_neurons)}
        self.spike_history = {i: [] for i in range(num_neurons)}
        self.W_history = []

    def set_weights(self, W: np.ndarray):
        """Set synaptic weight matrix"""
        self.W = W.copy()
        np.fill_diagonal(self.W, 0)

    def get_weights(self) -> np.ndarray:
        """Get current weight matrix"""
        return self.W.copy()

    def simulate_step(self, I_in: np.ndarray, t_step: int) -> np.ndarray:
        """
        Simulate one timestep of the SNN

        Args:
            I_in: Input currents for each neuron
            t_step: Current timestep number

        Returns:
            spikes: Boolean array indicating which neurons spiked
        """
        spikes = np.zeros(self.num_neurons, dtype=bool)

        # Compute recurrent input
        I_recurrent = np.zeros(self.num_neurons)
        for i in range(self.num_neurons):
            for j in range(self.num_neurons):
                if i != j and hasattr(self.neurons[j], '_last_spike_time'):
                    # Exponential decay of input from last spike
                    time_since_spike = t_step - self.neurons[j]._last_spike_time
                    if 0 <= time_since_spike < 50:  # Recent spike
                        I_recurrent[i] += self.W[j, i] * np.exp(-time_since_spike / 10.0)

        # Update each neuron
        for i in range(self.num_neurons):
            I_total = I_in[i] + I_recurrent[i] if isinstance(I_in, np.ndarray) else I_in
            spiked = self.neurons[i].step(I_total, self.dt)

            if spiked:
                spikes[i] = True
                self.neurons[i]._last_spike_time = t_step
                self.spike_history[i].append(t_step)

            # Record voltage
            self.V_history[i].append(self.neurons[i].V)

        # Update eligibility traces and apply STDP
        self._update_stdp(spikes)

        return spikes

    def _update_stdp(self, spikes: np.ndarray):
        """
        Update eligibility traces and apply STDP

        Pre-post (spike order) dependent plasticity
        """
        # Decay eligibility
        self.eligibility *= np.exp(-self.dt / self.tau_stdp)

        # Mark pre-synaptic activity
        for i in range(self.num_neurons):
            for j in range(self.num_neurons):
                if i != j:
                    if spikes[i]:  # Pre-synaptic spike
                        self.eligibility[i, j] += self.A_plus

        # Apply post-synaptic weight changes
        for j in range(self.num_neurons):
            if spikes[j]:  # Post-synaptic spike
                for i in range(self.num_neurons):
                    if i != j:
                        # LTP: if pre-synaptic activity preceded post-synaptic spike
                        self.W[i, j] += self.eligibility[i, j]
                        # LTD: reduce other weights
                        self.W[i, j] -= self.A_minus * np.sum(self.eligibility[:, j])

        # Clip weights
        self.W = np.clip(self.W, -1.0, 1.0)

    def simulate(self, duration: float, I_in_sequence: Optional[np.ndarray] = None
                ) -> Tuple[np.ndarray, Dict]:
        """
        Simulate the SNN for a given duration

        Args:
            duration: Simulation duration (ms)
            I_in_sequence: Input current sequence (num_steps x num_neurons)

        Returns:
            (time_array, results_dict)
        """
        num_steps = int(duration / self.dt)

        # Default input: Poisson-like random input
        if I_in_sequence is None:
            I_in_sequence = np.random.poisson(0.1, (num_steps, self.num_neurons)).astype(float)

        # Run simulation
        spike_raster = np.zeros((num_steps, self.num_neurons), dtype=bool)

        for t in range(num_steps):
            I_in = I_in_sequence[t] if t < len(I_in_sequence) else np.zeros(self.num_neurons)
            spikes = self.simulate_step(I_in, t)
            spike_raster[t, :] = spikes

        # Record weight evolution
        self.W_history.append(self.W.copy())

        return np.arange(num_steps) * self.dt, {
            'spike_raster': spike_raster,
            'V_history': self.V_history,
            'spike_times': self.spike_history,
            'weights': self.W.copy(),
            'dt': self.dt
        }

    def get_spike_counts(self) -> np.ndarray:
        """Get total spike count for each neuron"""
        return np.array([len(self.spike_history[i]) for i in range(self.num_neurons)])

    def get_firing_rates(self, duration: float) -> np.ndarray:
        """Get average firing rate (Hz) for each neuron"""
        spike_counts = self.get_spike_counts()
        return spike_counts / (duration / 1000.0)  # Convert to Hz

    def compute_correlation(self) -> np.ndarray:
        """
        Compute spike-time correlation matrix

        Measures how similar spike patterns are across neurons
        """
        correlation = np.zeros((self.num_neurons, self.num_neurons))

        for i in range(self.num_neurons):
            for j in range(self.num_neurons):
                if i == j:
                    correlation[i, j] = 1.0
                else:
                    spikes_i = set(self.spike_history[i])
                    spikes_j = set(self.spike_history[j])

                    if len(spikes_i) == 0 or len(spikes_j) == 0:
                        correlation[i, j] = 0.0
                    else:
                        overlap = len(spikes_i & spikes_j)
                        union = len(spikes_i | spikes_j)
                        correlation[i, j] = overlap / union if union > 0 else 0.0

        return correlation

    def reset(self):
        """Reset network state"""
        self.neurons = [LIFNeuron(tau_m=self.tau_m) for _ in range(self.num_neurons)]
        self.V_history = {i: [] for i in range(self.num_neurons)}
        self.spike_history = {i: [] for i in range(self.num_neurons)}
        self.eligibility = np.zeros((self.num_neurons, self.num_neurons))


class ChemicalToSNNComparator:
    """Compare chemical network dynamics with SNN dynamics"""

    @staticmethod
    def align_timescales(chem_conc: np.ndarray, snn_spikes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align chemical concentration with SNN spike rates

        Args:
            chem_conc: Chemical concentration array (T x N)
            snn_spikes: SNN spike raster (T x N)

        Returns:
            (normalized_chem, smoothed_spikes)
        """
        # Normalize chemical concentrations
        chem_norm = chem_conc / (np.max(chem_conc, axis=0) + 1e-6)

        # Smooth spikes with exponential filter
        window = 10
        smoothed = np.zeros_like(snn_spikes, dtype=float)
        for i in range(snn_spikes.shape[1]):
            for t in range(snn_spikes.shape[0]):
                for dt in range(-window, window + 1):
                    if 0 <= t + dt < snn_spikes.shape[0]:
                        smoothed[t, i] += snn_spikes[t + dt, i] * np.exp(-abs(dt) / 5.0)

        return chem_norm, smoothed / (2 * window + 1)

    @staticmethod
    def compute_similarity(chem_conc: np.ndarray, snn_spikes: np.ndarray) -> float:
        """
        Compute how similar chemical and SNN dynamics are

        Returns:
            similarity_score in [0, 1]
        """
        chem_norm, snn_smooth = ChemicalToSNNComparator.align_timescales(chem_conc, snn_spikes)

        # Correlation-based similarity
        similarities = []
        for i in range(min(chem_norm.shape[1], snn_smooth.shape[1])):
            corr = np.corrcoef(chem_norm[:, i], snn_smooth[:, i])[0, 1]
            if not np.isnan(corr):
                similarities.append((corr + 1) / 2)  # Normalize to [0, 1]

        return np.mean(similarities) if similarities else 0.0

    @staticmethod
    def extract_features(chem_conc: np.ndarray, snn_spikes: np.ndarray) -> Dict:
        """
        Extract comparable features from chemical and SNN networks

        Args:
            chem_conc: Chemical concentration array
            snn_spikes: SNN spike raster

        Returns:
            Dictionary of features
        """
        return {
            'chem_mean': np.mean(chem_conc, axis=0),
            'chem_std': np.std(chem_conc, axis=0),
            'chem_max': np.max(chem_conc, axis=0),
            'snn_firing_rate': np.sum(snn_spikes, axis=0) / snn_spikes.shape[0],
            'snn_spike_count': np.sum(snn_spikes, axis=0),
            'similarity': ChemicalToSNNComparator.compute_similarity(chem_conc, snn_spikes)
        }


if __name__ == "__main__":
    # Test SNN
    snn = SNNNetwork(num_neurons=4)

    # Create simple input pattern
    duration = 500  # ms
    num_steps = int(duration / snn.dt)
    I_in = np.zeros((num_steps, 4))
    I_in[100:200, 0] = 0.5  # Input to neuron 0
    I_in[300:400, 1] = 0.5  # Input to neuron 1

    # Simulate
    t, results = snn.simulate(duration, I_in)

    print("SNN Simulation Results:")
    print(f"Firing rates: {snn.get_firing_rates(duration)}")
    print(f"Spike counts: {snn.get_spike_counts()}")
    print(f"Final weights:\n{snn.W}")
