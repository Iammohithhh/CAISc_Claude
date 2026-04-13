# ChemNeuron 🧠⚗️

**Neural Networks Through Chemistry** — A groundbreaking AI system that discovers molecular reaction networks capable of learning, inspired by spiking neurons and STDP.

> Intersection of: RL × Deep Learning (SNNs, GNNs) × Neuroscience × Chemistry × Physics × Future

---

## Vision

What if we could design **chemical reaction networks that learn like brains**?

ChemNeuron uses:
- **Spiking Neural Networks (SNNs)** as the template for what computation should look like
- **Graph Neural Networks (GNNs)** to represent molecular structures and interactions
- **Reinforcement Learning (RL)** to discover networks with learning capacity
- **STDP (Spike-Timing-Dependent Plasticity)** to enable Hebbian learning at the molecular level
- **Physics-based modeling** of reaction kinetics and dynamics
- **Chemistry** to ensure molecular viability

---

## Project Structure

```
nexus/
├── backend/
│   ├── physics_engine.py      # ODE solver for reaction networks
│   ├── visualizer.py          # Beautiful interactive visualizations
│   ├── snn_engine.py          # Spiking neural network dynamics (Phase 2)
│   ├── gnn_encoder.py         # Graph neural network for molecules (Phase 3)
│   └── rl_agent.py            # RL discoverer (Phase 4+)
├── phase1_demo.py             # Phase 1 demonstration
├── phase2_snn_integration.py  # Phase 2 demonstration
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

---

## Phases

### **Phase 1: Reaction Network Generator + Physics Simulator** ✓
- Generate reaction networks (directed graphs of chemical reactions)
- Simulate dynamics with ODE solver
- Detect spikes in concentration time series
- Compute STDP-like learning matrices
- Beautiful visualizations (network graphs, dynamics, heatmaps)
- Interactive Plotly dashboards

**Key Innovation**: Treat chemical concentrations like neural spikes; apply STDP to strengthen "synaptic" connections between molecules.

### **Phase 2: Spiking Neural Network Integration** (Next)
- Implement LIF (Leaky Integrate-and-Fire) neurons
- Model neuron dynamics using Brian2
- Compare SNN spike patterns with chemical network behavior
- Learn which chemical networks behave like neural networks

### **Phase 3: Graph Neural Networks for Molecules**
- Represent molecules as graphs (atoms = nodes, bonds = edges)
- Use PyTorch Geometric for message passing
- Encode molecular structures
- Predict molecular properties from graphs

### **Phase 4: Advanced RL Agent**
- Multi-objective reward function (learning, robustness, novelty, complexity)
- Actor-Critic architecture
- Curriculum learning (start simple, increase difficulty)
- Discover high-value reaction networks

### **Phase 5: Full Pipeline**
- End-to-end discovery system
- Train on thousands of network candidates
- Analyze discovered networks for interpretability
- Generate research reports on novel findings

---

## Installation

```bash
# Clone and setup
git clone https://github.com/iammohithhh/caisc_claude.git
cd nexus

# Install dependencies
pip install -r requirements.txt
```

**Note**: All packages are free and open-source. No API calls required.

---

## Quick Start

### Phase 1: Run Demo

```bash
python phase1_demo.py
```

This generates:
- `phase1_simple_network.png` - Analysis of a 3-species oscillatory network
- `phase1_complex_network.png` - Analysis of a 5-species network with feedback
- `phase1_dashboard.html` - Interactive Plotly dashboard (open in browser)

### Expected Output

The demo creates two reaction networks:

**Simple Network (3 species)**:
```
M0 → M1 → M2 → M0 (oscillatory loop)
```

**Complex Network (5 species)**:
```
M0 → M1 → M2 → M3 → M4
  ↓    ↓        ↓
  └────┘────────┴─ (feedback loops)
```

Both networks are simulated with:
- Concentration dynamics (similar to membrane potential)
- Spike detection (peaks above threshold)
- STDP learning (modify reaction rates based on spike timing)
- Phase space analysis

---

## Key Insights

### 1. Chemical Networks Can "Spike"
When concentrations exceed thresholds, we treat them as spikes—exactly like neurons!

### 2. Molecules Can "Learn"
Using STDP principles:
- If molecule A's spike precedes B's spike: strengthen A→B (LTP)
- If A spikes after B: weaken A→B (LTD)
- This adjusts reaction rate constants

### 3. Networks Show Emergent Dynamics
Simple networks develop:
- Oscillations
- Chaotic behavior
- Pattern formation
- Self-organization

### 4. Novel Discovery Potential
By searching through network space with RL, we can discover:
- Networks with unexpected computational abilities
- Minimal systems with maximal complexity
- Bio-inspired information processing systems

---

## Technical Details

### Physics Engine (`physics_engine.py`)

**Reaction Dynamics** (ODE-based):
```python
d[M_i]/dt = sum(production_terms) - decay_rate * [M_i]
```

**Spike Detection**:
- Find local maxima in concentration time series
- Mark as spike if above threshold

**STDP Learning**:
```python
# If pre-synaptic (source) fires before post-synaptic (target):
Learning_strength += exp(-dt / tau_stdp)  # LTP (strengthen)

# If pre-synaptic fires after post-synaptic:
Learning_strength -= 0.5 * exp(dt / tau_stdp)  # LTD (weaken)

# Update reaction rate:
k_new = k_old + learning_rate * Learning_strength
```

### Visualizations

- **Network Graph**: Directed graph with node sizes = degree, colors = function
- **Dynamics Plot**: Concentration vs time for all species
- **Spike Raster**: Stars marking detected spikes on concentration traces
- **STDP Matrix**: Heatmap of learned connections (Blue=LTP, Red=LTD)
- **Phase Space**: 2D projection showing attractor structure
- **Interactive Dashboard**: Plotly multi-panel view with hover info

---

## Why This Is Novel

1. **Never Done Before**: No one has systematically searched for learning-capable chemical systems using RL + SNNs
2. **Fundamental Question**: Can chemistry implement computation in ways neurons do?
3. **Practical Impact**: Could revolutionize:
   - Bio-computing (programmable cells)
   - Drug design (learning molecular systems)
   - Materials science (adaptive materials)
   - Information storage (molecular memory)
4. **Bridging Disciplines**: Uniquely combines neuroscience, chemistry, physics, and AI

---

## Dependencies

- `torch` — Deep learning framework
- `torch_geometric` — Graph neural networks
- `brian2` — Spiking neural network simulator
- `scipy` — ODE solver, signal processing
- `numpy` — Numerical computation
- `matplotlib` — Static visualizations
- `plotly` — Interactive dashboards
- `networkx` — Graph algorithms
- `stable-baselines3` — RL training

---

## Future Work

- Phase 2: Full SNN integration
- Phase 3: Molecular GNN encoder
- Phase 4: Advanced multi-objective RL
- Phase 5: Scalable training pipeline
- Real Chemistry Validation: Compare predicted reactions with wet-lab data
- Quantum Extensions: Include quantum effects in molecular dynamics
- Publication: Scientific paper on discovered network properties

---

## Author

Built with ❤️ for deep tech AI × neuroscience × chemistry

---

## License

MIT License — Free to use, modify, and share.

---

**Status**: Phase 1 Complete ✓ | Phase 2 In Progress...
