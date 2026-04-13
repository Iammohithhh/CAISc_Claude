"""
Beautiful visualizations for ChemNeuron
Interactive plots with Plotly + Matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional
from physics_engine import ReactionNetwork


class ChemNeuronVisualizer:
    """Beautiful visualizations for reaction networks"""

    def __init__(self, figsize=(14, 10), style='dark'):
        self.figsize = figsize
        self.style = style
        self.setup_style()

    def setup_style(self):
        """Setup dark theme"""
        if self.style == 'dark':
            plt.style.use('dark_background')
            self.bg_color = '#0a0e27'
            self.primary_color = '#00d9ff'
            self.secondary_color = '#ff006e'
            self.tertiary_color = '#ffbe0b'
        else:
            plt.style.use('default')
            self.bg_color = 'white'
            self.primary_color = '#0066cc'
            self.secondary_color = '#ff0033'
            self.tertiary_color = '#ffaa00'

    def plot_network_graph(self, network: ReactionNetwork, ax=None) -> np.ndarray:
        """
        Plot reaction network as directed graph

        Beautiful network visualization with node sizes based on importance
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)

        # Build networkx graph
        G = nx.DiGraph()
        G.add_nodes_from(range(network.num_species))

        for rxn in network.reactions:
            G.add_edge(rxn['source'], rxn['target'], weight=rxn['rate'])

        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

        # Node sizes based on degree
        node_sizes = [300 + 100 * G.degree(node) for node in G.nodes()]

        # Draw edges with arrow
        nx.draw_networkx_edges(
            G, pos,
            ax=ax,
            edge_color=self.primary_color,
            width=2.5,
            alpha=0.8,
            arrows=True,
            arrowsize=20,
            connectionstyle='arc3,rad=0.1',
            arrowstyle='-|>'
        )

        # Draw nodes
        node_colors = [self.secondary_color if G.in_degree(node) > 0 else self.tertiary_color
                      for node in G.nodes()]
        nx.draw_networkx_nodes(
            G, pos,
            ax=ax,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.9,
            edgecolors=self.primary_color,
            linewidths=2
        )

        # Labels
        nx.draw_networkx_labels(
            G, pos,
            ax=ax,
            labels={i: network.species_names[i] for i in range(network.num_species)},
            font_size=11,
            font_weight='bold',
            font_color='white'
        )

        ax.set_title('Reaction Network Topology\n(Nodes = Species, Edges = Reactions)',
                    fontsize=14, fontweight='bold', pad=20)
        ax.axis('off')

        return ax

    def plot_dynamics(self, t: np.ndarray, concentrations: np.ndarray,
                     network: ReactionNetwork, ax=None):
        """Plot concentration dynamics over time"""
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)

        colors = plt.cm.rainbow(np.linspace(0, 1, network.num_species))

        for i in range(network.num_species):
            ax.plot(t, concentrations[:, i],
                   label=network.species_names[i],
                   linewidth=2.5,
                   color=colors[i],
                   alpha=0.85)

        ax.set_xlabel('Time', fontsize=12, fontweight='bold')
        ax.set_ylabel('Concentration', fontsize=12, fontweight='bold')
        ax.set_title('Chemical Species Dynamics\n(Concentration vs Time)',
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3)

        return ax

    def plot_spikes(self, t: np.ndarray, concentrations: np.ndarray,
                   spikes: Dict[int, List], network: ReactionNetwork, ax=None):
        """Plot spike raster diagram"""
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)

        colors = plt.cm.rainbow(np.linspace(0, 1, network.num_species))

        # Plot concentrations as background
        for i in range(network.num_species):
            ax.plot(t, concentrations[:, i] + i * 1.5,
                   alpha=0.3, linewidth=1.5, color=colors[i])

        # Plot spikes
        for species_idx in range(network.num_species):
            spike_times = spikes[species_idx]
            spike_y = [concentrations[st, species_idx] + species_idx * 1.5 for st in spike_times]
            ax.scatter(spike_times, spike_y,
                      s=150, marker='*', color=colors[species_idx],
                      edgecolors='yellow', linewidths=1.5, zorder=5,
                      label=f'{network.species_names[species_idx]} spikes')

        ax.set_xlabel('Time', fontsize=12, fontweight='bold')
        ax.set_ylabel('Species', fontsize=12, fontweight='bold')
        ax.set_title('Spike Raster\n(Stars = Detected Spikes)',
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.2)

        return ax

    def plot_learning_matrix(self, learning_matrix: np.ndarray,
                            network: ReactionNetwork, ax=None):
        """Plot STDP learning matrix as heatmap"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(learning_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)

        ax.set_xticks(range(network.num_species))
        ax.set_yticks(range(network.num_species))
        ax.set_xticklabels(network.species_names)
        ax.set_yticklabels(network.species_names)
        ax.set_xlabel('Post-synaptic (Target)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Pre-synaptic (Source)', fontsize=11, fontweight='bold')
        ax.set_title('STDP Learning Matrix\n(Red=LTD, Blue=LTP)',
                    fontsize=13, fontweight='bold', pad=15)

        plt.colorbar(im, ax=ax, label='Learning Strength')
        ax.grid(True, alpha=0.3, color='white', linewidth=0.5)

        return ax

    def plot_phase_space(self, concentrations: np.ndarray,
                        species_x: int, species_y: int,
                        network: ReactionNetwork, ax=None):
        """Plot phase space (2D projection)"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))

        # Color by time
        colors = plt.cm.plasma(np.linspace(0, 1, len(concentrations)))

        scatter = ax.scatter(concentrations[:, species_x],
                            concentrations[:, species_y],
                            c=range(len(concentrations)),
                            cmap='plasma',
                            s=30, alpha=0.8, edgecolors='white', linewidth=0.5)

        ax.set_xlabel(f'{network.species_names[species_x]} Concentration',
                     fontsize=11, fontweight='bold')
        ax.set_ylabel(f'{network.species_names[species_y]} Concentration',
                     fontsize=11, fontweight='bold')
        ax.set_title(f'Phase Space: {network.species_names[species_x]} vs {network.species_names[species_y]}\n(Color = Time Progress)',
                    fontsize=12, fontweight='bold', pad=15)

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Time Step', fontsize=10)
        ax.grid(True, alpha=0.3)

        return ax

    def create_interactive_dashboard(self, t: np.ndarray, concentrations: np.ndarray,
                                    spikes: Dict[int, List],
                                    network: ReactionNetwork,
                                    save_path: Optional[str] = None) -> go.Figure:
        """
        Create interactive Plotly dashboard

        Multiple subplots: dynamics, spikes, phase space
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Concentration Dynamics',
                'Phase Space (M0 vs M1)',
                'Species Activity Heatmap',
                'Network Info'
            ),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                   [{'type': 'heatmap'}, {'type': 'bar'}]]
        )

        colors = px.colors.qualitative.Plotly[:network.num_species]

        # Plot 1: Dynamics
        for i in range(network.num_species):
            fig.add_trace(
                go.Scatter(x=t, y=concentrations[:, i],
                          mode='lines',
                          name=network.species_names[i],
                          line=dict(width=2.5, color=colors[i % len(colors)]),
                          opacity=0.8),
                row=1, col=1
            )

        # Plot 2: Phase space
        fig.add_trace(
            go.Scatter(x=concentrations[:, 0], y=concentrations[:, 1],
                      mode='markers',
                      name='Phase Space',
                      marker=dict(
                          size=5,
                          color=range(len(concentrations)),
                          colorscale='Viridis',
                          showscale=True,
                          colorbar=dict(x=0.46, len=0.4)
                      )),
            row=1, col=2
        )

        # Plot 3: Heatmap of concentrations
        heatmap_data = concentrations[::5, :].T  # Downsample for visibility
        fig.add_trace(
            go.Heatmap(z=heatmap_data,
                      colorscale='Viridis',
                      name='Concentrations'),
            row=2, col=1
        )

        # Plot 4: Network stats bar
        stats = network.get_stats()
        fig.add_trace(
            go.Bar(x=['Reactions', 'Density', 'Mean Rate'],
                  y=[stats['num_reactions'], stats['density'] * 10, stats['mean_rate']],
                  marker_color=[colors[i % len(colors)] for i in range(3)],
                  showlegend=False),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(
            title_text="<b>ChemNeuron Interactive Dashboard</b>",
            height=900,
            showlegend=True,
            hovermode='closest',
            template='plotly_dark',
            font=dict(size=11)
        )

        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_yaxes(title_text="Concentration", row=1, col=1)

        fig.update_xaxes(title_text=f"{network.species_names[0]}", row=1, col=2)
        fig.update_yaxes(title_text=f"{network.species_names[1]}", row=1, col=2)

        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_xaxes(title_text="Metric", row=2, col=2)

        if save_path:
            fig.write_html(save_path)
            print(f"Dashboard saved to {save_path}")

        return fig


if __name__ == "__main__":
    # Test visualizer
    from physics_engine import ReactionNetwork

    net = ReactionNetwork(num_species=4, species_names=['M0', 'M1', 'M2', 'M3'])
    net.add_reaction(0, 1, 0.3)
    net.add_reaction(1, 2, 0.2)
    net.add_reaction(2, 0, 0.15)
    net.add_reaction(2, 3, 0.25)

    t = np.linspace(0, 50, 1000)
    t_sim, conc = net.simulate(t)
    spikes = net.detect_spikes(conc)

    visualizer = ChemNeuronVisualizer(style='dark')

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor('#0a0e27')

    ax1 = plt.subplot(2, 3, 1)
    visualizer.plot_network_graph(net, ax1)

    ax2 = plt.subplot(2, 3, 2)
    visualizer.plot_dynamics(t_sim, conc, net, ax2)

    ax3 = plt.subplot(2, 3, 3)
    visualizer.plot_spikes(t_sim, conc, spikes, net, ax3)

    learning_matrix = net.compute_learning_matrix(spikes)
    ax4 = plt.subplot(2, 3, 4)
    visualizer.plot_learning_matrix(learning_matrix, net, ax4)

    ax5 = plt.subplot(2, 3, 5)
    visualizer.plot_phase_space(conc, 0, 1, net, ax5)

    plt.tight_layout()
    plt.savefig('chemneuron_overview.png', dpi=150, facecolor='#0a0e27')
    print("Saved visualization to chemneuron_overview.png")

    # Interactive dashboard
    visualizer.create_interactive_dashboard(t_sim, conc, spikes, net,
                                           save_path='dashboard.html')
