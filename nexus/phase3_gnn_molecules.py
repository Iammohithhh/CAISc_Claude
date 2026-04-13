"""
ChemNeuron Phase 3: Graph Neural Networks for Molecules
Learn molecular representations and predict chemical properties
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
import os
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from gnn_encoder import (
    MoleculeGraph, GNNEncoder, PropertyPredictor,
    MolecularGNNPipeline, ReactionNetworkToGraph
)
from physics_engine import ReactionNetwork
from visualizer import ChemNeuronVisualizer
import networkx as nx


class Phase3Runner:
    """Run Phase 3: GNN for molecular learning"""

    def __init__(self):
        self.visualizer = ChemNeuronVisualizer(style='dark')
        self.pipeline = MolecularGNNPipeline(device='cpu')

    def generate_molecular_dataset(self, num_molecules: int = 20) -> tuple:
        """
        Generate synthetic molecular dataset

        Returns:
            (graphs, properties_targets)
        """
        graphs = []
        properties_list = []

        for i in range(num_molecules):
            # Random molecule size
            num_atoms = np.random.randint(5, 15)
            edge_density = np.random.uniform(0.2, 0.6)

            # Create graph
            graph = MoleculeGraph.create_simple_graph(num_atoms, edge_density)
            graphs.append(graph)

            # Compute theoretical properties
            props = MoleculeGraph.compute_molecular_properties(num_atoms, edge_density)
            properties_list.append([
                props['molecular_weight'],
                props['LogP'],
                props['HBD'],
                props['HBA'],
                props['TPSA'],
                props['RotBonds']
            ])

        # Normalize targets
        properties_array = np.array(properties_list)
        properties_mean = properties_array.mean(axis=0)
        properties_std = properties_array.std(axis=0) + 1e-6
        properties_normalized = (properties_array - properties_mean) / properties_std

        targets = torch.tensor(properties_normalized, dtype=torch.float32)

        return graphs, targets, properties_mean, properties_std

    def train_gnn(self, graphs: List, targets: torch.Tensor, epochs: int = 10) -> List[float]:
        """Train GNN on molecular dataset"""
        losses = []

        for epoch in range(epochs):
            # Simple training (single batch)
            loss = self.pipeline.train_step(graphs, targets)
            losses.append(loss)

            if (epoch + 1) % 2 == 0:
                print(f"  Epoch {epoch + 1}/{epochs}: Loss = {loss:.6f}")

        return losses

    def visualize_molecular_structures(self, graphs: List, num_show: int = 6):
        """Visualize molecular graph structures"""
        fig = plt.figure(figsize=(18, 12))
        fig.patch.set_facecolor('#0a0e27')
        fig.suptitle('Phase 3: Molecular Graph Structures',
                    fontsize=18, fontweight='bold', y=0.98)

        for idx in range(min(num_show, len(graphs))):
            ax = plt.subplot(2, 3, idx + 1)

            graph = graphs[idx]

            # Convert to networkx
            G = nx.DiGraph()
            G.add_nodes_from(range(graph.num_nodes))

            for i in range(graph.edge_index.shape[1]):
                u = graph.edge_index[0, i].item()
                v = graph.edge_index[1, i].item()
                G.add_edge(u, v)

            # Layout
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

            # Draw
            nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#00d9ff',
                                 width=2, alpha=0.8, arrows=True, arrowsize=15)

            node_colors = plt.cm.viridis(np.linspace(0, 1, graph.num_nodes))
            nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                                 node_size=500, edgecolors='#00d9ff', linewidths=2)

            nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_weight='bold')

            ax.set_title(f'Molecule {idx + 1}\n({graph.num_nodes} atoms, {graph.edge_index.shape[1]} bonds)',
                        fontweight='bold', fontsize=11)
            ax.axis('off')

        plt.tight_layout()
        plt.savefig('phase3_molecular_structures.png', dpi=150, facecolor='#0a0e27',
                   bbox_inches='tight')
        print("  ✓ Saved: phase3_molecular_structures.png")

    def plot_training_curves(self, losses: List[float]):
        """Plot GNN training curves"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor('#0a0e27')
        fig.suptitle('Phase 3: GNN Training Dynamics', fontsize=16, fontweight='bold')

        # Loss curve
        ax = axes[0]
        ax.plot(losses, linewidth=2.5, color='#00d9ff', marker='o', markersize=6)
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('MSE Loss', fontweight='bold')
        ax.set_title('Training Loss', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        # Moving average
        ax = axes[1]
        window = 3
        moving_avg = np.convolve(losses, np.ones(window) / window, mode='valid')
        ax.plot(moving_avg, linewidth=2.5, color='#ff006e', marker='s', markersize=6)
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Moving Avg Loss', fontweight='bold')
        ax.set_title(f'Smoothed Loss (window={window})', fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('phase3_training_curves.png', dpi=150, facecolor='#0a0e27',
                   bbox_inches='tight')
        print("  ✓ Saved: phase3_training_curves.png")

    def analyze_embeddings(self, graphs: List, num_vis: int = 12):
        """Analyze learned embeddings with dimensionality reduction"""
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE

        print("\n  Computing embeddings...")
        embeddings = []
        for graph in graphs[:num_vis]:
            emb = self.pipeline.encode_molecule(graph).numpy()
            embeddings.append(emb[0])

        embeddings = np.array(embeddings)

        # PCA
        print("  Applying PCA...")
        pca = PCA(n_components=2)
        embeddings_pca = pca.fit_transform(embeddings)

        # t-SNE
        print("  Applying t-SNE...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=3)
        embeddings_tsne = tsne.fit_transform(embeddings)

        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.patch.set_facecolor('#0a0e27')
        fig.suptitle('Phase 3: Learned Molecular Embeddings', fontsize=16, fontweight='bold')

        # PCA
        ax = axes[0]
        scatter1 = ax.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1],
                             s=200, c=range(len(embeddings_pca)), cmap='viridis',
                             alpha=0.8, edgecolors='#00d9ff', linewidths=2)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontweight='bold')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontweight='bold')
        ax.set_title('PCA Projection', fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=ax, label='Molecule ID')

        # t-SNE
        ax = axes[1]
        scatter2 = ax.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1],
                             s=200, c=range(len(embeddings_tsne)), cmap='plasma',
                             alpha=0.8, edgecolors='#ff006e', linewidths=2)
        ax.set_xlabel('t-SNE 1', fontweight='bold')
        ax.set_ylabel('t-SNE 2', fontweight='bold')
        ax.set_title('t-SNE Projection', fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax, label='Molecule ID')

        plt.tight_layout()
        plt.savefig('phase3_embeddings.png', dpi=150, facecolor='#0a0e27', bbox_inches='tight')
        print("  ✓ Saved: phase3_embeddings.png")

        return embeddings_pca, embeddings_tsne

    def predict_and_compare(self, graphs: List, num_show: int = 6):
        """Predict properties and show comparisons"""
        print("\n  Predicting molecular properties...")

        fig = plt.figure(figsize=(16, 10))
        fig.patch.set_facecolor('#0a0e27')
        fig.suptitle('Phase 3: Property Predictions', fontsize=16, fontweight='bold')

        property_names = ['MW', 'LogP', 'HBD', 'HBA', 'TPSA', 'RotBonds']

        for idx in range(min(num_show, len(graphs))):
            ax = plt.subplot(2, 3, idx + 1)

            graph = graphs[idx]
            predicted = self.pipeline.predict_properties(graph)

            # Normalize for display
            pred_values = [
                predicted['molecular_weight'] / 100,
                (predicted['LogP'] + 5) / 10,
                predicted['HBD'],
                predicted['HBA'],
                predicted['TPSA'] / 100,
                predicted['RotBonds']
            ]

            x_pos = np.arange(len(property_names))
            bars = ax.bar(x_pos, pred_values, color='#00d9ff', alpha=0.8, edgecolor='#ff006e', linewidth=2)

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

            ax.set_ylabel('Normalized Value', fontweight='bold')
            ax.set_title(f'Molecule {idx + 1}', fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(property_names, rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, max(pred_values) * 1.15)

        plt.tight_layout()
        plt.savefig('phase3_predictions.png', dpi=150, facecolor='#0a0e27', bbox_inches='tight')
        print("  ✓ Saved: phase3_predictions.png")

    def test_reaction_network_encoding(self):
        """Show how reaction networks can be encoded as molecular graphs"""
        print("\n  Testing reaction network as graphs...")

        # Create a reaction network
        net = ReactionNetwork(num_species=4, species_names=['A', 'B', 'C', 'D'])
        net.add_reaction(0, 1, 0.4)
        net.add_reaction(1, 2, 0.3)
        net.add_reaction(2, 3, 0.2)
        net.add_reaction(3, 0, 0.15)

        # Convert to graph
        graph = ReactionNetworkToGraph.from_reaction_network(net)
        print(f"    Network graph: {graph.num_nodes} nodes, {graph.edge_index.shape[1]} edges")

        # Encode
        embedding = self.pipeline.encode_molecule(graph)
        print(f"    Embedding dimension: {embedding.shape}")
        print(f"    Embedding stats: mean={embedding.mean():.4f}, std={embedding.std():.4f}")

        return graph, embedding


def main():
    print("\n" + "▓"*70)
    print("▓" + " "*68 + "▓")
    print("▓  ChemNeuron Phase 3: Graph Neural Networks for Molecules".center(70) + "▓")
    print("▓" + " "*68 + "▓")
    print("▓"*70)

    runner = Phase3Runner()

    print("\n[1] Generating molecular dataset...")
    graphs, targets, mean, std = runner.generate_molecular_dataset(num_molecules=20)
    print(f"    ✓ Generated {len(graphs)} molecules")

    print("\n[2] Visualizing molecular structures...")
    runner.visualize_molecular_structures(graphs, num_show=6)

    print("\n[3] Training GNN on molecular properties...")
    losses = runner.train_gnn(graphs, targets, epochs=10)

    print("\n[4] Plotting training dynamics...")
    runner.plot_training_curves(losses)

    print("\n[5] Analyzing learned embeddings...")
    runner.analyze_embeddings(graphs, num_vis=12)

    print("\n[6] Making property predictions...")
    runner.predict_and_compare(graphs, num_show=6)

    print("\n[7] Testing reaction network encoding...")
    graph, embedding = runner.test_reaction_network_encoding()

    # Statistics
    print("\n" + "="*70)
    print("PHASE 3 RESULTS")
    print("="*70)
    print(f"Final Training Loss: {losses[-1]:.6f}")
    print(f"Initial Training Loss: {losses[0]:.6f}")
    print(f"Loss Reduction: {(1 - losses[-1]/losses[0])*100:.1f}%")
    print(f"\nEmbedding Dimension: {embedding.shape[1]}")
    print(f"Molecules Processed: {len(graphs)}")

    print("\n" + "="*70)
    print("KEY INNOVATIONS")
    print("="*70)
    print("✓ Molecular graphs encoded via Graph Neural Networks")
    print("✓ Learned distributed representations of molecules")
    print("✓ Property prediction from graph structure")
    print("✓ Reaction networks can be treated as molecular graphs")
    print("✓ Enables inverse design: search for molecules with desired properties")

    print("\n" + "="*70)
    print("PHASE 3 COMPLETE ✓")
    print("="*70)
    print("\nNext: Phase 4 - Advanced RL Agent for Network Discovery")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
