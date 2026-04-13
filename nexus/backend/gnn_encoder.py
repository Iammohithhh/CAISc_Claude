"""
ChemNeuron Phase 3: Graph Neural Network Encoder
Represents molecules as graphs and learns molecular embeddings
Uses PyTorch and PyTorch Geometric for efficient graph processing
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GraphConv, global_mean_pool, global_max_pool
from torch_geometric.utils import dense_to_sparse
from typing import Dict, List, Tuple, Optional, Callable
import warnings
warnings.filterwarnings('ignore')


class MoleculeGraph:
    """Convert molecule SMILES to graph representation"""

    # Atom types (simplified periodic table)
    ATOM_TYPES = {
        'C': 0, 'N': 1, 'O': 2, 'H': 3, 'S': 4, 'P': 5,
        'F': 6, 'Cl': 7, 'Br': 8, 'I': 9, 'B': 10, 'Si': 11
    }

    # Bond types
    BOND_TYPES = {
        'single': 0,
        'double': 1,
        'triple': 2,
        'aromatic': 3
    }

    @staticmethod
    def create_simple_graph(num_atoms: int = 5, edge_density: float = 0.3) -> Data:
        """
        Create a random molecular-like graph

        Args:
            num_atoms: Number of atoms (nodes)
            edge_density: Probability of edge between atoms

        Returns:
            PyTorch Geometric Data object
        """
        # Node features: atom type (one-hot)
        num_atom_types = len(MoleculeGraph.ATOM_TYPES)
        x = torch.zeros(num_atoms, num_atom_types)

        # Random atom types
        atom_indices = torch.randint(0, num_atom_types, (num_atoms,))
        x[torch.arange(num_atoms), atom_indices] = 1

        # Create edges (adjacency matrix)
        adj = np.random.rand(num_atoms, num_atoms)
        adj = (adj < edge_density).astype(int)
        np.fill_diagonal(adj, 0)  # No self-loops
        adj = np.maximum(adj, adj.T)  # Make symmetric

        # Convert to edge list
        edge_index, edge_attr = dense_to_sparse(torch.tensor(adj, dtype=torch.float))

        # Edge features (bond types)
        num_edges = edge_index.shape[1]
        if num_edges > 0:
            edge_attr = torch.randint(0, len(MoleculeGraph.BOND_TYPES), (num_edges,)).float()
        else:
            edge_attr = torch.tensor([], dtype=torch.float)

        # Create PyG Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_atoms)

        return data

    @staticmethod
    def from_adjacency(adj_matrix: np.ndarray, node_features: Optional[np.ndarray] = None) -> Data:
        """
        Create graph from adjacency matrix

        Args:
            adj_matrix: (N x N) adjacency matrix
            node_features: (N x F) node feature matrix

        Returns:
            PyTorch Geometric Data object
        """
        adj_tensor = torch.tensor(adj_matrix, dtype=torch.float)
        edge_index, edge_attr = dense_to_sparse(adj_tensor)

        # Node features
        if node_features is None:
            num_atoms = adj_matrix.shape[0]
            x = torch.ones(num_atoms, 1)
        else:
            x = torch.tensor(node_features, dtype=torch.float)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                   num_nodes=adj_matrix.shape[0])

        return data

    @staticmethod
    def compute_molecular_properties(num_atoms: int, edge_density: float) -> Dict:
        """Compute theoretical molecular properties from graph structure"""
        return {
            'num_atoms': num_atoms,
            'molecular_weight': 12.0 * num_atoms + np.random.randn() * 5,
            'LogP': np.random.randn(),  # Lipophilicity
            'HBD': max(0, int(num_atoms * edge_density * 0.3)),  # Hydrogen bond donors
            'HBA': max(0, int(num_atoms * edge_density * 0.4)),  # Acceptors
            'TPSA': 40 + np.random.randn() * 30,  # Topological Polar Surface Area
            'RotBonds': max(0, int(edge_density * num_atoms * 0.2)),
            'lipinski_violations': max(0, int(np.random.rand() * 2)),
        }


class GNNEncoder(nn.Module):
    """
    Graph Neural Network for molecular property prediction
    Uses graph convolutions to learn node and graph embeddings
    """

    def __init__(self, in_channels: int = 12, hidden_channels: int = 64,
                 out_channels: int = 32, num_layers: int = 3, dropout: float = 0.1):
        """
        Initialize GNN encoder

        Args:
            in_channels: Input node feature dimension
            hidden_channels: Hidden layer dimension
            out_channels: Output embedding dimension
            num_layers: Number of GNN layers
            dropout: Dropout rate
        """
        super(GNNEncoder, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout

        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # First layer
        self.convs.append(GraphConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GraphConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Output layer
        self.convs.append(GraphConv(hidden_channels, out_channels))

        # Global pooling and MLP head
        self.mlp_head = nn.Sequential(
            nn.Linear(out_channels * 2, hidden_channels),  # *2 for mean+max pool
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through GNN

        Args:
            x: Node feature matrix (N x in_channels)
            edge_index: Edge index tensor (2 x E)
            batch: Batch assignment for graphs

        Returns:
            Graph embedding (batch_size x out_channels)
        """
        # Graph convolutions
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Final convolution (no activation)
        x = self.convs[-1](x, edge_index)

        # Global pooling
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_pooled = torch.cat([x_mean, x_max], dim=1)

        # MLP head
        graph_embedding = self.mlp_head(x_pooled)

        return graph_embedding


class PropertyPredictor(nn.Module):
    """Predict molecular properties from GNN embeddings"""

    def __init__(self, embedding_dim: int = 32, num_properties: int = 6, hidden_dim: int = 32):
        """
        Initialize property predictor

        Args:
            embedding_dim: Dimension of GNN embedding
            num_properties: Number of properties to predict
            hidden_dim: Hidden layer dimension
        """
        super(PropertyPredictor, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_properties)
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """Predict properties from embedding"""
        return self.mlp(embedding)


class MolecularGNNPipeline:
    """Complete pipeline for molecular encoding and property prediction"""

    def __init__(self, device: str = 'cpu', learning_rate: float = 1e-3):
        """
        Initialize pipeline

        Args:
            device: 'cpu' or 'cuda'
            learning_rate: Optimizer learning rate
        """
        self.device = torch.device(device)

        # Models
        self.gnn_encoder = GNNEncoder(
            in_channels=len(MoleculeGraph.ATOM_TYPES),
            hidden_channels=64,
            out_channels=32,
            num_layers=3
        ).to(self.device)

        self.property_predictor = PropertyPredictor(
            embedding_dim=32,
            num_properties=6,
            hidden_dim=32
        ).to(self.device)

        # Optimizer
        params = list(self.gnn_encoder.parameters()) + list(self.property_predictor.parameters())
        self.optimizer = torch.optim.Adam(params, lr=learning_rate)

        # Loss function
        self.loss_fn = nn.MSELoss()

        self.training_history = {'loss': [], 'property_loss': []}

    def encode_molecule(self, graph: Data) -> torch.Tensor:
        """
        Encode a molecule graph into embedding

        Args:
            graph: PyTorch Geometric Data object

        Returns:
            Graph embedding vector
        """
        graph = graph.to(self.device)
        with torch.no_grad():
            embedding = self.gnn_encoder(graph.x, graph.edge_index)
        return embedding.detach().cpu()

    def predict_properties(self, graph: Data) -> Dict[str, float]:
        """
        Predict molecular properties

        Args:
            graph: PyTorch Geometric Data object

        Returns:
            Dictionary of predicted properties
        """
        embedding = self.encode_molecule(graph)
        embedding = embedding.to(self.device)

        with torch.no_grad():
            properties = self.property_predictor(embedding)

        properties = properties.detach().cpu().numpy()[0]

        return {
            'molecular_weight': float(properties[0]) * 100 + 150,  # Denormalize
            'LogP': float(properties[1]),
            'HBD': max(0, int(properties[2])),
            'HBA': max(0, int(properties[3])),
            'TPSA': float(properties[4]) * 50 + 40,
            'RotBonds': max(0, int(properties[5]))
        }

    def train_step(self, graphs: List[Data], targets: torch.Tensor) -> float:
        """
        Single training step

        Args:
            graphs: List of molecular graphs
            targets: Target properties (batch_size x num_properties)

        Returns:
            Loss value
        """
        self.gnn_encoder.train()
        self.property_predictor.train()

        # Encode all graphs
        embeddings = []
        for graph in graphs:
            graph = graph.to(self.device)
            emb = self.gnn_encoder(graph.x, graph.edge_index)
            embeddings.append(emb)

        embeddings = torch.cat(embeddings, dim=0)
        targets = targets.to(self.device)

        # Predict and compute loss
        predictions = self.property_predictor(embeddings)
        loss = self.loss_fn(predictions, targets)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train_epoch(self, graphs_list: List[List[Data]], targets_list: List[torch.Tensor],
                   batch_size: int = 4) -> float:
        """
        Train for one epoch

        Args:
            graphs_list: List of graph batches
            targets_list: List of target batches
            batch_size: Batch size

        Returns:
            Average loss
        """
        total_loss = 0

        for graphs, targets in zip(graphs_list, targets_list):
            loss = self.train_step(graphs, targets)
            total_loss += loss

        avg_loss = total_loss / len(graphs_list)
        self.training_history['loss'].append(avg_loss)

        return avg_loss

    def evaluate(self, graphs: List[Data], targets: torch.Tensor) -> float:
        """
        Evaluate on a batch of graphs

        Args:
            graphs: List of molecular graphs
            targets: Target properties

        Returns:
            MSE loss
        """
        self.gnn_encoder.eval()
        self.property_predictor.eval()

        embeddings = []
        for graph in graphs:
            graph = graph.to(self.device)
            with torch.no_grad():
                emb = self.gnn_encoder(graph.x, graph.edge_index)
            embeddings.append(emb)

        embeddings = torch.cat(embeddings, dim=0)
        targets = targets.to(self.device)

        with torch.no_grad():
            predictions = self.property_predictor(embeddings)
            loss = self.loss_fn(predictions, targets)

        return loss.item()


class ReactionNetworkToGraph:
    """Convert reaction network to molecular graph representation"""

    @staticmethod
    def from_reaction_network(reaction_network) -> Data:
        """
        Convert reaction network to graph

        Each species = node
        Each reaction = edge with rate as weight
        """
        num_species = reaction_network.num_species
        adj = reaction_network.get_adjacency_matrix()

        # Node features (simple: one-hot by species)
        x = torch.eye(num_species)

        # Convert adjacency to edge list
        adj_tensor = torch.tensor(adj, dtype=torch.float)
        edge_index, edge_attr = dense_to_sparse(adj_tensor)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                   num_nodes=num_species)

        return data


if __name__ == "__main__":
    # Test GNN pipeline
    print("Testing Molecular GNN Pipeline...")

    # Create random molecules
    num_molecules = 8
    graphs = [MoleculeGraph.create_simple_graph(num_atoms=np.random.randint(5, 15))
             for _ in range(num_molecules)]

    # Create target properties
    targets = torch.randn(num_molecules, 6)

    # Initialize pipeline
    pipeline = MolecularGNNPipeline()

    # Test single prediction
    print("\nPredicting properties for first molecule...")
    props = pipeline.predict_properties(graphs[0])
    print(f"Predicted properties: {props}")

    # Train on small batch
    print("\nTraining for 3 epochs...")
    for epoch in range(3):
        loss = pipeline.train_step(graphs, targets)
        print(f"  Epoch {epoch + 1}: Loss = {loss:.4f}")

    print("\nGNN Pipeline test complete!")
