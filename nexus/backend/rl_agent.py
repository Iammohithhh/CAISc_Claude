"""
ChemNeuron Phase 4: Advanced Reinforcement Learning Agent
Discovers novel reaction networks with learning capacity
Uses Actor-Critic architecture and multi-objective optimization
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class NetworkGenerator:
    """Generate reaction network topologies"""

    @staticmethod
    def random_network(num_species: int, edge_prob: float = 0.2):
        """Generate random network adjacency"""
        adj = np.random.rand(num_species, num_species) < edge_prob
        np.fill_diagonal(adj, False)
        return adj.astype(float)

    @staticmethod
    def add_reaction(adj: np.ndarray, src: int, tgt: int, rate: float) -> np.ndarray:
        """Add reaction to adjacency matrix"""
        adj_new = adj.copy()
        adj_new[src, tgt] = rate
        return adj_new

    @staticmethod
    def remove_reaction(adj: np.ndarray, src: int, tgt: int) -> np.ndarray:
        """Remove reaction"""
        adj_new = adj.copy()
        adj_new[src, tgt] = 0.0
        return adj_new

    @staticmethod
    def modify_rate(adj: np.ndarray, src: int, tgt: int, new_rate: float) -> np.ndarray:
        """Modify reaction rate"""
        adj_new = adj.copy()
        if adj_new[src, tgt] > 0:
            adj_new[src, tgt] = np.clip(new_rate, 0.01, 2.0)
        return adj_new


class ActorNetwork(nn.Module):
    """Actor network for policy"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        """
        Args:
            state_dim: Dimension of network state (flattened adjacency + features)
            action_dim: Number of possible actions
            hidden_dim: Hidden layer dimension
        """
        super(ActorNetwork, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return action probabilities"""
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        action_probs = self.softmax(self.fc3(x))
        return action_probs


class CriticNetwork(nn.Module):
    """Critic network for value function"""

    def __init__(self, state_dim: int, hidden_dim: int = 128):
        """
        Args:
            state_dim: Dimension of network state
            hidden_dim: Hidden layer dimension
        """
        super(CriticNetwork, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        self.relu = nn.ReLU()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return value estimate"""
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        value = self.fc3(x)
        return value


class NetworkEnv:
    """Environment for reaction network discovery"""

    def __init__(self, num_species: int = 4, max_reactions: int = 12):
        """
        Args:
            num_species: Number of chemical species
            max_reactions: Maximum number of reactions allowed
        """
        self.num_species = num_species
        self.max_reactions = max_reactions
        self.action_space_size = (
            2 +  # Add or remove reaction
            num_species * num_species  # Which reaction (src, tgt)
            + 10  # Rate modification
        )

        self.state_dim = num_species * num_species + num_species  # Adjacency + features

        self.reset()

    def reset(self) -> np.ndarray:
        """Reset to random network"""
        edge_prob = np.random.uniform(0.1, 0.3)
        self.adjacency = NetworkGenerator.random_network(self.num_species, edge_prob)
        self.step_count = 0
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Flatten network state"""
        # Adjacency matrix + number of reactions (feature)
        num_reactions = np.count_nonzero(self.adjacency)
        features = np.array([num_reactions / self.max_reactions,
                            np.sum(self.adjacency) / 10.0])

        state = np.concatenate([self.adjacency.flatten(), features])
        return state.astype(np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Execute action

        Args:
            action: Action index

        Returns:
            (next_state, reward, done)
        """
        self.step_count += 1
        done = self.step_count > 50

        # Decode action
        action_type = action % 2  # 0=add, 1=modify
        reaction_id = (action // 2) % (self.num_species ** 2)
        rate_mod_id = (action // (2 * self.num_species ** 2)) % 10

        src = reaction_id // self.num_species
        tgt = reaction_id % self.num_species

        if src == tgt:
            return self._get_state(), -0.1, done

        reward = 0.0

        if action_type == 0:  # Add/modify reaction
            if self.adjacency[src, tgt] == 0 and np.count_nonzero(self.adjacency) < self.max_reactions:
                new_rate = 0.1 + 0.2 * (rate_mod_id / 10.0)
                self.adjacency = NetworkGenerator.add_reaction(self.adjacency, src, tgt, new_rate)
                reward += 0.1
            else:
                # Modify existing rate
                if self.adjacency[src, tgt] > 0:
                    delta_rate = 0.3 + 0.1 * (rate_mod_id - 5)
                    new_rate = self.adjacency[src, tgt] + delta_rate
                    self.adjacency = NetworkGenerator.modify_rate(self.adjacency, src, tgt, new_rate)
                    reward += 0.05
        else:  # Remove reaction
            if self.adjacency[src, tgt] > 0:
                self.adjacency = NetworkGenerator.remove_reaction(self.adjacency, src, tgt)
                reward -= 0.05

        # Reward structure (multi-objective)
        reward += self._compute_network_reward()

        return self._get_state(), reward, done

    def _compute_network_reward(self) -> float:
        """Compute multi-objective reward"""
        reward = 0.0

        # Objective 1: Moderate connectivity (not too sparse, not too dense)
        num_reactions = np.count_nonzero(self.adjacency)
        target_reactions = self.max_reactions // 2
        connectivity_penalty = 0.05 * abs(num_reactions - target_reactions) / target_reactions
        reward -= connectivity_penalty

        # Objective 2: Strong feedback loops (cycles in network)
        feedback_strength = self._detect_feedback()
        reward += 0.1 * feedback_strength

        # Objective 3: Robustness (varied edge weights)
        nonzero_weights = self.adjacency[self.adjacency > 0]
        if len(nonzero_weights) > 0:
            weight_variance = np.var(nonzero_weights)
            reward += 0.05 * (weight_variance / 0.5)  # Normalize

        # Objective 4: Avoid isolated species
        isolated = np.sum((self.adjacency.sum(axis=0) == 0) & (self.adjacency.sum(axis=1) == 0))
        isolated_penalty = 0.1 * (isolated / self.num_species)
        reward -= isolated_penalty

        return reward

    def _detect_feedback(self) -> float:
        """Detect presence of feedback loops"""
        # Simple heuristic: detect cycles
        adj_binary = (self.adjacency > 0).astype(float)

        # Check for paths (simplified)
        feedback = 0.0
        for i in range(self.num_species):
            for j in range(self.num_species):
                if i != j and adj_binary[i, j] > 0:
                    # Check if path exists from j back to i
                    for k in range(self.num_species):
                        if adj_binary[j, k] > 0 and adj_binary[k, i] > 0:
                            feedback += 0.1

        return min(feedback, 1.0)  # Cap at 1.0


class ActorCriticAgent:
    """Actor-Critic RL Agent for network discovery"""

    def __init__(self, state_dim: int, action_dim: int, learning_rate: float = 1e-3,
                 gamma: float = 0.99, device: str = 'cpu'):
        """
        Args:
            state_dim: State dimension
            action_dim: Action space size
            learning_rate: Learning rate
            gamma: Discount factor
            device: 'cpu' or 'cuda'
        """
        self.device = torch.device(device)
        self.gamma = gamma

        # Networks
        self.actor = ActorNetwork(state_dim, action_dim).to(self.device)
        self.critic = CriticNetwork(state_dim).to(self.device)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)

        # Memory
        self.memory = deque(maxlen=1000)

        # Stats
        self.episode_rewards = []
        self.episode_losses = []

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using policy

        Args:
            state: Current state
            training: If True, sample from distribution; if False, take argmax

        Returns:
            Action index
        """
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            action_probs = self.actor(state_tensor)

        if training and random.random() < 0.1:  # Epsilon-greedy exploration
            return random.randint(0, self.actor.fc3.out_features - 1)

        if training:
            action = torch.multinomial(action_probs, 1).item()
        else:
            action = action_probs.argmax().item()

        return action

    def store_transition(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool):
        """Store transition in memory"""
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self, batch_size: int = 32) -> Tuple[float, float]:
        """
        Train on a batch

        Returns:
            (actor_loss, critic_loss)
        """
        if len(self.memory) < batch_size:
            return 0.0, 0.0

        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Critic update
        values = self.critic(states).squeeze()
        next_values = self.critic(next_states).squeeze()
        targets = rewards + self.gamma * next_values * (1 - dones)
        critic_loss = F.mse_loss(values, targets.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        advantages = (targets.detach() - values.detach()).unsqueeze(1)
        action_probs = self.actor(states)
        action_probs_taken = action_probs.gather(1, actions.unsqueeze(1))
        actor_loss = -(torch.log(action_probs_taken) * advantages).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item(), critic_loss.item()

    def train_episode(self, env: NetworkEnv, max_steps: int = 50) -> float:
        """
        Train for one episode

        Returns:
            Episode reward
        """
        state = env.reset()
        episode_reward = 0.0

        for _ in range(max_steps):
            action = self.select_action(state, training=True)
            next_state, reward, done = env.step(action)

            self.store_transition(state, action, reward, next_state, done)
            episode_reward += reward

            state = next_state
            if done:
                break

        # Train on batch
        actor_loss, critic_loss = self.train_step(batch_size=32)

        self.episode_rewards.append(episode_reward)
        self.episode_losses.append((actor_loss + critic_loss) / 2)

        return episode_reward

    def get_network(self, env: NetworkEnv) -> np.ndarray:
        """Get best network found so far"""
        state = env.reset()
        for _ in range(100):
            action = self.select_action(state, training=False)
            next_state, _, done = env.step(action)
            state = next_state
            if done:
                break

        return env.adjacency.copy()


if __name__ == "__main__":
    print("Testing RL Agent...")

    # Create environment
    env = NetworkEnv(num_species=4, max_reactions=12)

    # Create agent
    agent = ActorCriticAgent(
        state_dim=env.state_dim,
        action_dim=env.action_space_size,
        learning_rate=1e-3
    )

    # Train
    print("Training for 20 episodes...")
    for episode in range(20):
        reward = agent.train_episode(env)
        if (episode + 1) % 5 == 0:
            print(f"  Episode {episode + 1}: Reward = {reward:.4f}")

    print("\nRL Agent test complete!")
