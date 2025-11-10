"""Reinforcement learning agents for trade policy optimisation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

try:  # pragma: no cover
    import torch
    from torch import Tensor, nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    Tensor = object  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


@dataclass
class DeepQConfig:
    state_dim: int
    action_dim: int
    hidden_dim: int = 128
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 32
    buffer_size: int = 10_000
    min_buffer: int = 256
    device: str = "cpu"


if nn is not None:

    class QNetwork(nn.Module):
        def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
            )

        def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
            return self.net(x)

else:  # pragma: no cover

    class QNetwork:  # type: ignore[misc]
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required for QNetwork")


class ReplayBuffer:
    def __init__(self, capacity: int, device: str = "cpu"):
        self.capacity = capacity
        self.device = device
        self._data: List[Tuple[Tensor, int, float, Tensor, bool]] = []
        self._index = 0

    def push(self, state: Tensor, action: int, reward: float, next_state: Tensor, done: bool) -> None:
        if len(self._data) < self.capacity:
            self._data.append((state, action, reward, next_state, done))
        else:
            self._data[self._index] = (state, action, reward, next_state, done)
        self._index = (self._index + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        indices = torch.randint(0, len(self._data), (batch_size,))
        states, actions, rewards, next_states, dones = zip(*(self._data[i] for i in indices))
        return (
            torch.stack(states).to(self.device),
            torch.tensor(actions, dtype=torch.int64, device=self.device),
            torch.tensor(rewards, dtype=torch.float32, device=self.device),
            torch.stack(next_states).to(self.device),
            torch.tensor(dones, dtype=torch.bool, device=self.device),
        )

    def __len__(self) -> int:
        return len(self._data)


class DeepQTradingAgent:
    def __init__(self, config: DeepQConfig):
        if torch is None or nn is None:
            raise RuntimeError("PyTorch is required for DeepQTradingAgent")
        self.config = config
        self.device = torch.device(config.device)
        self.online = QNetwork(config.state_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.target = QNetwork(config.state_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.lr)
        self.buffer = ReplayBuffer(config.buffer_size, device=config.device)
        self.steps = 0
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

    def select_action(self, state: Tensor) -> int:
        if torch.rand(()) < self.epsilon:
            return int(torch.randint(0, self.config.action_dim, (1,)))
        state = state.to(self.device)
        q_values = self.online(state.unsqueeze(0))
        return int(torch.argmax(q_values, dim=1).item())

    def push_transition(self, state: Tensor, action: int, reward: float, next_state: Tensor, done: bool) -> None:
        self.buffer.push(state.detach().cpu(), action, reward, next_state.detach().cpu(), done)

    def update(self) -> Optional[float]:
        if len(self.buffer) < self.config.min_buffer:
            return None
        states, actions, rewards, next_states, dones = self.buffer.sample(self.config.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        q_values = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target(next_states).max(1).values
            targets = rewards + self.config.gamma * next_q * (~dones)
        loss = nn.functional.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 1.0)
        self.optimizer.step()
        self.steps += 1
        if self.steps % 20 == 0:
            self.target.load_state_dict(self.online.state_dict())
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        return float(loss.detach())


@dataclass
class PPOConfig:
    state_dim: int
    action_dim: int
    hidden_dim: int = 64
    gamma: float = 0.99
    clip_ratio: float = 0.2
    lr: float = 3e-4
    update_epochs: int = 4
    device: str = "cpu"


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: Tensor) -> Tuple[Tensor, Tensor]:  # type: ignore[override]
        features = self.net(state)
        logits = self.policy_head(features)
        value = self.value_head(features)
        return logits, value.squeeze(-1)


class PPOTradingAgent:
    def __init__(self, config: PPOConfig):
        if torch is None or nn is None:
            raise RuntimeError("PyTorch is required for PPOTradingAgent")
        self.config = config
        self.device = torch.device(config.device)
        self.network = PolicyNetwork(config.state_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config.lr)

    def act(self, state: Tensor) -> Tuple[int, Tensor, Tensor]:
        state = state.to(self.device)
        logits, value = self.network(state.unsqueeze(0))
        distribution = torch.distributions.Categorical(logits=logits)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        return int(action.item()), log_prob.squeeze(0), value.squeeze(0)

    def update(self, trajectories: List[Tuple[Tensor, Tensor, float, Tensor, Tensor]]) -> float:
        returns = []
        discounted = 0.0
        for _, _, reward, _, value in reversed(trajectories):
            discounted = reward + self.config.gamma * discounted
            returns.insert(0, discounted)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)
        states = torch.stack([transition[0] for transition in trajectories]).to(self.device)
        actions = torch.tensor([transition[1] for transition in trajectories], dtype=torch.int64, device=self.device)
        old_log_probs = torch.stack([transition[3] for transition in trajectories]).to(self.device)
        values = torch.stack([transition[4] for transition in trajectories]).to(self.device)
        advantages = returns_tensor - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        total_loss = 0.0
        for _ in range(self.config.update_epochs):
            logits, new_values = self.network(states)
            distribution = torch.distributions.Categorical(logits=logits)
            new_log_probs = distribution.log_prob(actions)
            ratio = (new_log_probs - old_log_probs).exp()
            surrogate1 = ratio * advantages
            surrogate2 = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * advantages
            actor_loss = -torch.min(surrogate1, surrogate2).mean()
            critic_loss = nn.functional.mse_loss(new_values.squeeze(-1), returns_tensor)
            loss = actor_loss + 0.5 * critic_loss
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()
            total_loss += float(loss.detach())
        return total_loss / self.config.update_epochs


__all__ = [
    "DeepQConfig",
    "DeepQTradingAgent",
    "PPOConfig",
    "PPOTradingAgent",
]
