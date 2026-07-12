import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i : i + self.batch_size] for i in batch_start]

        return batches

    def get_memories(self):
        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.probs),
            np.array(self.vals),
            np.array(self.rewards),
            np.array(self.dones),
        )

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []


class ActorNetwork(nn.Module):
    def __init__(
        self,
        n_actions,
        input_dims,
        alpha,
        device,
        fc1_dims=100,
        fc2_dims=100,
        fc3_dims=100,
        chkpt_dir="tmp/docker/models",
    ):
        super(ActorNetwork, self).__init__()

        os.makedirs(chkpt_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(
            chkpt_dir, f"actor_torch_ppo_{int(os.getenv('REPLICA_ID', '0'))}"
        )
        self.actor = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, fc3_dims),
            nn.ReLU(),
            nn.Linear(fc3_dims, n_actions),
            nn.Softmax(dim=-1),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = device
        self.to(self.device)

    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)
        return dist

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            return self.actor(state).argmax().item()

    def act_batch(self, states):
        states = torch.as_tensor(np.asarray(states), dtype=torch.float32).to(
            self.device
        )
        with torch.no_grad():
            probs = self.actor(states)
        return probs.argmax(dim=-1).cpu().numpy()

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file, map_location=self.device))


class CriticNetwork(nn.Module):
    def __init__(
        self,
        input_dims,
        alpha,
        device,
        fc1_dims=100,
        fc2_dims=100,
        fc3_dims=100,
        chkpt_dir="tmp/docker/models",
    ):
        super(CriticNetwork, self).__init__()

        os.makedirs(chkpt_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(
            chkpt_dir, f"critic_torch_ppo_{int(os.getenv('REPLICA_ID', '0'))}"
        )
        self.critic = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, fc3_dims),
            nn.ReLU(),
            nn.Linear(fc3_dims, 1),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = device
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)

        return value

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file, map_location=self.device))


def load_policy(path, n_actions, input_dims, device=torch.device("cpu")):
    """Load a trained actor for inference.

    Accepts a raw actor state_dict file, a run checkpoint.pt (as written by
    main_vec.py), or a directory containing an actor checkpoint.
    """
    if os.path.isdir(path):
        candidates = sorted(
            f
            for f in os.listdir(path)
            if f.startswith(("actor", "new_actor")) or f == "checkpoint.pt"
        )
        if not candidates:
            raise FileNotFoundError(f"no actor checkpoint found in {path}")
        path = os.path.join(path, candidates[0])
    payload = torch.load(path, map_location=device, weights_only=False)
    if isinstance(payload, dict) and "agent" in payload:
        payload = payload["agent"]["actor"]
    actor = ActorNetwork(n_actions, input_dims, alpha=1e-4, device=device)
    actor.load_state_dict(payload)
    actor.eval()
    return actor


class Agent:
    def __init__(
        self,
        n_actions,
        input_dims,
        gamma=0.99,
        alpha=0.0003,
        beta=0.001,
        gae_lambda=0.95,
        policy_clip=0.2,
        batch_size=64,
        n_epochs=10,
        entropy_coef=0.013,
        max_grad_norm=0.5,
        target_kl=0.05,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl

        self.batch_size = batch_size
        self.actor = ActorNetwork(n_actions, input_dims, alpha, device)
        self.critic = CriticNetwork(input_dims, beta, device)
        self.memory = PPOMemory(batch_size)

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print("... saving models ...")
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print("... loading models ...")
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def get_checkpoint(self):
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor.optimizer.state_dict(),
            "critic_optimizer": self.critic.optimizer.state_dict(),
        }

    def load_checkpoint_state(self, ckpt):
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor.optimizer.load_state_dict(ckpt["actor_optimizer"])
        self.critic.optimizer.load_state_dict(ckpt["critic_optimizer"])

    def choose_action(self, observation):
        state = torch.tensor(observation, dtype=torch.float).to(self.actor.device)

        with torch.no_grad():
            dist = self.actor(state)
            value = self.critic(state)
            action = dist.sample()
            probs = dist.log_prob(action)

        probs = torch.squeeze(probs).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()

        return action, probs, value

    def choose_action_batch(self, observations):
        states = torch.as_tensor(np.asarray(observations), dtype=torch.float32).to(
            self.actor.device
        )

        with torch.no_grad():
            dist = self.actor(states)
            values = self.critic(states)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)

        return (
            actions.cpu().numpy(),
            log_probs.cpu().numpy(),
            values.squeeze(-1).cpu().numpy(),
        )

    def calculate_advantage_old(self):
        (
            state_arr,
            action_arr,
            old_prob_arr,
            values,
            reward_arr,
            dones_arr,
        ) = self.memory.get_memories()

        advantage = np.zeros(len(reward_arr), dtype=np.float32)

        for t in range(len(reward_arr) - 1):
            discount = 1
            a_t = 0
            for k in range(t, len(reward_arr) - 1):
                a_t += discount * (
                    reward_arr[k]
                    + self.gamma * values[k + 1] * (1 - int(dones_arr[k]))
                    - values[k]
                )
                discount *= self.gamma * self.gae_lambda
            advantage[t] = a_t

        return (
            advantage,
            state_arr,
            action_arr,
            old_prob_arr,
            values,
            reward_arr,
            dones_arr,
        )

    def calculate_advantage(self):
        (
            state_arr,
            action_arr,
            old_prob_arr,
            values,
            reward_arr,
            dones_arr,
        ) = self.memory.get_memories()

        batch_size = dones_arr.shape[0]

        advantage = np.zeros(batch_size + 1)

        for t in reversed(range(batch_size - 1)):
            delta = (
                reward_arr[t]
                + self.gamma * values[t + 1] * (1 - int(dones_arr[t]))
                - values[t]
            )
            advantage[t] = delta + (
                self.gamma
                * self.gae_lambda
                * advantage[t + 1]
                * (1 - int(dones_arr[t]))
            )

        return (
            advantage[:-1],
            state_arr,
            action_arr,
            old_prob_arr,
            values,
            reward_arr,
            dones_arr,
        )

    def learn(self):
        (
            advantage,
            state_arr,
            action_arr,
            old_prob_arr,
            values,
            reward_arr,
            dones_arr,
        ) = self.calculate_advantage()

        metrics = self.update(state_arr, action_arr, old_prob_arr, values, advantage)

        self.memory.clear_memory()

        return metrics

    def update(self, state_arr, action_arr, old_prob_arr, values, advantage):
        device = self.actor.device
        advantage = torch.tensor(advantage, dtype=torch.float32).to(device)
        values = torch.tensor(values, dtype=torch.float32).to(device)
        # convert the rollout once; minibatches below index these tensors
        states_all = torch.tensor(
            np.asarray(state_arr), dtype=torch.float32
        ).to(device)
        actions_all = torch.tensor(np.asarray(action_arr)).to(device)
        old_probs_all = torch.tensor(
            np.asarray(old_prob_arr), dtype=torch.float32
        ).to(device)

        # explained variance of the rollout value predictions, before this update
        returns_all = advantage + values
        explained_variance = (
            1 - torch.var(returns_all - values) / (torch.var(returns_all) + 1e-8)
        ).item()

        actor_losses = []
        critic_losses = []
        entropies = []
        approx_kls = []
        clip_fractions = []
        grad_norms_actor = []
        grad_norms_critic = []

        n_samples = len(state_arr)
        epochs_used = 0
        stop = False
        for _ in range(self.n_epochs):
            if stop:
                break
            epochs_used += 1
            indices = np.arange(n_samples, dtype=np.int64)
            np.random.shuffle(indices)
            batches = [
                torch.as_tensor(indices[i : i + self.batch_size], device=device)
                for i in range(0, n_samples, self.batch_size)
            ]

            for batch in batches:
                states = states_all[batch]
                old_probs = old_probs_all[batch]
                actions = actions_all[batch]

                dist = self.actor(states)
                critic_value = self.critic(states)

                critic_value = torch.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = (new_probs - old_probs).exp()

                # returns use the raw advantage; the actor loss uses it normalized
                returns = advantage[batch] + values[batch]
                adv_batch = advantage[batch]
                adv_batch = (adv_batch - adv_batch.mean()) / (adv_batch.std() + 1e-8)

                weighted_probs = adv_batch * prob_ratio
                weighted_clipped_probs = (
                    torch.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip)
                    * adv_batch
                )
                entropy = dist.entropy().mean()
                actor_loss = (
                    -torch.min(weighted_probs, weighted_clipped_probs).mean()
                    - self.entropy_coef * entropy
                )

                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5 * critic_loss

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                grad_norm_actor = nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.max_grad_norm
                )
                grad_norm_critic = nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.max_grad_norm
                )
                self.actor.optimizer.step()
                self.critic.optimizer.step()

                with torch.no_grad():
                    log_ratio = new_probs - old_probs
                    approx_kl = ((prob_ratio - 1) - log_ratio).mean().item()
                    clip_fraction = (
                        ((prob_ratio - 1).abs() > self.policy_clip)
                        .float()
                        .mean()
                        .item()
                    )

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropies.append(entropy.item())
                approx_kls.append(approx_kl)
                clip_fractions.append(clip_fraction)
                grad_norms_actor.append(grad_norm_actor.item())
                grad_norms_critic.append(grad_norm_critic.item())

                if self.target_kl is not None and approx_kl > self.target_kl:
                    stop = True  # policy moved too far; skip remaining epochs
                    break

        return {
            "epochs_used": float(epochs_used),
            "actor_loss": float(np.mean(actor_losses)),
            "critic_loss": float(np.mean(critic_losses)),
            "entropy": float(np.mean(entropies)),
            "approx_kl": float(np.mean(approx_kls)),
            "clip_fraction": float(np.mean(clip_fractions)),
            "explained_variance": explained_variance,
            "grad_norm_actor": float(np.mean(grad_norms_actor)),
            "grad_norm_critic": float(np.mean(grad_norms_critic)),
        }
