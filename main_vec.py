import argparse
import json
import os
import time
from collections import deque
from itertools import count

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from ppo_torch import Agent
from pong import make

FRAME_SKIP = 4
SCORE_THRESHOLD = 0.4
MIN_EPISODES_BETWEEN_PROMOTIONS = 100
TRANSITIONS_PER_UPDATE = 4096
CHECKPOINT_INTERVAL = 50  # updates between periodic checkpoints


def parse_args():
    parser = argparse.ArgumentParser(description="Vectorized PPO self-play trainer")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        default=None,
        metavar="RUN_DIR",
        help="resume a previous run from RUN_DIR "
        "(bare --resume uses the run pointed to by runs/latest)",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=32,
        help="parallel game instances (default: 32)",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="directory name for a new run under runs/ (default: timestamped)",
    )
    parser.add_argument(
        "--init-weights",
        default=None,
        metavar="MODEL_DIR",
        help="start a NEW run but initialize agent and opponent from "
        "actor/critic checkpoint files in MODEL_DIR (fresh optimizers/counters)",
    )
    args = parser.parse_args()
    if args.resume and (args.run_name or args.init_weights):
        parser.error("--resume cannot be combined with --run-name/--init-weights")
    return args


def invert_batch(observations):
    # mirror the state so the opponent sees itself as the right-side player
    inverted = observations.copy()
    inverted[:, 0] = observations[:, 1]
    inverted[:, 1] = observations[:, 0]
    inverted[:, 2] = 1.0 - observations[:, 2]
    inverted[:, 4] = -observations[:, 4]
    return inverted


def update_latest_symlink(run_dir):
    link = os.path.join("runs", "latest")
    if os.path.islink(link):
        os.remove(link)
    if not os.path.exists(link):
        os.symlink(os.path.basename(os.path.normpath(run_dir)), link)


def save_checkpoint(run_dir, agent, agent1, train_state):
    path = os.path.join(run_dir, "checkpoint.pt")
    checkpoint = {
        "agent": agent.get_checkpoint(),
        "opponent_actor": agent1.actor.state_dict(),
        "train_state": train_state,
    }
    torch.save(checkpoint, path + ".tmp")
    os.replace(path + ".tmp", path)


if __name__ == "__main__":
    args = parse_args()

    # CPU beats GPU for networks this small (per-call CUDA overhead dominates)
    device = torch.device("cpu")

    num_envs = args.num_envs
    rollout_steps = TRANSITIONS_PER_UPDATE // num_envs
    replica_id = int(os.getenv("REPLICA_ID", "0"))

    checkpoint = None
    if args.resume:
        run_dir = (
            os.path.join("runs", "latest") if args.resume == "latest" else args.resume
        )
        run_dir = os.path.realpath(run_dir)
        checkpoint_path = os.path.join(run_dir, "checkpoint.pt")
        if not os.path.isfile(checkpoint_path):
            raise SystemExit(f"no checkpoint found at {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        print(
            f"resuming {run_dir} from step {checkpoint['train_state']['n_steps']:,}"
        )
    else:
        run_name = args.run_name or time.strftime(
            f"ppo_pong_vec_replica{replica_id}_%Y%m%d_%H%M%S"
        )
        run_dir = os.path.join("runs", run_name)
        if os.path.exists(run_dir):
            raise SystemExit(
                f"{run_dir} already exists; use --resume to continue it "
                "or pick a different --run-name"
            )
        os.makedirs(run_dir)
        print(f"starting new run: {run_dir}")
    update_latest_symlink(run_dir)

    envs = [make("Pong-v0") for _ in range(num_envs)]
    num_actions = envs[0].num_actions
    state_size = envs[0].state_size

    agent = Agent(
        n_actions=num_actions,
        input_dims=(state_size,),
        batch_size=256,
        device=device,
    )

    # frozen opponent: replaced with a copy of the agent on each promotion
    agent1 = Agent(n_actions=num_actions, input_dims=(state_size,), device=device)

    if checkpoint:
        agent.load_checkpoint_state(checkpoint["agent"])
        agent1.actor.load_state_dict(checkpoint["opponent_actor"])
    elif args.init_weights:
        actor_file = os.path.join(args.init_weights, f"actor_torch_ppo_{replica_id}")
        critic_file = os.path.join(args.init_weights, f"critic_torch_ppo_{replica_id}")
        agent.actor.load_state_dict(torch.load(actor_file, map_location=device))
        agent.critic.load_state_dict(torch.load(critic_file, map_location=device))
        agent1.actor.load_state_dict(agent.actor.state_dict())
        print(f"initialized weights from {args.init_weights}")
    else:
        agent1.actor.load_state_dict(agent.actor.state_dict())
    agent1.actor.eval()

    train_state = checkpoint["train_state"] if checkpoint else {}
    n_steps = train_state.get("n_steps", 0)
    start_update = train_state.get("update", 0)
    promotions = train_state.get("promotions", 0)
    episodes_done = train_state.get("episodes_done", 0)
    episodes_since_promotion = train_state.get("episodes_since_promotion", 0)
    score_history = deque(train_state.get("score_history", [-1.0] * 100), maxlen=100)
    avg_scores = train_state.get("avg_scores", [])

    def snapshot_train_state(update):
        return {
            "n_steps": n_steps,
            "update": update + 1,
            "promotions": promotions,
            "episodes_done": episodes_done,
            "episodes_since_promotion": episodes_since_promotion,
            "score_history": list(score_history),
            "avg_scores": avg_scores,
            "num_envs": num_envs,
        }

    gamma = agent.gamma
    gae_lambda = agent.gae_lambda

    json_save_directory = "tmp/docker/json_files/"
    os.makedirs(json_save_directory, exist_ok=True)
    json_file_path = f"{json_save_directory}avg_score_replica{replica_id}.json"

    # same log_dir on resume -> TensorBoard curves continue seamlessly
    writer = SummaryWriter(log_dir=run_dir)

    observations = np.array([env.reset() for env in envs], dtype=np.float32)
    ep_scores = np.zeros(num_envs)
    ep_lengths = np.zeros(num_envs, dtype=int)
    recent_lengths = deque(maxlen=500)

    for update in count(start_update):
        update_start = time.time()

        states_buf = np.zeros((rollout_steps, num_envs, state_size), dtype=np.float32)
        actions_buf = np.zeros((rollout_steps, num_envs), dtype=np.int64)
        logprobs_buf = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        values_buf = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        rewards_buf = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        dones_buf = np.zeros((rollout_steps, num_envs), dtype=np.float32)

        for t in range(rollout_steps):
            actions, log_probs, values = agent.choose_action_batch(observations)
            bot_actions = agent1.actor.act_batch(invert_batch(observations))

            states_buf[t] = observations
            actions_buf[t] = actions
            logprobs_buf[t] = log_probs
            values_buf[t] = values

            for k, env in enumerate(envs):
                done = False
                reward = 0
                for _ in range(FRAME_SKIP):
                    obs_, _, reward, done = env.step([bot_actions[k], actions[k]])
                    if done:
                        break
                rewards_buf[t, k] = reward
                dones_buf[t, k] = float(done)
                ep_scores[k] += reward
                ep_lengths[k] += 1
                if done:
                    score_history.append(ep_scores[k])
                    recent_lengths.append(ep_lengths[k])
                    episodes_done += 1
                    episodes_since_promotion += 1
                    ep_scores[k] = 0.0
                    ep_lengths[k] = 0
                    # env auto-resets inside step(); fetch the fresh state
                    observations[k] = env.get_state()
                else:
                    observations[k] = obs_
            n_steps += num_envs

        # GAE across the rollout, vectorized over envs, bootstrapped past the horizon
        _, _, next_values = agent.choose_action_batch(observations)
        advantages = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        last_adv = np.zeros(num_envs, dtype=np.float32)
        for t in reversed(range(rollout_steps)):
            mask = 1.0 - dones_buf[t]
            delta = rewards_buf[t] + gamma * next_values * mask - values_buf[t]
            last_adv = delta + gamma * gae_lambda * mask * last_adv
            advantages[t] = last_adv
            next_values = values_buf[t]

        metrics = agent.update(
            states_buf.reshape(-1, state_size),
            actions_buf.reshape(-1),
            logprobs_buf.reshape(-1),
            values_buf.reshape(-1),
            advantages.reshape(-1),
        )

        avg_score = float(np.mean(score_history))
        avg_scores.append(avg_score)
        steps_per_sec = TRANSITIONS_PER_UPDATE / (time.time() - update_start)

        for key, value in metrics.items():
            writer.add_scalar(f"train/{key}", value, n_steps)
        writer.add_scalar("episode/avg_score_100", avg_score, n_steps)
        if recent_lengths:
            writer.add_scalar("episode/length", float(np.mean(recent_lengths)), n_steps)
        writer.add_scalar("episode/count", episodes_done, n_steps)
        writer.add_scalar("selfplay/promotions", promotions, n_steps)
        writer.add_scalar("perf/steps_per_sec", steps_per_sec, n_steps)

        if (update + 1) % 25 == 0:
            avg_score_data = {
                "replica_id": replica_id,
                "avg_scores": avg_scores,
                "episodes_time": [],
            }
            with open(json_file_path, "w") as json_file:
                json.dump(avg_score_data, json_file)

        promoted = (
            avg_score >= SCORE_THRESHOLD
            and episodes_since_promotion >= MIN_EPISODES_BETWEEN_PROMOTIONS
        )
        if promoted:
            promotions += 1
            print(f"\npromotion {promotions}: saving models, updating opponent")
            agent.save_models()  # champion export for play.py / tournament.py
            agent1.actor.load_state_dict(agent.actor.state_dict())
            agent1.actor.eval()
            score_history.extend([-1.0] * 100)
            episodes_since_promotion = 0

        if promoted or (update + 1) % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(run_dir, agent, agent1, snapshot_train_state(update))

        print(
            "update",
            update,
            "episodes",
            episodes_done,
            "avg score %.2f" % avg_score,
            "steps %d" % n_steps,
            "steps/s %d" % steps_per_sec,
            "promotions",
            promotions,
            end="\r",
        )
