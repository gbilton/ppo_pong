import argparse
import glob
import json
import os
import random
import time
from collections import deque
from itertools import count

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from perfect_bot import PerfectDefender
from ppo_torch import Agent, load_policy
from pong import make

FRAME_SKIP = 4
SCORE_THRESHOLD = 0.4
SCORE_WINDOW = 300  # episodes averaged for the promotion gate (fresh policy)
MIN_EPISODES_BETWEEN_PROMOTIONS = 2000  # spacing keeps pool members distinct
TRANSITIONS_PER_UPDATE = 4096
CHECKPOINT_INTERVAL = 50  # updates between periodic checkpoints

# opponent pool: play the newest champion most, but never forget the anchor
# (the run's origin) or past champions - prevents self-play cycling.
# the scripted geometry bot gets a fixed share: it only concedes to
# objectively hard shots, so its reward signal is pure offense quality.
POOL_MAX = 20
P_BOT = 0.15
P_LATEST = 0.5  # of the non-bot share
P_ANCHOR = 0.25
BOT_IDX = -1

# fixed-reference evaluation: deterministic games vs an external yardstick,
# logged as eval/* - the honest strength curve that promotions can't fake
EVAL_INTERVAL = int(os.getenv("EVAL_INTERVAL", "200"))  # updates
EVAL_EPISODES = 32
EVAL_ENVS = 8


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
    parser.add_argument(
        "--eval-ref",
        default="tmp/docker/legacy_models/actor_goat",
        metavar="MODEL",
        help="fixed reference model for eval/* metrics (skip if missing)",
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


def save_checkpoint(run_dir, agent, latest_opponent, train_state):
    path = os.path.join(run_dir, "checkpoint.pt")
    checkpoint = {
        "agent": agent.get_checkpoint(),
        "opponent_actor": latest_opponent.state_dict(),
        "train_state": train_state,
    }
    torch.save(checkpoint, path + ".tmp")
    os.replace(path + ".tmp", path)


def load_pool(pool_dir, num_actions, state_size, device):
    """Anchor (pool_000) plus the most recent POOL_MAX-1 champions."""
    files = sorted(glob.glob(os.path.join(pool_dir, "pool_*.pt")))
    if len(files) > POOL_MAX:
        files = [files[0]] + files[-(POOL_MAX - 1) :]
    return [
        load_policy(f, num_actions, (state_size,), device=device) for f in files
    ]


def sample_opponent(pool_size):
    r = random.random()
    if r < P_BOT:
        return BOT_IDX
    r = (r - P_BOT) / (1 - P_BOT)
    if r < P_LATEST or pool_size == 1:
        return pool_size - 1
    if r < P_LATEST + P_ANCHOR or pool_size == 2:
        return 0
    return random.randrange(1, pool_size - 1)


def evaluate_vs_ref(agent, ref_actor, eval_envs, episodes=EVAL_EPISODES):
    """Deterministic (argmax) games vs the fixed reference. Returns metrics."""
    obs = np.array([e.reset() for e in eval_envs], dtype=np.float32)
    wins = losses = timeouts = 0
    decisions = 0
    while wins + losses + timeouts < episodes and decisions < 60_000:
        actions = agent.actor.act_batch(obs)
        bots = ref_actor.act_batch(invert_batch(obs))
        decisions += len(eval_envs)
        for k, env in enumerate(eval_envs):
            done = False
            for _ in range(FRAME_SKIP):
                o, r1, r2, done = env.step([bots[k], actions[k]])
                if done:
                    break
            if done:
                if r2 == 1 and r1 == -1:
                    wins += 1
                elif r1 == 1 and r2 == -1:
                    losses += 1
                else:
                    timeouts += 1
                obs[k] = env.get_state()
            else:
                obs[k] = o
    total = max(1, wins + losses + timeouts)
    return {
        "winrate": wins / total,
        "score": (wins - losses) / total,
        "timeout_frac": timeouts / total,
    }


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

    if checkpoint:
        agent.load_checkpoint_state(checkpoint["agent"])
    elif args.init_weights:
        actor_file = os.path.join(args.init_weights, f"actor_torch_ppo_{replica_id}")
        critic_file = os.path.join(args.init_weights, f"critic_torch_ppo_{replica_id}")
        agent.actor.load_state_dict(torch.load(actor_file, map_location=device))
        agent.critic.load_state_dict(torch.load(critic_file, map_location=device))
        print(f"initialized weights from {args.init_weights}")

    # ---- opponent pool ----
    pool_dir = os.path.join(run_dir, "pool")
    os.makedirs(pool_dir, exist_ok=True)
    anchor_path = os.path.join(pool_dir, "pool_000.pt")
    if not os.path.exists(anchor_path):
        # anchor = the run's origin (for old-format resumes: last opponent)
        seed_state = (
            checkpoint["opponent_actor"] if checkpoint else agent.actor.state_dict()
        )
        torch.save(seed_state, anchor_path)
    pool = load_pool(pool_dir, num_actions, state_size, device)
    print(f"opponent pool: {len(pool)} member(s)")

    eval_envs = [make("Pong-v0") for _ in range(EVAL_ENVS)]
    eval_bot = PerfectDefender()
    eval_ref = None
    if args.eval_ref and os.path.exists(args.eval_ref):
        eval_ref = load_policy(args.eval_ref, num_actions, (state_size,), device)
        print(f"eval reference: {args.eval_ref}")

    train_state = checkpoint["train_state"] if checkpoint else {}
    n_steps = train_state.get("n_steps", 0)
    start_update = train_state.get("update", 0)
    promotions = train_state.get("promotions", 0)
    episodes_done = train_state.get("episodes_done", 0)
    episodes_since_promotion = train_state.get("episodes_since_promotion", 0)
    score_history = deque(
        train_state.get("score_history", [-1.0] * SCORE_WINDOW), maxlen=SCORE_WINDOW
    )
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
    env_opp = np.array([sample_opponent(len(pool)) for _ in range(num_envs)])
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
            inverted = invert_batch(observations)
            bot_actions = np.full(num_envs, 2, dtype=np.int64)
            for idx in set(env_opp.tolist()):
                mask = env_opp == idx
                opponent = eval_bot if idx == BOT_IDX else pool[idx]
                bot_actions[mask] = opponent.act_batch(inverted[mask])

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
                    # bot episodes are a curriculum, not a fair gate: the
                    # promotion score only counts learned opponents
                    if env_opp[k] != BOT_IDX:
                        score_history.append(ep_scores[k])
                        episodes_since_promotion += 1
                    recent_lengths.append(ep_lengths[k])
                    episodes_done += 1
                    ep_scores[k] = 0.0
                    ep_lengths[k] = 0
                    # env auto-resets inside step(); fresh state, fresh opponent
                    observations[k] = env.get_state()
                    env_opp[k] = sample_opponent(len(pool))
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
        writer.add_scalar("selfplay/pool_size", len(pool), n_steps)
        writer.add_scalar("perf/steps_per_sec", steps_per_sec, n_steps)

        if (update + 1) % EVAL_INTERVAL == 0:
            if eval_ref is not None:
                results = evaluate_vs_ref(agent, eval_ref, eval_envs)
                for key, value in results.items():
                    writer.add_scalar(f"eval/vs_ref_{key}", value, n_steps)
                print(
                    f"\neval vs ref @ {n_steps:,}: winrate {results['winrate']:.2f} "
                    f"score {results['score']:+.2f} "
                    f"timeouts {results['timeout_frac']:.2f}"
                )
            bot_results = evaluate_vs_ref(agent, eval_bot, eval_envs)
            for key, value in bot_results.items():
                writer.add_scalar(f"eval/vs_bot_{key}", value, n_steps)
            print(
                f"eval vs bot @ {n_steps:,}: winrate {bot_results['winrate']:.2f} "
                f"score {bot_results['score']:+.2f} "
                f"timeouts {bot_results['timeout_frac']:.2f}"
            )

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
            print(f"\npromotion {promotions}: champion joins the pool")
            agent.save_models()  # champion export for play.py / tournament.py
            champ_path = os.path.join(pool_dir, f"pool_{promotions:03d}.pt")
            torch.save(agent.actor.state_dict(), champ_path)
            pool = load_pool(pool_dir, num_actions, state_size, device)
            score_history.extend([-1.0] * SCORE_WINDOW)
            episodes_since_promotion = 0

        if promoted or (update + 1) % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(run_dir, agent, pool[-1], snapshot_train_state(update))

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
