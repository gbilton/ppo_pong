import json
import os
import time
from itertools import count

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from ppo_torch import Agent
from pong import make, Tools


if __name__ == "__main__":
    env = make("Pong-v0")

    batch_size = 256
    N = 4000

    agent = Agent(
        n_actions=env.num_actions, input_dims=(env.state_size,), batch_size=batch_size
    )

    # frozen opponent: starts as a copy of the untrained agent, replaced on promotion
    agent1 = Agent(n_actions=env.num_actions, input_dims=(env.state_size,))
    agent1.actor.load_state_dict(agent.actor.state_dict())
    agent1.actor.eval()

    SCORE_THRESHOLD = 0.4
    score_history = [-1 for _ in range(100)]
    learn_iters = 0
    n_steps = 0
    promotions = 0
    j = 0

    replica_id = int(os.getenv("REPLICA_ID", "0"))

    json_save_directory = "tmp/docker/json_files/"
    os.makedirs(json_save_directory, exist_ok=True)
    json_file_path = f"{json_save_directory}avg_score_replica{replica_id}.json"

    run_name = time.strftime(f"ppo_pong_replica{replica_id}_%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=os.path.join("runs", run_name))

    avg_scores = []
    episodes_time = []
    for i in count():
        start_time = time.time()
        j += 1
        observation = env.reset()
        done = False
        score = 0
        episode_steps = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            inverted_observation = Tools.invert(observation)
            bot_action = agent1.actor.act(inverted_observation)
            for _ in range(4):
                observation_, _, reward, done = env.step([bot_action, action])
                if done:
                    break
            n_steps += 1
            episode_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                metrics = agent.learn()
                learn_iters += 1
                for key, value in metrics.items():
                    writer.add_scalar(f"train/{key}", value, n_steps)
            observation = observation_

        episode_time = time.time() - start_time
        episodes_time.append(episode_time)
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        avg_scores.append(avg_score)

        writer.add_scalar("episode/score", score, i)
        writer.add_scalar("episode/avg_score_100", avg_score, i)
        writer.add_scalar("episode/length", episode_steps, i)
        writer.add_scalar("selfplay/promotions", promotions, i)

        if (i + 1) % 100 == 0:
            avg_score_data = {
                "replica_id": replica_id,
                "avg_scores": avg_scores,
                "episodes_time": episodes_time,
            }
            with open(json_file_path, "w") as json_file:
                json.dump(avg_score_data, json_file)

        if avg_score >= SCORE_THRESHOLD and j >= 100:
            promotions += 1
            print(f"\npromotion {promotions}: saving models, updating opponent")
            agent.save_models()
            agent1.actor.load_state_dict(agent.actor.state_dict())
            agent1.actor.eval()
            score_history = [-1 for _ in range(100)]
            j = 0

        print(
            "episode",
            i,
            "score %.1f" % score,
            "avg score %.1f" % avg_score,
            "time_steps",
            n_steps,
            "learning_steps",
            learn_iters,
            end="\r",
        )
