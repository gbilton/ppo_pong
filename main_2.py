import json
import numpy as np
import random
from itertools import count

from ppo_torch import Agent
from pong import *


if __name__ == "__main__":
    env = make("Pong-v0")

    batch_size = int(os.getenv("BATCH_SIZE", "0"))
    N = int(os.getenv("N", "0"))

    agent = Agent(
        n_actions=env.num_actions, input_dims=(env.state_size,), batch_size=batch_size
    )
    agent.alpha = float(os.getenv("ALPHA", "0"))

    agent.load_models()

    bots = []
    for i in range(1, 5):
        bot_name = f"actor_torch_ppo_{i}"
        bot = Agent(n_actions=env.num_actions, input_dims=(env.state_size,))
        bot.actor.checkpoint_file = os.path.join("./tmp/docker/models", bot_name)
        bot.actor.load_checkpoint()
        bot.actor.eval()
        bots.append(bot)

    score_history = [-1 for _ in range(100)]
    score_threshold = 0.1
    learn_iters = 0
    avg_score = 0
    n_steps = 0
    j = 0

    # Directory where you want to save the JSON files
    json_save_directory = "/app/tmp/docker/json_files/"

    # Initialize replica ID
    replica_id = int(os.getenv("REPLICA_ID", "0"))

    # File path for the JSON data
    json_file_path = f"{json_save_directory}avg_score_replica{replica_id}.json"

    avg_scores = []
    episodes_time = []
    for i in count():
        start_time = time.time()
        bot = random.choice(bots)
        j += 1
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            inverted_observation = Tools.invert(observation)
            bot_action = bot.actor.act(inverted_observation)
            for _ in range(4):
                observation_, _, reward, done = env.step([bot_action, action])
                if done:
                    break
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                print(random.choice(list(range(10))), end="\r")
                agent.learn()
                learn_iters += 1
            observation = observation_
        episode_time = time.time() - start_time
        episodes_time.append(episode_time)
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        avg_scores.append(avg_score)

        if (i + 1) % 100 == 0:
            # Create a dictionary with replica ID and avg_score
            avg_score_data = {
                "replica_id": replica_id,
                "avg_scores": avg_scores,
                "episodes_time": episodes_time,
            }

            # Save the dictionary as JSON
            with open(json_file_path, "w") as json_file:
                json.dump(avg_score_data, json_file)

        if avg_score >= score_threshold and j >= 100:
            print("!!!!!!!!!!UPDATED!!!!!!!!!!")
            score_threshold += 0.1
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
