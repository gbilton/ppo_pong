import json
from ppo_torch import Agent
from pong import *
import itertools
from multiprocessing import Pool


def simulate_match(match):
    selected_agent_left, selected_agent_right = match
    agent_left = selected_agent_left["agent"]
    agent_left_name = selected_agent_left["agent_name"]
    agent_right = selected_agent_right["agent"]
    agent_right_name = selected_agent_right["agent_name"]

    match_results = {}

    score = np.array([0, 0])
    observation = env.reset()
    while True:
        action2 = agent_left.actor.act(observation)

        inverted_observation = Tools.invert(observation)
        action1 = agent_right.actor.act(inverted_observation)

        actions = [action1, action2]
        observation, r1, r2, done = env.step(actions)

        if done:
            print(score, end="\r")
            if r1 == -1 and r2 == 1:
                score += np.array([1, 0])
            if r1 == 1 and r2 == -1:
                score += np.array([0, 1])

        if np.max(score) >= 100:
            match_results["match"] = f"{agent_left_name} X {agent_right_name}"
            match_results["score"] = score.tolist()
            if score[0] > score[1]:
                match_results["winner"] = agent_left_name
                match_results["loser"] = agent_right_name
            else:
                match_results["winner"] = agent_right_name
                match_results["loser"] = agent_left_name
            return match_results


if __name__ == "__main__":
    env = make("Pong-v0")
    device = torch.device("cpu")

    names = [
        "actor",
        "actor_goat",
        "actor copy",
        "actor_torch_ppo",
        "actor_goat copy",
        "actor_goat copy 2",
        "actor_goat copy 3",
        "actor_goat copy 4",
        "actor_goat copy 5",
        "actor_torch_ppo_1",
    ]
    agents = []
    for agent_name in names:
        agent = Agent(
            n_actions=env.num_actions, input_dims=(env.state_size,), device=device
        )
        agent.actor.checkpoint_file = os.path.join(
            "./tmp/docker/legacy_models", agent_name
        )
        agent.actor.load_checkpoint()
        agent.actor.eval()
        agents.append({"agent_name": agent_name, "agent": agent})

    match_combinations = list(itertools.combinations(agents, 2))
    results = []

    # Parallelize match simulations
    with Pool(processes=10) as pool:  # Set the number of processes you want to use
        results = pool.map(simulate_match, match_combinations)

    with open("tournament_results.json", "w") as f:
        json.dump(results, f)
