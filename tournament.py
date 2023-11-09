import json
from ppo_torch import Agent
from pong import *
import itertools


if __name__ == "__main__":
    env = make("Pong-v0")
    device = torch.device("cpu")

    agents = [
        "actor_goat",
        "actor copy",
        "actor_torch_ppo",
        "actor_goat copy",
        "actor_goat copy 2",
        "actor_torch_ppo_1",
    ]
    for i in agents:
        agent_name = f"actor_torch_ppo_{i}"
        agent = Agent(
            n_actions=env.num_actions, input_dims=(env.state_size,), device=device
        )
        agent.actor.checkpoint_file = os.path.join("./tmp/docker/models", agent_name)
        agent.actor.load_checkpoint()
        agent.actor.eval()
        agents.append({"agent_name": agent_name, "agent": agent})

    match_combinations = list(itertools.combinations(agents, 2))

    score = np.array([0, 0])
    observation = env.reset()

    results = []
    for match in match_combinations:
        selected_agent_left = match[0]
        selected_agent_right = match[1]

        agent_left = selected_agent_left["agent"]
        agent_left_name = selected_agent_left["agent_name"]

        agent_right = selected_agent_right["agent"]
        agent_right_name = selected_agent_right["agent_name"]
        match_results = {}  # Simulate matches
        while True:
            # env.render()
            # keys = pygame.key.get_pressed()
            # env.player1.key_movement(keys)
            # env.player2.key_movement(keys)

            action2 = agent_left.actor.act(observation)

            inverted_observation = Tools.invert(observation)
            # action1 = p.act(inverted_observation)
            action1 = agent_right.actor.act(inverted_observation)
            # action1 = 2

            actions = [action1, action2]
            observation, r1, r2, done = env.step(actions)

            if done:
                if r1 == -1 and r2 == 1:
                    score += np.array([1, 0])
                if r1 == 1 and r2 == -1:
                    score += np.array([0, 1])
                print(f"score = {score}", end="\r")

            if np.max(score) >= 2:
                print(f"Final Score: {score}")
                match_results["match"] = f"{agent_left_name} X {agent_right_name}"
                match_results["score"] = score.tolist()
                if score[0] > score[1]:
                    match_results["winner"] = agent_left_name
                    match_results["loser"] = agent_right_name
                    print("Winner: ", agent_left_name)
                    print("Loser: ", agent_right_name)
                else:
                    match_results["winner"] = agent_right_name
                    match_results["loser"] = agent_left_name
                    print("Winner: ", agent_right_name)
                    print("Loser: ", agent_left_name)
                score = np.array([0, 0])
                break
        results.append(match_results)

    with open("tournament_results.json", "w") as f:
        json.dump(results, f)
