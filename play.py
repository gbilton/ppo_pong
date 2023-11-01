import os

from ppo_torch import Agent
from pong import *

if __name__ == "__main__":
    env = make("Pong-v0")

    device = torch.device("cpu")

    p = DQN(device=device)
    p.load_state_dict(torch.load("tmp/dqn/model100x3.pth", map_location=device))
    p.eval()

    # agent_name = "actor_torch_ppo_1"
    agent_name = "actor_goat"
    agent = Agent(
        n_actions=env.num_actions, input_dims=(env.state_size,), device=device
    )
    agent.actor.checkpoint_file = os.path.join("./tmp/docker/legacy_models", agent_name)
    agent.actor.load_checkpoint()
    agent.actor.eval()

    bot_name = "actor_torch_ppo_4"
    bot = Agent(n_actions=env.num_actions, input_dims=(env.state_size,), device=device)
    bot.actor.checkpoint_file = os.path.join("./tmp/docker/legacy_models", bot_name)
    bot.actor.load_checkpoint()
    bot.actor.eval()

    score = np.array([0, 0])
    observation = env.reset()
    while True:
        # env.render()
        # keys = pygame.key.get_pressed()
        # env.player1.key_movement(keys)
        # env.player2.key_movement(keys)
        action2 = agent.actor.act(observation)

        inverted_observation = Tools.invert(observation)
        # action1 = p.act(inverted_observation)
        action1 = bot.actor.act(inverted_observation)
        # action1 = 2

        actions = [action1, action2]
        observation, r1, r2, done = env.step(actions)

        if done:
            if r1 == -1 and r2 == 1:
                score += np.array([1, 0])
            if r1 == 1 and r2 == -1:
                score += np.array([0, 1])
            print(f"score = {score}", end="\r")
