from ppo_torch import Agent

from pong import *


if __name__ == '__main__':
    env = make('Pong-v0')

    device = torch.device('cpu')

    p = DQN(device=device)
    p.load_state_dict(torch.load('tmp/dqn/model100x3.pth', map_location=device))
    p.eval()

    agent = Agent(n_actions=env.num_actions, input_dims=(env.state_size,), device=device)
    agent.actor.load_checkpoint()
    agent.actor.eval()

    observation = env.reset()
    score = np.array([0,0])
    timestep = 0
    while True:
        env.render()
        # keys=pygame.key.get_pressed()
        # env.player1.key_movement(keys)
        # env.player2.key_movement(keys)
        action2 = agent.actor.act(observation)
        inverted_observation = Tools.invert(observation)
        action1 = p.act(inverted_observation)
        # action1 = 2
        # action1 = agent.actor.act(inverted_observation)
        # action2, _, _ = agent.choose_action(observation)
        actions = [action1, action2]
        # for i in range(4):
        observation, _, r, done = env.step(actions)
        timestep+=1
        if timestep >= 10000:
            observation = env.reset()
            timestep = 0
        if done:
            if r == 1:
                score += np.array([r,0])
            if r == -1:
                score += np.array([0,1])
            print(f'score = {score}', end='\r')
