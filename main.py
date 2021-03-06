import numpy as np
import random
import sys
import random
from itertools import count

from ppo_torch import Agent
from pong import *


if __name__ == '__main__':
    env = make('Pong-v0')

    batch_size = 1000
    N = 4000

    agent = Agent(n_actions=env.num_actions,input_dims=(env.state_size,), batch_size=batch_size)

    agent.load_models()

    agent1 = Agent(n_actions=env.num_actions,input_dims=(env.state_size,))
    agent1.actor.load_state_dict(agent.actor.state_dict())
    agent1.actor.eval()

    # p = DQN()
    # p.load_state_dict(torch.load('tmp/dqn/model100x3.pth'))
    # p.eval()

    best_score = -1
    score_history = [-1 for i in range(100)]

    learn_iters = 0
    avg_score = 0
    n_steps = 0
    j = 0

    for i in count():
        j+=1
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            inverted_observation = Tools.invert(observation)
            bot_action = agent1.actor.act(inverted_observation)
            # bot_action = p.act(inverted_observation[:-1])
            for _ in range(4):
                observation_, _, reward, done = env.step([bot_action, action])
                if done:
                    break
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                print(random.choice(list(range(10))), end = '\r')
                agent.learn()
                learn_iters += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        # if avg_score > best_score and j >= 100:
            # best_score = avg_score
            # agent.save_models()

        if avg_score >= 0.4 and j >= 100:
            print('!!!!!!!!!!UPDATED!!!!!!!!!!')
            agent.save_models()
            score_history = [-1 for _ in range(100)]
            agent1.actor.load_state_dict(agent.actor.state_dict())
            agent1.actor.eval()
            best_score = -1
            j = 0


        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', n_steps, 'learning_steps', learn_iters, end='\r')
