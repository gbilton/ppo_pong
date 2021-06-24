import os
from itertools import count

from ppo_torch import Agent
from pong import *

def save_imgs(win, i):
    i = str(i).zfill(5)
    pygame.image.save(win, f'recordings/frame_{i}.jpeg')

def clear_recordings():
    for img in os.listdir('recordings'):
        os.remove(f'recordings/{img}')

if __name__ == '__main__':
    if os.path.exists('episode.mp4'):
        os.remove('episode.mp4')
    clear_recordings()

    env = make('Pong-v0')

    p = DQN()
    p.load_state_dict(torch.load('tmp/dqn/model100x3.pth'))
    p.eval()

    agent = Agent(n_actions=env.num_actions, input_dims=(env.state_size,))
    agent.actor.load_checkpoint()
    agent.actor.eval()

    observation = env.reset()
    for i in count():
        env.render()
        save_imgs(env.win, i)

        action2 = agent.actor.act(observation)

        inverted_observation = Tools.invert(observation)
        action1 = p.act(inverted_observation[:-1])
        actions = [action1, action2]

        observation, _, r, done = env.step(actions)

        if done:
            break

    os.system('ffmpeg -r 90 -f image2 -i recordings/frame_%05d.jpeg episode.mp4')
    clear_recordings()
