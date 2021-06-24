import os

from ppo_torch import Agent
from pong import *

def save_imgs(win, i):
    i = str(i).zfill(5)
    pygame.image.save(win, f'recordings/frame_{i}.png')

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
    i=0
    record = True
    while True:
        env.render()
        i+=1

        if record:
            save_imgs(env.win, i)
        # with open(f'recordings/frame_{i}', 'wb') as f:
            # f.write(frame)

        # keys=pygame.key.get_pressed()
        # env.player1.key_movement(keys)
        # env.player2.key_movement(keys)
        action2 = agent.actor.act(observation)
        inverted_observation = Tools.invert(observation)
        action1 = p.act(inverted_observation)
        # action1 = 2
        # action2, _, _ = agent.choose_action(observation)
        actions = [action1, action2]
        # for i in range(4):
        observation, _, r, done = env.step(actions)

        if done:
            break
            if r == 1:
                score += np.array([r,0])
            if r == -1:
                score += np.array([0,1])
            print(f'score = {score}', end='\r')

    if record:
        os.system('ffmpeg -r 60 -f image2 -s 1920x1080 -i recordings/frame_%05d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p test.mp4')
        for img in os.listdir('recordings'):
            os.remove(f'recordings/{img}')
