import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.distributions.categorical import Categorical
import math
import numpy as np
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
from collections import namedtuple
import random
import matplotlib.pyplot as plt
from matplotlib import style
from itertools import count
import time
import sys


class Player:
    def __init__(self, x, y, height, width, vel, up_key, down_key, num_actions, state_size, screen_height):
        self.x = x
        self.y = y
        self.height = height
        self.width = width
        self.vel = vel
        self.up_key = up_key
        self.down_key = down_key
        self.num_actions = num_actions
        self.screen_height = screen_height
    def draw(self, win):
        pygame.draw.rect(win, (255, 255, 255), (self.x, self.y, self.width, self.height))
    def key_movement(self, keys):
        if keys[getattr(pygame, self.up_key)]:
            self.moveup()
        if keys[getattr(pygame, self.down_key)]:
            self.movedown()
    def movement(self, action):
        if action == 0:
            self.moveup()
        if action == 1:
            self.movedown()
    def moveup(self):
        if self.y <= 0:
            self.y -= 0
        else:
            self.y -= self.vel
    def movedown(self):
        if self.y + self.height >= self.screen_height:
            self.y += 0
        else:
            self.y += self.vel
    def select_action(self, state):
        rate = self.strategy.get_exploration_rate(self.current_step)
        # if self.memory.can_provide_sample(self.memory.capacity):
        self.current_step += 1
        if rate > random.random():
            return torch.tensor(random.randrange(self.num_actions)).to(self.device) # explore
        else:
            with torch.no_grad():
                return self.policy_net(state).argmax(dim=1) # exploit
    def get_current(self, states, actions):
        return self.policy_net(states).gather(dim=1, index=actions)
    def get_next(self, next_states):
        return self.target_net(next_states).max(dim=1, keepdim=True).values
    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    def follow(self):
        if ball.y > self.y + self.height/2:
            return 1
        if ball.y < self.y + self.height/2:
            return 0
    def hardcodedai(self):
        return self.follow()
class Ball:
    def __init__ (self, x, y, r, velx, vely):
        self.x = x
        self.y = y
        self.r = r
        self.velx = velx
        self.vely = vely

    def draw(self, win):
        pygame.draw.circle(win, (0, 0, 0), (self.x, self.y), self.r)

    def move(self):
        self.x += self.velx
        self.y += self.vely

    def bounce_x_increase(self, maxvel):
        if abs(self.velx) > maxvel:
            if self.velx > 0:
                self.velx = (self.velx) * -1
            else:
                self.velx = (self.velx) * -1
        else:
            if self.velx > 0:
                self.velx = (self.velx + 1) * -1
            else:
                self.velx = (self.velx - 1) * -1
        return self.velx

    def bounce(self, player1, player2, screen_height, maxvel):
        bounce, x, _, _ = self.collision(player1, player2)
        if bounce:
            if x == 1:
                self.bounce_x_increase(maxvel)
                self.vely = self.vely - 5
            if x == 2:
                self.bounce_x_increase(maxvel)
                self.vely = self.vely - 1
            if x == 3:
                self.bounce_x_increase(maxvel)
            if x == 4:
                self.bounce_x_increase(maxvel)
                self.vely = self.vely + 1
            if x == 5:
                self.bounce_x_increase(maxvel)
                self.vely = self.vely + 5

        if self.groundcollision(screen_height):
            self.vely = (self.vely) * -1

    def collision(self, player1, player2):
        collision = False
        P1 = False
        P2 = False
        x = 0
        reward1 = 0
        reward2 = 0
        if self.x + self.r >= player1.x and self.x - self.r <= player1.x + player1.width:
            if self.y + self.r >= player1.y and self.y + self.r <= player1.y + player1.height:
                collision = True
                P1 = True

        if self.x - self.r <= player2.x + player2.width and self.x + self.r >= player2.x:
            if self.y + self.r >= player2.y and self.y + self.r <= player2.y + player2.height:
                collision = True
                P2 = True


        if collision & P1:
            self.x = player1.x - self.r
            if self.y + self.r > player1.y and self.y <= math.floor(player1.y + player1.height/8):
                x = 1
            if self.y > math.floor(player1.y + player1.height/8) and self.y <=  math.floor(player1.y + 3*player1.height/8):
                x = 2
            if self.y > math.floor(player1.y + 3*player1.height/8) and self.y <= math.floor(player1.y + 5*player1.height/8):
                x = 3
            if self.y > math.floor(player1.y + 5*player1.height/8) and self.y <= math.floor(player1.y + 7*player1.height/8):
                x = 4
            if self.y > math.floor(player1.y + 7*player1.height/8) and self.y - self.r <= math.floor(player1.y + player1.height):
                x = 5
            reward1 = 0
        if collision & P2:
            self.x = player2.x + player2.width + self.r
            if self.y + self.r > player2.y and self.y <= math.floor(player2.y + player2.height/8):
                x = 1
            if self.y > math.floor(player2.y + player2.height/8) and self.y <=  math.floor(player2.y + 3*player2.height/8):
                x = 2
            if self.y > math.floor(player2.y + 3*player2.height/8) and self.y <= math.floor(player2.y + 5*player2.height/8):
                x = 3
            if self.y > math.floor(player2.y + 5*player2.height/8) and self.y <= math.floor(player2.y + 7*player2.height/8):
                x = 4
            if self.y > math.floor(player2.y + 7*player2.height/8) and self.y - self.r <= math.floor(player2.y + player2.height):
                x = 5
            reward2 = 0
        return collision, x, reward1, reward2

    def groundcollision(self, screen_height):
        groundcollision = False
        if self.y + self.r >= screen_height:
            groundcollision = True
        if self.y - self.r <= 0:
            groundcollision = True
        return groundcollision
class Pong_env():
    screen_width = 800
    def __init__(self):
        # self.screen_width = 800
        self.screen_height = 500
        self.run = True
        self.maxvel = 10
        self.player_speed = 5
        self.height = 56
        self.width = 10
        self.player1_x = self.screen_width - 50 - self.width
        self.player1_y = self.screen_height/2 - self.height/2
        self.player2_x = 50
        self.player2_y = self.screen_height/2 - self.height/2
        self.r = [-3, -2, -1, 1, 2, 3]
        self.timestep = 0
        self.max_timestep = 3000
        # self.r = [-3, -2, -1]
        self.ball_speed_x = 2 * random.choice(self.r)
        self.ball_speed_y = 2 * random.choice(self.r)
        self.num_actions = 3
        self.state_size = 7
        self.win = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.player2 = Player(self.player2_x, self.player2_y, self.height, self.width, self.player_speed, "K_w", "K_s", self.num_actions, self.state_size, self.screen_height)
        self.player1 = Player(self.player1_x, self.player1_y, self.height, self.width, self.player_speed, "K_UP", "K_DOWN", self.num_actions, self.state_size, self.screen_height)
        self.ball = Ball(math.floor(self.screen_width/2), math.floor(self.screen_width/2), 6, self.ball_speed_x, self.ball_speed_y)
        self.clock = pygame.time.Clock()

    def step(self, actions):
        self.timestep += 1
        self.player1.movement(actions[0])
        self.player2.movement(actions[1])
        self.ball.bounce(self.player1, self.player2, self.screen_height, self.maxvel)
        self.ball.move()
        state = self.get_state()
        _, _, _, reward_collision2 = self.ball.collision(self.player1, self.player2)
        done, reward1, reward2 = self.score(self.player1_x, self.player1_y, self.player2_x, self.player2_y)
        return state, reward1, reward2, done

    def reset(self):
        self.timestep = 0
        self.ball.x = math.floor(self.screen_width/2)
        self.ball.y = math.floor(self.screen_height/2)
        self.ball.velx = 2 * random.choice(self.r)
        self.ball.vely = random.choice(self.r)
        self.player1.x = self.player1_x
        self.player1.y = self.player1_y
        self.player2.x = self.player2_x
        self.player2.y = self.player2_y
        state = self.get_state()

        return state

    def score(self, player1_x, player1_y, player2_x, player2_y):
        point = False
        reward1 = 0
        reward2 = 0

        if self.ball.x + self.ball.r >= self.screen_width:
            reward1 = -1
            reward2 = 1
            point = True

        if self.ball.x - self.ball.r <= 0:
            reward1 = 1
            reward2 = -1
            point = True

        if point:
            self.reset()

        if self.timestep >= self.max_timestep:
            done = True
            reward1, reward2 = -1, -1
            self.reset()
            return done, reward1, reward2

        return point, reward1, reward2

    def get_state(self):
        state = np.array([self.player1.y/500, self.player2.y/500, self.ball.x/800, self.ball.y/500, self.ball.velx/10, self.ball.vely/10, self.timestep/self.max_timestep])
        return state

    def render(self):
        for event in pygame.event.get():
	        if event.type == pygame.QUIT:
	            pygame.quit()
	            sys.exit()
        self.win.fill((0,100,100))
        self.player1.draw(self.win)
        self.player2.draw(self.win)
        self.ball.draw(self.win)
        pygame.display.update()
        self.clock.tick(90)
class Tools():
    def np_to_torch(state):
        state = torch.tensor(state, dtype = torch.float32)
        state = torch.reshape(state, (1,len(state)))
        state = state.to(device)
        return state
    def extract_tensors(experiences):

        batch = Experience(*zip(*experiences))

        t1 = torch.cat(batch.state)
        t2 = torch.cat(batch.action2)
        t3 = torch.cat(batch.reward)
        t4 = torch.cat(batch.next_state)
        t5 = batch.done
        return (t1, t2, t3, t4, t5)
    def preprocess(action):
        action = torch.tensor(action)
        action = torch.reshape(action, (1, 1))
        return action
    def invert(state):
        p1_y = state[0]
        p2_y = state[1]
        bx = state[2]
        by = state[3]
        bvx = state[4]
        bvy = state[5]
        timestep = state[6]
        return np.array([p2_y, p1_y, getattr(Pong_env, 'screen_width')/800 - bx, by, -bvx, bvy, timestep])
class DQN(nn.Module):
    def __init__(self, input_size = 6, output_size = 3,
            device=torch.device('cpu')):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.out = nn.Linear(100, output_size)

        self.device = device
        self.to(self.device)

    def forward(self, t):
        t = torch.tanh(self.fc1(t))
        t = torch.tanh(self.fc2(t))
        t = torch.tanh(self.fc3(t))
        t = self.out(t)
        return t

    def act(self, observation):
        observation = observation[:-1]
        observation = torch.tensor(observation, dtype=torch.float).to(self.device)
        action = self.forward(observation)
        return action.argmax().item()

def createaxes():
    fig = plt.figure(1)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(224)
    return fig, ax1, ax2, ax3
def plotaxes(fig, ax1, ax2, ax3, values, limit, losses, times, avgscore):

    ax1.cla()
    ax2.cla()
    ax3.cla()

    ax1.set_title('Pong')
    ax2.set_title('Loss')
    ax3.set_title('Avg. Episode duration')

    ax1.set_xlabel('Games')
    ax2.set_xlabel('Episodes')
    ax3.set_xlabel('Episodes (x500)')

    ax3.set_ylabel('Time (s)')

    ax1.set_ylim(-21, 21)
    ax1.plot(values, 'b')
    ax1.plot(limit, 'g--')
    ax1.plot(avgscore, color='#f59542', linewidth = 2.0)
    ax2.plot(losses, 'r')
    ax3.plot(times, 'y')

    plt.tight_layout()
    plt.show()
def plot(values, valor):
    plt.figure(1)
    plt.clf()
    plt.title('Pong')
    plt.xlabel('Game')
    plt.ylabel('Score')
    plt.plot(np.arange(0, len(values)), values)
    plt.plot(np.arange(0, len(values)), (valor*np.ones(len(values))), 'r', '--')
def plottime(values):
    plt.figure(4)
    plt.clf()
    plt.title('Average time per episode in seconds')
    plt.ylabel('time(s)')
    plt.plot(values)
def plotscores(values):
    plt.figure(3)
    plt.clf()
    plt.title('Pong')
    plt.xlabel('Game')
    plt.ylabel('Score')
    plt.plot(values)
def plotloss(values):
    plt.figure(2)
    plt.clf()
    plt.title('Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.plot(values)

    plt.pause(0.0001)
def load_model(model, current_step = 0):
    checkpoint = torch.load(model+'-checkpoint.tar')
    player2.policy_net.load_state_dict(checkpoint['policy_net_p2'])
    player2.optimizer.load_state_dict(checkpoint['optimizer_p2'])
    player2.memory.memory = checkpoint['memory2']
    player2.current_step = current_step
    player2.policy_net.train()
    player2.update_target_net()
    point = torch.load(model + '.tar')
    player1.policy_net.load_state_dict(checkpoint['policy_net_p2'])
    player1.policy_net.eval()
    print("Model loaded successfully")
def save_model(model, player, checkpoint = False):
    if checkpoint:
        torch.save({
        'policy_net_p2': player2.policy_net.state_dict(),
        'optimizer_p2': player2.optimizer.state_dict(),
        'memory2': player2.memory.memory
        }, model+'-checkpoint.tar')
    else:
        torch.save({
        'policy_net_p2': player2.policy_net.state_dict(),
        'optimizer_p2': player2.optimizer.state_dict(),
        }, model+'.tar')
def save_placar(placar, avgscore, points):
    valor = placar[0] - placar[1]
    points = np.append(points, valor)
    np.save('points.npy', points)
    avgscore = np.append(avgscore, np.sum(points[-10:])/10)
    np.save('avgscore.npy', avgscore)
    placar = np.array([0,0], dtype = int)
    return valor, points, avgscore, placar
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def make(env_name):
    if env_name == 'Pong-v0':
        env = Pong_env()
        return env
