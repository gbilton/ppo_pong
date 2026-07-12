import argparse

import numpy as np
import torch

from ppo_torch import load_policy
from pong import make, Tools, pygame

FRAME_SKIP = 4  # match training: the agent picks an action every 4th frame


def parse_args():
    parser = argparse.ArgumentParser(
        description="Play Pong against a trained agent (arrow keys, right paddle)"
    )
    parser.add_argument(
        "--model",
        default="tmp/docker/models/actor_torch_ppo_0",
        help="agent checkpoint: actor file, run checkpoint.pt, or model dir "
        "(default: current champion)",
    )
    parser.add_argument(
        "--opponent",
        default=None,
        metavar="MODEL",
        help="instead of playing yourself, watch MODEL play on the right side",
    )
    parser.add_argument(
        "--points", type=int, default=None, help="end the game at this many points"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cpu")
    env = make("Pong-v0")

    agent_actor = load_policy(
        args.model, env.num_actions, (env.state_size,), device=device
    )
    opponent_actor = None
    if args.opponent:
        opponent_actor = load_policy(
            args.opponent, env.num_actions, (env.state_size,), device=device
        )

    score = np.array([0, 0])  # [human/right, agent/left]
    observation = env.reset()
    agent_action = 2
    opponent_action = 2
    frame = 0

    while True:
        env.render()
        if opponent_actor is None:
            keys = pygame.key.get_pressed()
            env.player1.key_movement(keys)

        if frame % FRAME_SKIP == 0:
            agent_action = agent_actor.act(observation)
            if opponent_actor is not None:
                opponent_action = opponent_actor.act(Tools.invert(observation))
        frame += 1

        observation, r1, r2, done = env.step([opponent_action, agent_action])

        if done:
            if r1 == 1 and r2 == -1:
                score += np.array([1, 0])
            elif r1 == -1 and r2 == 1:
                score += np.array([0, 1])
            print(f"you {score[0]} x {score[1]} agent", end="\r")
            if args.points and score.max() >= args.points:
                winner = "you win!" if score[0] > score[1] else "agent wins"
                print(f"\nfinal: you {score[0]} x {score[1]} agent - {winner}")
                break
