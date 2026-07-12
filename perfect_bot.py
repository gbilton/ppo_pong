"""Scripted 'perfect defender' — a check on the interception math.

Claim under test: with paddle speed 5, ball speed <= 10, and a 680 px
crossing, a defender that predicts the landing point and re-centers
between exchanges can never concede. Run it against any checkpoint:

    .venv/bin/python perfect_bot.py --model tmp/docker/legacy_models/actor_goat_v2
"""

import argparse

import numpy as np
import torch

from ppo_torch import load_policy
from pong import make, Tools

FRAME_SKIP = 4

SCREEN_W = 800.0
SCREEN_H = 500.0
BALL_R = 6.0
PADDLE_H = 56.0
FACE_X = 66.0  # left paddle face (x=60) plus ball radius
CENTER_Y = SCREEN_H / 2
DEADBAND = 2.0
# the env catches only if the ball's BOTTOM edge is inside the paddle span,
# so the effective window for the ball's center is shifted up by the radius:
# [paddle_y - r, paddle_y + h - r], centered at paddle_y + (h - r) / 2
AIM_OFFSET = (PADDLE_H - BALL_R) / 2


class PerfectDefender:
    """Plays the left seat (same frame trained agents use; invert for right)."""

    def act(self, state):
        own_y = state[1] * SCREEN_H  # paddle top edge
        own_center = own_y + AIM_OFFSET
        bx = state[2] * SCREEN_W
        by = state[3] * SCREEN_H
        vx = state[4] * 10.0
        vy = state[5] * 10.0

        if vx < 0 and bx > FACE_X:  # incoming: intercept the landing point
            t = (bx - FACE_X) / -vx
            target = self._fold(by + vy * t)
        else:  # outgoing or degenerate: re-center
            target = CENTER_Y

        if own_center > target + DEADBAND:
            return 0  # up
        if own_center < target - DEADBAND:
            return 1  # down
        return 2  # stay

    def act_batch(self, states):
        return np.array([self.act(s) for s in states], dtype=np.int64)

    @staticmethod
    def _fold(y):
        """Reflect a straight-line y into the court via wall bounces."""
        lo, hi = BALL_R, SCREEN_H - BALL_R
        span = hi - lo
        z = (y - lo) % (2 * span)
        return lo + (z if z <= span else 2 * span - z)


def run_side(env, left, right, episodes, left_skip=FRAME_SKIP, right_skip=FRAME_SKIP):
    """left/right are objects with .act(); right sees inverted state."""
    wins = losses = timeouts = 0  # from the LEFT player's perspective
    observation = env.reset()
    left_action = right_action = 2
    frame = 0
    while wins + losses + timeouts < episodes:
        if frame % left_skip == 0:
            left_action = left.act(observation)
        if frame % right_skip == 0:
            right_action = right.act(Tools.invert(observation))
        frame += 1
        observation, r_right, r_left, done = env.step([right_action, left_action])
        if done:
            if r_left == 1 and r_right == -1:
                wins += 1
            elif r_right == 1 and r_left == -1:
                losses += 1
            else:
                timeouts += 1
    return wins, losses, timeouts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perfect-defender check")
    parser.add_argument("--model", default="tmp/docker/legacy_models/actor_goat_v2")
    parser.add_argument("--episodes", type=int, default=30, help="rallies per side")
    parser.add_argument(
        "--bot-skip",
        type=int,
        default=FRAME_SKIP,
        help="frames between bot decisions (models always use 4)",
    )
    args = parser.parse_args()

    env = make("Pong-v0")
    model = load_policy(
        args.model, env.num_actions, (env.state_size,), device=torch.device("cpu")
    )
    bot = PerfectDefender()

    print(
        f"perfect defender (skip={args.bot_skip}) vs {args.model}, "
        f"{args.episodes} rallies per side"
    )

    w, l, t = run_side(env, bot, model, args.episodes, left_skip=args.bot_skip)
    print(f"bot as LEFT :  bot {w}  model {l}  timeouts {t}", flush=True)
    conceded = l

    w2, l2, t2 = run_side(env, model, bot, args.episodes, right_skip=args.bot_skip)
    print(f"bot as RIGHT:  bot {l2}  model {w2}  timeouts {t2}", flush=True)
    conceded += w2

    total = 2 * args.episodes
    print(
        f"\nverdict: bot conceded {conceded}/{total} rallies "
        f"({'PERFECT DEFENSE CONFIRMED' if conceded == 0 else 'analysis wrong somewhere'})"
    )
