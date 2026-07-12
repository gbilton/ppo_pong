"""Record rallies where a model scores on the scripted perfect defender.

Runs bot (left) vs a checkpoint (right) headlessly, buffering the tail of
each rally; when the bot concedes, the buffer is written to an mp4.

    .venv/bin/python record_concessions.py --model tmp/docker/legacy_models/actor_goat_v2 --count 5
"""

import argparse
import os

import imageio.v2 as imageio
import numpy as np
import torch

from perfect_bot import PerfectDefender
from ppo_torch import load_policy
from pong import make, Tools, pygame

FRAME_SKIP = 4
SCALE = 0.5
CAPTURE_EVERY = 2  # buffer every 2nd frame -> 45 fps playback
BUFFER_FRAMES = 450  # ~10 seconds of tail
FREEZE_FRAMES = 40  # hold the final frame so the miss is visible


def draw_frame(env, surface):
    surface.fill((8, 16, 12))
    for p in (env.player1, env.player2):
        pygame.draw.rect(
            surface,
            (230, 240, 235),
            (p.x * SCALE, p.y * SCALE, p.width * SCALE, p.height * SCALE),
        )
    pygame.draw.circle(
        surface,
        (87, 242, 135),
        (env.ball.x * SCALE, env.ball.y * SCALE),
        max(2, int(env.ball.r * SCALE)),
    )
    return np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="tmp/docker/legacy_models/actor_goat_v2")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--max-rallies", type=int, default=600)
    parser.add_argument("--outdir", default="/tmp/concessions")
    parser.add_argument(
        "--bot-side", choices=["left", "right"], default="left",
        help="which paddle the scripted bot controls",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    env = make("Pong-v0")
    model = load_policy(
        args.model, env.num_actions, (env.state_size,), device=torch.device("cpu")
    )
    bot = PerfectDefender()
    surface = pygame.Surface((int(800 * SCALE), int(500 * SCALE)))

    observation = env.reset()
    buffer = []
    captured = rallies = frame = 0
    bot_action = model_action = 2

    while captured < args.count and rallies < args.max_rallies:
        if frame % FRAME_SKIP == 0:
            if args.bot_side == "left":
                bot_action = bot.act(observation)
                model_action = model.act(Tools.invert(observation))
            else:
                bot_action = bot.act(Tools.invert(observation))
                model_action = model.act(observation)
        if frame % CAPTURE_EVERY == 0:
            buffer.append(draw_frame(env, surface))
            if len(buffer) > BUFFER_FRAMES:
                buffer.pop(0)
        frame += 1
        if args.bot_side == "left":
            observation, r_model, r_bot, done = env.step([model_action, bot_action])
        else:
            observation, r_bot, r_model, done = env.step([bot_action, model_action])

        if done:
            rallies += 1
            if r_model == 1 and r_bot == -1:  # the bot got beaten
                buffer.append(draw_frame(env, surface))
                frames = buffer + [buffer[-1]] * FREEZE_FRAMES
                captured += 1
                path = os.path.join(args.outdir, f"concession_{captured}.mp4")
                imageio.mimwrite(path, frames, fps=45, quality=8)
                print(f"[{rallies} rallies] captured #{captured}: {path}", flush=True)
            buffer = []
            observation = env.get_state()

    print(f"done: {captured} concessions in {rallies} rallies")
