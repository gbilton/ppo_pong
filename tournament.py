import argparse
import itertools
import json
import os

import torch

from perfect_bot import PerfectDefender
from ppo_torch import load_policy
from pong import make, Tools

FRAME_SKIP = 4  # match training: agents pick an action every 4th frame
MAX_DECISIONS_PER_GAME = 50_000  # safety net against endless stalemates


def parse_args():
    parser = argparse.ArgumentParser(
        description="Round-robin tournament between saved checkpoints"
    )
    parser.add_argument(
        "models",
        nargs="+",
        help="two or more checkpoints: actor files, run checkpoint.pt files, "
        "or model directories",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=5,
        help="points needed to win a leg (default: 5)",
    )
    parser.add_argument(
        "--legs",
        type=int,
        default=2,
        help="legs per pairing, sides swap each leg (default: 2)",
    )
    parser.add_argument(
        "--output",
        default="tournament_results.json",
        help="where to write results JSON",
    )
    parser.add_argument("--render", action="store_true", help="show the games")
    return parser.parse_args()


def play_leg(env, left_actor, right_actor, points, render=False):
    """First to `points` wins. Returns (left_score, right_score)."""
    left = right = 0
    observation = env.reset()
    decisions = 0
    while max(left, right) < points and decisions < MAX_DECISIONS_PER_GAME:
        left_action = left_actor.act(observation)
        right_action = right_actor.act(Tools.invert(observation))
        decisions += 1
        for _ in range(FRAME_SKIP):
            if render:
                env.render()
            observation, r_right, r_left, done = env.step(
                [right_action, left_action]
            )
            if done:
                break
        if done:
            if r_left == 1 and r_right == -1:
                left += 1
            elif r_right == 1 and r_left == -1:
                right += 1
            # timeout (-1, -1): no point awarded
    return left, right


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cpu")
    env = make("Pong-v0")

    def display_name(path):
        base = os.path.basename(os.path.normpath(path))
        return base if base != "checkpoint.pt" else os.path.basename(
            os.path.dirname(os.path.normpath(path))
        )

    players = {}
    for path in args.models:
        if path in ("bot", "__bot__"):
            name, player = "geometry_bot", PerfectDefender()
        else:
            name = display_name(path)
            player = load_policy(
                path, env.num_actions, (env.state_size,), device=device
            )
        if name in players:
            name = f"{name}#{sum(k.startswith(name) for k in players) + 1}"
        players[name] = player
    if len(players) < 2:
        raise SystemExit("need at least two distinct models")

    standings = {
        name: {"wins": 0, "draws": 0, "losses": 0, "pf": 0, "pa": 0}
        for name in players
    }
    matches = []

    for name_a, name_b in itertools.combinations(players, 2):
        total_a = total_b = 0
        for leg in range(args.legs):
            # swap sides each leg to cancel any side bias
            if leg % 2 == 0:
                sa, sb = play_leg(
                    env, players[name_a], players[name_b], args.points, args.render
                )
            else:
                sb, sa = play_leg(
                    env, players[name_b], players[name_a], args.points, args.render
                )
            total_a += sa
            total_b += sb

        if total_a > total_b:
            result = name_a
            standings[name_a]["wins"] += 1
            standings[name_b]["losses"] += 1
        elif total_b > total_a:
            result = name_b
            standings[name_b]["wins"] += 1
            standings[name_a]["losses"] += 1
        else:
            result = "draw"
            standings[name_a]["draws"] += 1
            standings[name_b]["draws"] += 1
        standings[name_a]["pf"] += total_a
        standings[name_a]["pa"] += total_b
        standings[name_b]["pf"] += total_b
        standings[name_b]["pa"] += total_a

        matches.append(
            {
                "match": f"{name_a} X {name_b}",
                "score": [total_a, total_b],
                "winner": result,
            }
        )
        print(f"{name_a} {total_a} x {total_b} {name_b}  ->  {result}")

    ranked = sorted(
        standings.items(),
        key=lambda kv: (kv[1]["wins"], kv[1]["pf"] - kv[1]["pa"]),
        reverse=True,
    )
    print("\nstandings:")
    print(f"{'model':40s} {'W':>3} {'D':>3} {'L':>3} {'PF':>4} {'PA':>4}")
    for name, s in ranked:
        print(
            f"{name:40s} {s['wins']:>3} {s['draws']:>3} {s['losses']:>3}"
            f" {s['pf']:>4} {s['pa']:>4}"
        )

    with open(args.output, "w") as f:
        json.dump({"matches": matches, "standings": standings}, f, indent=2)
    print(f"\nresults written to {args.output}")
