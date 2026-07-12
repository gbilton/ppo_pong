# ppo_pong — PPO self-play Pong

Custom pygame Pong env + PPO (actor/critic MLPs, 7-dim state, 3 actions).
Training is **CPU-only by design** — the nets are tiny, so per-call GPU
overhead loses to CPU ~3x. More CPU cores = more steps/sec.

## Key files

- `main_vec.py` — THE trainer: 32 vectorized envs, opponent pool, resume,
  TensorBoard, fixed-reference evals. (`main.py` is the legacy single-env one.)
- `ppo_torch.py` — Agent/networks; KL early-stop; entropy auto-controller
  (targets 0.65, adjusts entropy_coef each update, persisted in checkpoints).
- `pong.py` — env. Physics: ball accelerates +1 velx per paddle hit up to
  `maxvel=22` (past ~16 px/frame no defender can cover the court, so every
  rally decides); paddle-hit spin capped at |vely|<=15.
- `perfect_bot.py` — scripted geometry defender. Concedes only to objectively
  hard shots => used as a pure-offense grader (eval) and as 15% of training
  opponents. Also a benchmark: `perfect_bot.py --model <ckpt>`.
- `dashboard.py` — Flask control panel on :8787 (telemetry, start/stop/resume,
  tournaments, browser play via WebSocket at /play).
- `tournament.py` / `play.py` — round-robin between checkpoints / play vs a model.
- `bootstrap_vps.sh` — new-machine setup + start (see file header).

## Running

```bash
.venv/bin/python main_vec.py                      # fresh run
.venv/bin/python main_vec.py --resume             # continue runs/latest
.venv/bin/python main_vec.py --init-weights DIR   # new run, warm-started
```

Each run is self-contained in `runs/<name>/`: TB events + `checkpoint.pt`
(weights, BOTH optimizers, counters, entropy_coef) + `pool/` (opponents).
Checkpoints every 50 updates (~20s) — stopping is always safe.

Operations run in tmux sessions on the training box:
`ppo` (trainer), `tb` (tensorboard :6006, in a restart loop), `ui` (dashboard :8787).
Access via `ssh -L 8787:localhost:8787 -L 6006:localhost:6006 <host>`.
**Stop training ONLY via `tmux kill-session -t ppo` or the dashboard API**
(`curl -X POST localhost:8787/api/train/stop`); a `pkill -f` pattern that
matches the tmux server kills every session (this has happened).

## Self-play design (and why)

- **Opponent pool, not latest-only**: each episode samples 15% geometry bot,
  then latest champion / anchor (run origin) / random past champion at
  0.5/0.25/0.25. Latest-only self-play provably cycled (10 promotions that
  lost to their own seed). The anchor is never evicted.
- **Promotion gate**: mean score over last 300 non-bot episodes >= 0.4, with
  >= 2000 non-bot episodes between promotions (spacing keeps pool members
  distinct). Promotions export to `tmp/docker/models/` and `runs/<run>/pool/`.
- **Gate != audit.** The honest strength metrics, every 200 updates:
  `eval/vs_<refname>_*` (deterministic games vs the fixed --eval-ref model,
  default actor_goat) and `eval/vs_geometry_bot_*` (offense quality).
  Tournaments (dashboard or CLI) are the final judge.

## Reading TensorBoard (healthy ranges)

- `train/approx_kl` < ~0.03 (early-stop trips above 0.05: `epochs_used` < 10)
- `train/clip_fraction` ~0.05-0.2; `train/entropy` ~0.65 (auto-controlled;
  `train/entropy_coef` shows the controller working)
- `train/explained_variance`: sags at promotions (opponent changes), fine
- `eval/vs_geometry_bot_score` trending up = offense improving (the goal)
- promotions flowing but not >1/min; `episode/length` inflating toward the
  timeout cap = stalemate regime (should not happen since maxvel=22)

## Model zoo (`tmp/docker/legacy_models/`)

- `actor_goat` — historical champion (pre-2026-07-11). Dethroned.
- `actor_goat_v2` + `critic_goat_v2` — pool_004 of the ppo_vec_pool run:
  beat actor_goat 10-0. Seed dir: `tmp/docker/goat_v2_seed/`.
- `run1_gpu/`, `run2_vec/` — earlier run champions (weaker, for tournaments).

## Migrating to a new machine (one step)

The repo carries a snapshot of the live run (`runs/ppo_vec_accel`) and the
model zoo, so a fresh box needs only:

    curl -fsSL https://raw.githubusercontent.com/gbilton/ppo_pong/main/bootstrap_vps.sh | bash

That clones, installs (CPU torch), and resumes training + tensorboard +
dashboard in tmux. **Before retiring the old training box**, refresh the
snapshot from a machine that can reach it: `./snapshot_push.sh [host]`
(rsyncs state into the repo, commits, pushes). Old runs' full TB histories
are archived on the Mac at ~/src/personal/ppo_pong/runs (not in git).
