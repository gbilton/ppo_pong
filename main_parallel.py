import json
import shutil
import numpy as np
from ppo_torch import Agent
from pong import *
from multiprocessing import (
    get_start_method,
    set_start_method,
    Process,
    Event,
    Queue,
    Lock,
    cpu_count,
)

# Ensure the "spawn" start method is used for multiprocessing
if get_start_method(allow_none=True) != "spawn":
    set_start_method("spawn")


def worker_agent(
    worker_id,
    data_queue,
    score_queue,
    stop_event,
    update_started,
    update_complete_event,
    policy_lock,
    update_workers,
):
    print(f"Starting worker {worker_id}")
    env = make("Pong-v0")

    device = torch.device("cpu")

    agent = Agent(
        n_actions=env.num_actions, input_dims=(env.state_size,), device=device
    )
    agent.actor.checkpoint_file = os.path.join("./tmp/docker/legacy_models", "actor")
    agent.critic.checkpoint_file = os.path.join("./tmp/docker/legacy_models", "critic")

    with policy_lock:
        agent.load_models()

    # bot_names = ["actor", "actor copy", "actor_torch_ppo_1", "actor_goat"]
    bot_names = [
        "actor copy",
        "actor_torch_ppo",
        "actor_goat copy",
        "actor_goat copy 2",
        "actor_goat copy 3",
        "actor_goat copy 3",
        "actor_goat copy 4",
        "actor_goat copy 4",
        "actor_goat copy 4",
        "actor_goat copy 5",
        "actor_goat copy 5",
        "actor_goat copy 5",
        "actor_goat copy 5",
        "actor_torch_ppo_1",
    ]
    bots = []
    for bot_name in bot_names:
        bot = Agent(
            n_actions=env.num_actions, input_dims=(env.state_size,), device=device
        )
        bot.actor.checkpoint_file = os.path.join("./tmp/docker/legacy_models", bot_name)
        bot.actor.load_checkpoint()
        bot.actor.eval()
        bots.append(bot)

    p = DQN(device=device)
    p.load_state_dict(torch.load("tmp/dqn/model100x3.pth", map_location=device))
    p.eval()
    bots.append(p)

    n_steps = 0
    datas = []
    while not stop_event.is_set():
        bot = random.choice(bots)
        observation = env.reset()
        done = False
        score = 0

        while not done:
            if update_workers.is_set():
                bot.actor.load_state_dict(agent.actor.state_dict())
                bot.actor.eval()
                update_workers.clear()

            if (n_steps + 1) % 4000 == 0:
                data_queue.put(datas)
                datas = []

            if update_started.is_set():
                update_complete_event.wait()
                update_complete_event.clear()
                with policy_lock:
                    agent.load_models()

            action, prob, val = agent.choose_action(observation)
            inverted_observation = Tools.invert(observation)
            if isinstance(bot, DQN):
                bot_action = bot.act(inverted_observation)
            else:
                bot_action = bot.actor.act(inverted_observation)
            for _ in range(4):
                observation_, _, reward, done = env.step([bot_action, action])
                if done:
                    break
            score += reward
            data = (observation, action, prob, val, reward, done)
            datas.append(data)
            n_steps += 1
            observation = observation_

        score_queue.put(score)


def central_learner(
    data_queue,
    score_queue,
    stop_event,
    update_started,
    update_complete_event,
    policy_lock,
    num_workers,
    update_workers,
):
    print(f"Starting central agent")

    env = make("Pong-v0")

    central_agent = Agent(
        n_actions=env.num_actions,
        input_dims=(env.state_size,),
        batch_size=1000,  # You can adjust this as needed
    )
    central_agent.actor.checkpoint_file = os.path.join(
        "./tmp/docker/legacy_models", "actor"
    )
    central_agent.critic.checkpoint_file = os.path.join(
        "./tmp/docker/legacy_models", "critic"
    )

    with policy_lock:
        central_agent.load_models()

    json_save_directory = "./tmp/docker/json_files/"
    json_file_path = f"{json_save_directory}avg_score_replica5.json"

    update_threshold = 4000 * num_workers
    memories = 0
    scores = [-1 for _ in range(100)]
    avg_scores = []
    best_score = -1

    while not stop_event.is_set():
        if not data_queue.empty():
            # observation, action, prob, val, reward, done = data_queue.get()
            datas = data_queue.get()
            for data in datas:
                observation, action, prob, val, reward, done = data
                central_agent.remember(observation, action, prob, val, reward, done)
                memories += 1

        if not score_queue.empty():
            score = score_queue.get()
            scores.append(score)
            avg_score = np.mean(scores[-100:])
            avg_scores.append(avg_score)
            if avg_score > best_score:
                best_score = avg_score
                torch.save(
                    central_agent.actor.state_dict(),
                    "./tmp/docker/legacy_models/actor_goat",
                )
                torch.save(
                    central_agent.critic.state_dict(),
                    "./tmp/docker/legacy_models/critic_goat",
                )
                # update_workers.set()
                # scores = [-1 for _ in range(100)]

        if memories >= update_threshold:
            # Notify all workers that a policy update is starting
            update_started.set()

            # Update the central agent's model using experiences in the buffer
            print("Learning...")
            central_agent.learn()
            memories = 0

            print("Saving Model...")
            central_agent.save_models()

            scores = scores[-100:]

            # Notify all workers that the policy update is complete
            update_complete_event.set()
            update_started.clear()

        if len(scores) % 100 == 0:
            avg_score_data = {
                "replica_id": 5,
                "avg_scores": avg_scores,
                "episodes_time": 1,
            }

            with open(json_file_path, "w") as json_file:
                json.dump(avg_score_data, json_file)


if __name__ == "__main__":
    shutil.copy(
        src="./tmp/docker/legacy_models/actor_goat",
        dst="./tmp/docker/legacy_models/actor",
    )
    shutil.copy(
        src="./tmp/docker/legacy_models/critic_goat",
        dst="./tmp/docker/legacy_models/critic",
    )

    num_workers = cpu_count() - 2

    data_queue = Queue()
    score_queue = Queue()
    stop_event = Event()
    policy_lock = Lock()

    # Create an event to signal the start of a policy update
    update_started = Event()

    # Create an event to signal the completion of a policy update
    update_complete_event = Event()

    update_workers = Event()

    workers = []
    for i in range(1, num_workers + 1):
        worker = Process(
            target=worker_agent,
            args=(
                i,
                data_queue,
                score_queue,
                stop_event,
                update_started,
                update_complete_event,
                policy_lock,
                update_workers,
            ),
        )
        workers.append(worker)

    for worker in workers:
        worker.start()

    central_learner_process = Process(
        target=central_learner,
        args=(
            data_queue,
            score_queue,
            stop_event,
            update_started,
            update_complete_event,
            policy_lock,
            num_workers,
            update_workers,
        ),
    )
    central_learner_process.start()

    for process in workers:
        process.join()

    central_learner_process.join()
