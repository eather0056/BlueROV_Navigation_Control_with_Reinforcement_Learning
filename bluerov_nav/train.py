"""Explicit opt-in training/evaluation. No import-time simulator, downloads or W&B."""
import argparse
import importlib
import json
from pathlib import Path
import random
import numpy as np
from .config import validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-config", action="store_true")
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--evaluate", metavar="CHECKPOINT")
    parser.add_argument("--output", help="New directory; existing directories are never overwritten")
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()
    try:
        config = validate(json.loads(Path(args.config).read_text()))
    except (ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    if args.check_config:
        print("Configuration validated; no simulator, weights, or training started.")
        return
    if not args.output or args.episodes < 1:
        parser.error("--output is required; --episodes must be positive")
    # Lazy import allows config/reward checks without machine-learning dependencies.
    import torch
    from .networks import Actor, Value, TwinQ
    from .algorithms import advantages, fit_value, policy_gradient, trpo_update, OffPolicy
    from .replay import Replay
    from .environment import KEYS
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    rng = np.random.default_rng(config["seed"])
    device = torch.device(config.get("device", "cpu"))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    (output / "provenance.txt").write_text(
        "New navigation experiment. Not a reproduction or validation of the published results.\n"
        f"PyTorch {torch.__version__}; numpy {np.__version__}; device {device}\n")
    actor = Actor(config["history_length"]).to(device)
    value = Value(config["history_length"]).to(device)
    algorithm, settings = config["algorithm"], config["optimizer"]
    optimizer_actor = torch.optim.Adam(actor.parameters(), lr=settings["learning_rate"])
    optimizer_value = torch.optim.Adam(value.parameters(), lr=settings["learning_rate"])
    off = None
    if algorithm in {"sac", "td3"}:
        off = OffPolicy(algorithm, actor, TwinQ(config["history_length"]).to(device), settings)
        replay = Replay(settings["replay_capacity"], rng)
    if args.evaluate:
        # Load only trusted, new-format checkpoints produced by this runner.
        saved = torch.load(args.evaluate, map_location=device, weights_only=True)
        if saved["algorithm"] != algorithm or saved["history_length"] != config["history_length"]:
            raise ValueError("Checkpoint algorithm/history do not match config")
        actor.load_state_dict(saved["actor"])
        actor.eval()

    def tensor(x):
        return torch.as_tensor(x, dtype=torch.float32, device=device)

    def one(obs):
        return {k: tensor(obs[k]).unsqueeze(0) for k in KEYS}

    def stack(obs):
        return {k: tensor(np.stack([o[k] for o in obs])) for k in KEYS}

    def select(obs, deterministic=False):
        with torch.no_grad():
            batched = one(obs)
            if deterministic or algorithm == "td3":
                a = actor.deterministic(batched)
                latent, lp = None, None
            else:
                a, lp, latent = actor.sample(batched)
        return a[0].cpu().numpy(), lp, latent

    module, factory = config["env_factory"].split(":")
    make_env = getattr(importlib.import_module(module), factory)
    env = make_env(config["env"], config["history_length"], config["seed"])
    cumulative_episodes, cumulative_steps = 0, 0
    episode_return, episode_steps = 0.0, 0
    try:
        obs = env.reset()
        if args.evaluate:
            for episode in range(args.episodes):
                total, count = 0.0, 0
                while True:
                    action, _, _ = select(obs, deterministic=True)
                    obs, reward, term, trunc, info = env.step(action)
                    total, count = total + reward, count + 1
                    if term or trunc:
                        with (output / "evaluation.jsonl").open("a") as f:
                            f.write(json.dumps({"episode": episode + 1, "return": total,
                                                "steps": count, **info}) + "\n")
                        if episode + 1 < args.episodes:
                            obs = env.reset()
                        break
            return
        for iteration in range(1, config["iterations"] + 1):
            rows, completed, losses = [], [], {}
            for _ in range(config["batch_steps"]):
                action, logp, latent = select(obs)
                if off:
                    if cumulative_steps < settings["warmup_steps"]:
                        action = rng.uniform(-1, 1, size=2).astype(np.float32)
                    elif algorithm == "td3":
                        action = np.clip(action + rng.normal(0, settings["exploration_noise"], 2), -1, 1).astype(np.float32)
                nxt, reward, term, trunc, info = env.step(action)
                episode_return += reward
                episode_steps += 1
                cumulative_steps += 1
                if off:
                    replay.add(obs, action, reward, nxt, term)
                    if len(replay) >= settings["replay_batch"] and cumulative_steps >= settings["warmup_steps"]:
                        for _ in range(settings["gradient_steps"]):
                            b = replay.sample(settings["replay_batch"])
                            batch = ({k: tensor(v) for k, v in b[0].items()}, tensor(b[1]), tensor(b[2]),
                                     {k: tensor(v) for k, v in b[3].items()}, tensor(b[4]))
                            losses = off.update(batch)
                else:
                    with torch.no_grad():
                        v, nv = value(one(obs)), value(one(nxt))
                    rows.append((obs, latent.detach(), logp.detach(), reward, term, term or trunc, v, nv))
                obs = nxt
                if term or trunc:
                    cumulative_episodes += 1
                    record = {"episode": cumulative_episodes, "iteration": iteration,
                              "return": episode_return, "steps": episode_steps,
                              "terminated": term, "truncated": trunc, **info}
                    completed.append(record)
                    with (output / "episodes.jsonl").open("a") as f:
                        f.write(json.dumps(record) + "\n")
                    episode_return, episode_steps = 0.0, 0
                    obs = env.reset()
            if not off:
                observations = stack([r[0] for r in rows])
                latents, old_log = torch.cat([r[1] for r in rows]), torch.cat([r[2] for r in rows])
                rewards = tensor([[r[3]] for r in rows])
                terminated = tensor([[r[4]] for r in rows])
                ends = tensor([[r[5]] for r in rows])
                values, next_values = torch.cat([r[6] for r in rows]), torch.cat([r[7] for r in rows])
                adv, returns = advantages(rewards, values, next_values, terminated, ends,
                                          settings["gamma"], settings["gae_lambda"])
                if algorithm == "trpo":
                    losses["value_loss"] = fit_value(value, optimizer_value, observations, returns, settings["l2_reg"])
                    losses.update(trpo_update(actor, observations, latents, adv, settings["max_kl"], settings["damping"]))
                else:
                    history = []
                    epochs = settings["epochs"] if algorithm == "ppo" else 1
                    batch_size = settings["minibatch_size"] if algorithm == "ppo" else len(rows)
                    for _ in range(epochs):
                        indices = rng.permutation(len(rows))
                        for start in range(0, len(rows), batch_size):
                            idx = torch.as_tensor(indices[start:start + batch_size], device=device)
                            minibatch = {k: v[idx] for k, v in observations.items()}
                            vl = fit_value(value, optimizer_value, minibatch, returns[idx], settings["l2_reg"])
                            pl = policy_gradient(actor, optimizer_actor, minibatch, latents[idx], adv[idx],
                                                 old_log[idx], settings["clip"] if algorithm == "ppo" else None)
                            history.append((pl, vl))
                    losses = dict(zip(("policy_loss", "value_loss"), np.mean(history, axis=0).tolist()))
            metrics = {"iteration": iteration, "environment_transitions": cumulative_steps,
                       "cumulative_episodes": cumulative_episodes, "completed_this_iteration": len(completed),
                       "average_episode_return": (float(np.mean([r["return"] for r in completed])) if completed else None),
                       "success_ratio": (float(np.mean([r["success"] for r in completed])) if completed else None),
                       **losses}
            with (output / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(metrics, allow_nan=False) + "\n")
            print(json.dumps(metrics, allow_nan=False), flush=True)
            if iteration % config["checkpoint_interval"] == 0 or iteration == config["iterations"]:
                checkpoint = {"actor": actor.state_dict(), "value": value.state_dict(),
                              "algorithm": algorithm, "history_length": config["history_length"],
                              "iteration": iteration}
                if off:
                    checkpoint["critic"] = off.critic.state_dict()
                torch.save(checkpoint, output / f"policy-{iteration:04d}.pt")
    finally:
        env.close()


if __name__ == "__main__":
    main()
