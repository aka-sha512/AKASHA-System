# AKASHA System

Deep reinforcement learning for dynamic job scheduling, in Julia. Jobs arrive over time with different amounts of work, and an agent learns which computing resource each job should go to. Two DRL agents (DQN and SAC) are trained and compared against three classic scheduling baselines, on the workload they trained for and on a heavier one they never saw.

## Requirements and setup

- Julia (tested with 1.13.1)

From this folder:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Dependencies are listed in `Project.toml`: Flux (neural networks), UnicodePlots (charts), and FileIO, FreeType and ImageIO (PNG export). `Manifest.toml` is not committed, so `instantiate` resolves the latest compatible versions.

## Running

Run commands from the project folder, since outputs are written to `artifacts/` and `results/` relative to the current directory.

| Command | What it does |
|---|---|
| `julia --project=. process.jl` | The original interactive simulator: asks for a resource count and sequence length, then runs round-robin scheduling and prints every step |
| `julia --project=. env.jl` | Runs the three baselines for one episode and prints their metrics |
| `julia --project=. core.jl` | Trains both agents, saves them, evaluates every policy on both workloads, and prints and saves the charts (about 25 seconds) |

`core.jl` is split into `##` cells, so it can also be run cell by cell in VS Code (Shift+Enter). It activates its own project environment on its first line. Rerunning the build cell starts over, and rerunning the train cell continues training the same agents.

## Project layout

```
process.jl    simulator: Bernoulli arrivals, dispatch, shortest-job-first processing
env.jl        RL environment on top of process.jl: reset!, step!, observation,
              metrics, baselines, run_episode, evaluate
core.jl       DQN and SAC agents, training, saving, evaluation, charts
artifacts/    trained networks: dqn.jls, sac.jls
results/      charts, each as .png and .txt
```

```julia
using Flux, Serialization
include("env.jl")
dqn = deserialize("artifacts/dqn.jls").network
evaluate(env -> argmax(dqn(observation(env))), Env(3, 200))
```

Charts are in `results/`: the training curve, every metric on the nominal workload, and nominal against shifted for wait, completion time and jobs completed.

## License

GPL-2.0. See `LICENSE`.
