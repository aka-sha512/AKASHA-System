using Flux
using Statistics: mean
using UnicodePlots

include("env.jl")

const γ = 0.95
const learning_rate = 1e-3
const batch_size = 32
const buffer_size = 10_000
const sync_every = 200
const ε_min = 0.05
const ε_decay = 10_000
const clip_norm = 10.0
episodes = 100

## replay entries, q network and the learning step
struct Transition
    obs::Vector{Float32}
    action::Int
    reward::Float32
    next_obs::Vector{Float32}
    done::Bool
end

# q network: state in, one value per resource out
q_network(obs_length::Int, resource_count::Int) =
    Chain(Dense(obs_length => 32, relu), Dense(32 => resource_count))

# one gradient step towards r + γ Q_target(next state, a*)
# dqn: a* = argmax Q_target; double dqn: a* = argmax Q_model, scored by Q_target
function learn!(model, target, opt_state, batch::Vector{Transition}; double::Bool)
    obs = stack(t.obs for t in batch)
    next_obs = stack(t.next_obs for t in batch)
    rewards = Float32[t.reward for t in batch]
    alive = Float32[!t.done for t in batch]
    taken = CartesianIndex.([t.action for t in batch], 1:length(batch))

    next_q = target(next_obs)
    best = double ? argmax(model(next_obs); dims=1) : argmax(next_q; dims=1)
    y = rewards .+ γ .* alive .* vec(next_q[best])
    grads = Flux.gradient(m -> Flux.huber_loss(m(obs)[taken], y), model)
    Flux.update!(opt_state, model, grads[1])
end

mutable struct Agent
    double::Bool
    rng::MersenneTwister
    model::Chain
    target::Chain
    opt_state::Any
    buffer::Vector{Transition}
    steps::Int
    episode_rewards::Vector{Float64}
end

# same seed, same starting weights: the two agents differ only in their target
function Agent(env::Env; double::Bool)
    Random.seed!(static_SEED)
    model = q_network(length(reset!(env)), length(env.resource_queues))
    opt_state = Flux.setup(OptimiserChain(ClipNorm(clip_norm), Adam(learning_rate)), model)
    Agent(double, MersenneTwister(static_SEED), model, deepcopy(model), opt_state, Transition[], 0, Float64[])
end

function train!(agent::Agent, env::Env, episodes::Int)
    resource_count = length(env.resource_queues)
    for _ in 1:episodes
        obs = reset!(env; SEED=1000 + length(agent.episode_rewards) + 1)
        done = false
        while !done
            ε = max(ε_min, 1 - agent.steps / ε_decay)
            action = rand(agent.rng) < ε ? rand(agent.rng, 1:resource_count) : argmax(agent.model(obs))
            next_obs, reward, done = step!(env, action)

            transition = Transition(obs, action, reward, next_obs, done)
            length(agent.buffer) < buffer_size ? push!(agent.buffer, transition) :
                (agent.buffer[mod1(agent.steps + 1, buffer_size)] = transition)
            obs = next_obs
            agent.steps += 1

            length(agent.buffer) >= batch_size &&
                learn!(agent.model, agent.target, agent.opt_state, rand(agent.rng, agent.buffer, batch_size); agent.double)
            agent.steps % sync_every == 0 && Flux.loadmodel!(agent.target, agent.model)
        end
        push!(agent.episode_rewards, env.reward)
    end
    return agent
end

## build the environment and both agents (rerun to start over)
env = Env(3, 200)
agents = ["dqn" => Agent(env; double=false), "double dqn" => Agent(env; double=true)]

## train (rerun to keep training the same agents)
for (_, agent) in agents
    train!(agent, env, episodes)
end

## training summary
for (name, agent) in agents
    r = agent.episode_rewards
    println(rpad(name, 12), "$(length(r)) episodes, $(agent.steps) steps, reward: first 10 ",
        "$(round(mean(r[1:10]); digits=1)), last 10 $(round(mean(r[end-9:end]); digits=1)), worst $(round(minimum(r); digits=1))")
end

## compare the trained agents with the baselines
greedy_policy(model) = env -> argmax(model(observation(env)))

results = [name => run_episode(policy, env)
           for (name, policy) in ["random" => random_policy(MersenneTwister(static_SEED)),
                                  "round-robin" => round_robin(),
                                  "shortest-queue" => shortest_queue,
                                  [name => greedy_policy(agent.model) for (name, agent) in agents]...]]
foreach(((name, m),) -> println(rpad(name, 16), m), results)

## visual: training reward per episode, 10-episode moving average
moving_mean(r) = [mean(r[max(1, i - 9):i]) for i in eachindex(r)]
reward_plot = lineplot(moving_mean(last(agents[1]).episode_rewards); name=first(agents[1]),
    title="training reward (10-episode mean)", xlabel="episode", ylabel="reward", width=60, height=15)
for (name, agent) in agents[2:end]
    lineplot!(reward_plot, moving_mean(agent.episode_rewards); name)
end
display(reward_plot)

## visual: each policy against each metric
policy_names = first.(results)
for (metric, label) in [:reward => "cumulative reward",
                        :mean_wait => "mean wait (sequences)",
                        :utilization => "utilization",
                        :completed => "jobs completed"]
    display(barplot(policy_names, [getfield(m, metric) for (_, m) in results]; title=label, width=40))
end
