using Pkg; Pkg.activate(@__DIR__)
using Flux
using Statistics: mean, std
using Printf
using UnicodePlots
using FileIO, FreeType   # lets UnicodePlots' savefig write PNG
using Serialization

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
eval_seeds = 1:30   # training uses seeds 1001 and up
train_chances = 0.5:0.05:0.8   # each training episode draws its arrival chance from here
model_dir = "artifacts"   # trained networks, one .jls per agent
plot_dir = "results"      # every chart as .png and .txt

## dqn: replay entries, q network and the learning step
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

# one gradient step towards r + γ max Q_target(next state)
function learn!(model, target, opt_state, batch::Vector{Transition})
    obs = stack(t.obs for t in batch)
    next_obs = stack(t.next_obs for t in batch)
    rewards = Float32[t.reward for t in batch]
    alive = Float32[!t.done for t in batch]
    taken = CartesianIndex.([t.action for t in batch], 1:length(batch))

    y = rewards .+ γ .* alive .* vec(maximum(target(next_obs); dims=1))
    grads = Flux.gradient(m -> Flux.huber_loss(m(obs)[taken], y), model)
    Flux.update!(opt_state, model, grads[1])
end

mutable struct DQNAgent
    rng::MersenneTwister
    model::Chain
    target::Chain
    opt_state::Any
    buffer::Vector{Transition}
    steps::Int
    episode_rewards::Vector{Float64}
end

function DQNAgent(env::Env)
    Random.seed!(static_SEED)
    model = q_network(length(reset!(env)), length(env.resource_queues))
    opt_state = Flux.setup(OptimiserChain(ClipNorm(clip_norm), Adam(learning_rate)), model)
    DQNAgent(MersenneTwister(static_SEED), model, deepcopy(model), opt_state, Transition[], 0, Float64[])
end

# varied training: a new arrival chance every episode, so the agent sees light and heavy load
# episode k's seed and arrival chance depend only on k, so every agent trains on the same workloads
function start_training_episode!(env::Env, episode::Int, chances)
    env.advance_chance = rand(MersenneTwister(1000 + episode), chances)
    return reset!(env; SEED=1000 + episode)
end

function train!(agent::DQNAgent, env::Env, episodes::Int; chances=train_chances)
    resource_count = length(env.resource_queues)
    for _ in 1:episodes
        obs = start_training_episode!(env, length(agent.episode_rewards) + 1, chances)
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
                learn!(agent.model, agent.target, agent.opt_state, rand(agent.rng, agent.buffer, batch_size))
            agent.steps % sync_every == 0 && Flux.loadmodel!(agent.target, agent.model)
        end
        push!(agent.episode_rewards, env.reward)
    end
    return agent
end

## sac (discrete): an actor, two q networks and a learned temperature α, all trained from the replay buffer
const initial_α = 0.1
const target_entropy_ratio = 0.5   # α adjusts so the actor's entropy settles near this share of log(resources)

mutable struct SACAgent
    rng::MersenneTwister
    actor::Chain
    critics::Vector{Chain}
    targets::Vector{Chain}
    log_α::Vector{Float32}
    target_entropy::Float32
    actor_state::Any
    critic_states::Vector{Any}
    α_state::Any
    buffer::Vector{Transition}
    steps::Int
    episode_rewards::Vector{Float64}
end

function SACAgent(env::Env)
    Random.seed!(static_SEED)
    obs_length, resource_count = length(reset!(env)), length(env.resource_queues)
    actor = q_network(obs_length, resource_count)   # same shape; its outputs are action logits
    critics = [q_network(obs_length, resource_count) for _ in 1:2]
    optimiser = OptimiserChain(ClipNorm(clip_norm), Adam(learning_rate))
    log_α = Float32[log(initial_α)]
    SACAgent(MersenneTwister(static_SEED), actor, critics, deepcopy.(critics), log_α,
        target_entropy_ratio * log(resource_count),
        Flux.setup(optimiser, actor), Any[Flux.setup(optimiser, c) for c in critics],
        Flux.setup(Adam(learning_rate), log_α), Transition[], 0, Float64[])
end

# draw an action from the actor's softmax
function sample_action(rng, logits)
    p = softmax(logits)
    return something(findfirst(>=(rand(rng)), cumsum(p)), length(p))
end

function learn!(agent::SACAgent, batch::Vector{Transition})
    obs = stack(t.obs for t in batch)
    next_obs = stack(t.next_obs for t in batch)
    rewards = Float32[t.reward for t in batch]
    alive = Float32[!t.done for t in batch]
    taken = CartesianIndex.([t.action for t in batch], 1:length(batch))
    α = exp(agent.log_α[1])

    # critics: r + γ Σ π(a'|s') [min Q_target(s', a') − α log π(a'|s')]
    next_logp = logsoftmax(agent.actor(next_obs))
    next_q = min.(agent.targets[1](next_obs), agent.targets[2](next_obs))
    y = rewards .+ γ .* alive .* vec(sum(exp.(next_logp) .* (next_q .- α .* next_logp); dims=1))
    for (critic, state) in zip(agent.critics, agent.critic_states)
        grads = Flux.gradient(m -> Flux.huber_loss(m(obs)[taken], y), critic)
        Flux.update!(state, critic, grads[1])
    end

    # actor: minimise Σ π(a|s) [α log π(a|s) − min Q(s, a)]
    q = min.(agent.critics[1](obs), agent.critics[2](obs))
    actor_grads = Flux.gradient(agent.actor) do m
        logp = logsoftmax(m(obs))
        mean(sum(exp.(logp) .* (α .* logp .- q); dims=1))
    end
    Flux.update!(agent.actor_state, agent.actor, actor_grads[1])

    # temperature: lower α when the actor is more random than the target entropy, raise it when less
    logp = logsoftmax(agent.actor(obs))
    entropy = mean(-sum(exp.(logp) .* logp; dims=1))
    α_grads = Flux.gradient(la -> exp(la[1]) * (entropy - agent.target_entropy), agent.log_α)
    Flux.update!(agent.α_state, agent.log_α, α_grads[1])
end

function train!(agent::SACAgent, env::Env, episodes::Int; chances=train_chances)
    for _ in 1:episodes
        obs = start_training_episode!(env, length(agent.episode_rewards) + 1, chances)
        done = false
        while !done
            action = sample_action(agent.rng, agent.actor(obs))
            next_obs, reward, done = step!(env, action)

            transition = Transition(obs, action, reward, next_obs, done)
            length(agent.buffer) < buffer_size ? push!(agent.buffer, transition) :
                (agent.buffer[mod1(agent.steps + 1, buffer_size)] = transition)
            obs = next_obs
            agent.steps += 1

            length(agent.buffer) >= batch_size && learn!(agent, rand(agent.rng, agent.buffer, batch_size))
            if agent.steps % sync_every == 0
                foreach(Flux.loadmodel!, agent.targets, agent.critics)
            end
        end
        push!(agent.episode_rewards, env.reward)
    end
    return agent
end

## build the environments and both agents (rerun to start over)
env = Env(3, 200)         # evaluation, nominal workload
train_env = Env(3, 200)   # training, arrival chance varies per episode
agents = ["dqn" => DQNAgent(env), "sac" => SACAgent(env)]

## train (rerun to keep training the same agents)
for (_, agent) in agents
    train!(agent, train_env, episodes)
end

## training summary
for (name, agent) in agents
    r = agent.episode_rewards
    println(rpad(name, 12), "$(length(r)) episodes, $(agent.steps) steps, reward: first 10 ",
        "$(round(mean(r[1:10]); digits=1)), last 10 $(round(mean(r[end-9:end]); digits=1)), worst $(round(minimum(r); digits=1))")
end

## save the trained models
# the network that picks actions: dqn's q network, sac's actor
policy_network(agent::DQNAgent) = agent.model
policy_network(agent::SACAgent) = agent.actor

mkpath(model_dir)
for (name, agent) in agents
    path = joinpath(model_dir, replace(name, " " => "_") * ".jls")
    serialize(path, (network=policy_network(agent), episode_rewards=agent.episode_rewards, steps=agent.steps))
    println("saved $path")
end
# reload later with: network_policy(deserialize("artifacts/dqn.jls").network)

## compare the trained agents with the baselines
# evaluation is greedy for both: the best q-value, or the actor's most likely action
network_policy(network) = env -> argmax(network(observation(env)))
greedy_policy(agent) = network_policy(policy_network(agent))

results = [name => evaluate(policy, env; seeds=eval_seeds)
           for (name, policy) in ["random" => random_policy(MersenneTwister(static_SEED)),
                                  "round-robin" => round_robin(),
                                  "shortest-queue" => shortest_queue,
                                  [name => greedy_policy(agent) for (name, agent) in agents]...]]

# mean ± 95% confidence interval over the evaluation seeds
ci95(x) = 1.96 * std(x) / sqrt(length(x))
column(runs, metric) = getfield.(runs, metric)
report_metrics = [:reward, :mean_wait, :mean_completion, :utilization, :completed]

println("mean ± 95% CI over $(length(eval_seeds)) evaluation seeds")
@printf("%-16s", "policy"); foreach(m -> @printf("%18s", m), report_metrics); println()
for (name, runs) in results
    @printf("%-16s", name)
    foreach(m -> @printf("%18s", @sprintf("%.2f ± %.2f", mean(column(runs, m)), ci95(column(runs, m)))), report_metrics)
    println()
end

# same seeds for every policy, so compare per seed against the strongest baseline
baseline = Dict(results)["shortest-queue"]
println("\nreward minus shortest-queue, same seed")
for (name, runs) in results
    name == "shortest-queue" && continue
    diff = column(runs, :reward) .- column(baseline, :reward)
    @printf("%-16s%8.2f ± %.2f   (better on %d/%d seeds)\n", name, mean(diff), ci95(diff), count(>(0), diff), length(diff))
end

## generalization: same trained agents on a heavier workload they never saw
# 0.8 × 4.5 = 3.6 units of work per sequence against a capacity of 3 (about 120% load)
shifted_env = Env(3, 200; advance_chance=0.8, job_time=3:6)
shifted_results = [name => evaluate(policy, shifted_env; seeds=eval_seeds)
                   for (name, policy) in ["random" => random_policy(MersenneTwister(static_SEED)),
                                          "round-robin" => round_robin(),
                                          "shortest-queue" => shortest_queue,
                                          [name => greedy_policy(agent) for (name, agent) in agents]...]]

println("\nshifted workload: advance_chance $(shifted_env.advance_chance), job_time $(shifted_env.job_time)")
println("mean ± 95% CI over $(length(eval_seeds)) evaluation seeds")
@printf("%-16s", "policy"); foreach(m -> @printf("%18s", m), report_metrics); println()
for (name, runs) in shifted_results
    @printf("%-16s", name)
    foreach(m -> @printf("%18s", @sprintf("%.2f ± %.2f", mean(column(runs, m)), ci95(column(runs, m)))), report_metrics)
    println()
end

shifted_baseline = Dict(shifted_results)["shortest-queue"]
println("\nshifted reward minus shortest-queue, same seed")
for (name, runs) in shifted_results
    name == "shortest-queue" && continue
    diff = column(runs, :reward) .- column(shifted_baseline, :reward)
    @printf("%-16s%8.2f ± %.2f   (better on %d/%d seeds)\n", name, mean(diff), ci95(diff), count(>(0), diff), length(diff))
end

## visuals: shown here and written to plot_dir
mkpath(plot_dir)
function show_and_save(plot, name)
    display(plot)
    for ext in ("png", "txt")
        savefig(plot, joinpath(plot_dir, "$name.$ext"))
    end
end

## visual: training reward per episode, 10-episode moving average
moving_mean(r) = [mean(r[max(1, i - 9):i]) for i in eachindex(r)]
reward_plot = lineplot(moving_mean(last(agents[1]).episode_rewards); name=first(agents[1]),
    title="training reward (10-episode mean)", xlabel="episode", ylabel="reward", width=60, height=15,
    canvas=AsciiCanvas)   # the png font has no braille, the default canvas
for (name, agent) in agents[2:end]
    lineplot!(reward_plot, moving_mean(agent.episode_rewards); name)
end
show_and_save(reward_plot, "training_reward")

## visual: each policy against each metric
policy_names = first.(results)
for (metric, label) in [:reward => "cumulative reward",
                        :mean_wait => "mean wait (sequences)",
                        :mean_completion => "mean completion time (sequences)",
                        :utilization => "utilization",
                        :completed => "jobs completed"]
    show_and_save(barplot(policy_names, [mean(column(runs, metric)) for (_, runs) in results];
        title="$label, mean of $(length(eval_seeds)) seeds", width=40), "nominal_$metric")
end

## visual: nominal against shifted workload (barplot needs values ≥ 0, so no reward here)
for (metric, label) in [
    :mean_wait => "mean wait (sequences)",
    :mean_completion => "mean completion time (sequences)",
    :completed => "jobs completed"]
    labels, values = String[], Float64[]
    for ((name, nominal), (_, shifted)) in zip(results, shifted_results)
        push!(labels, "$name (nominal)", "$name (shifted)")
        push!(values, mean(column(nominal, metric)), mean(column(shifted, metric)))
    end
    show_and_save(barplot(labels, values; title="$label: nominal vs shifted", width=40), "shifted_$metric")
end
