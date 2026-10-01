
# Flux for machine learning
using Flux
using Statistics: mean

include("env.jl")

const γ = 0.95
const batch_size = 32
const sync_every = 200
const ε_decay = 10_000

# network, target copy and optimizer
rng = MersenneTwister(static_SEED)
env = Env(3, 200)
resource_count = length(env.resource_queues)
model = Chain(Dense(length(reset!(env)) => 32, relu), Dense(32 => resource_count))
target = deepcopy(model)
opt_state = Flux.setup(Adam(1e-3), model)

# one gradient step towards r + γ max Q_target(next state)
function learn!(batch)
    obs = stack(first.(batch))
    actions = [b[2] for b in batch]
    rewards = Float32[b[3] for b in batch]
    next_obs = stack([b[4] for b in batch])
    alive = Float32[!b[5] for b in batch]

    y = rewards .+ γ .* alive .* vec(maximum(target(next_obs); dims=1))
    taken = CartesianIndex.(actions, 1:length(batch))
    grads = Flux.gradient(m -> mean((m(obs)[taken] .- y) .^ 2), model)
    Flux.update!(opt_state, model, grads[1])
end

# training seeds stay clear of seed 1
function train!(episodes)
    buffer, steps, episode_rewards = [], 0, Float64[]
    for episode in 1:episodes
        obs, done = reset!(env; SEED=1000 + episode), false  
        while !done
            ε = max(0.05, 1 - steps / ε_decay)
            action = rand(rng) < ε ? rand(rng, 1:resource_count) : argmax(model(obs))
            next_obs, reward, done = step!(env, action)
            push!(buffer, (obs, action, reward, next_obs, done))
            length(buffer) > 10_000 && popfirst!(buffer)
            obs, steps = next_obs, steps + 1

            length(buffer) >= batch_size && learn!(rand(rng, buffer, batch_size))
            steps % sync_every == 0 && Flux.loadmodel!(target, model)
        end
        push!(episode_rewards, env.reward)
    end
    return episode_rewards
end

episode_rewards = train!(100)
println("training reward: first 10 episodes $(round(mean(episode_rewards[1:10]); digits=1)), ",
    "last 10 $(round(mean(episode_rewards[end-9:end]); digits=1))")

# compare with the baselines
dqn = env -> argmax(model(observation(env)))
for (name, policy) in ["random" => random_policy(MersenneTwister(static_SEED)),
                       "round-robin" => round_robin(),
                       "shortest-queue" => shortest_queue,
                       "dqn" => dqn]
    println(rpad(name, 16), run_episode(policy, env))
end
