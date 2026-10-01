include("process.jl")

const wait_cost = 0.1
const busy_bonus = 0.1
const queue_cap = 10

mutable struct Env
    rng::MersenneTwister
    advance_chance::Float64
    job_time::UnitRange{Int}
    waiting_jobs::Vector{Int}
    resource_queues::Vector{Vector{Int}}
    sequence::Int
    sequence_count::Int
    completed::Int
    wait_time::Int
    busy_time::Int
    reward::Float64
end

# workload defaults to process.jl's constants; pass others to test generalization
Env(resource_count::Int, sequence_count::Int;
    advance_chance::Float64=advance_chance, job_time::UnitRange{Int}=job_time) =
    Env(MersenneTwister(static_SEED), advance_chance, job_time, Int[], [Int[] for _ in 1:resource_count],
        0, sequence_count, 0, 0, 0, 0.0)

# state: waiting count, next job's duration, then per resource its queue length and work left
# state: waiting count, next job's duration, then per resource its queue length and work left
# every value is clipped to [0, 1]: counts at queue_cap jobs, work at queue_cap full-length jobs,
# so an overloaded system reads as "full" instead of values the network never saw in training
function observation(env::Env)
    scale = last(job_time)
    next_job = isempty(env.waiting_jobs) ? 0 : first(env.waiting_jobs)
    obs = Float32[min(length(env.waiting_jobs), queue_cap) / queue_cap, min(next_job / scale, 1)]
    for queue in env.resource_queues
        push!(obs, min(length(queue), queue_cap) / queue_cap, min(sum(queue; init=0) / (scale * queue_cap), 1))
    end
    return obs
end

# start an episode with the first arrival drawn
function reset!(env::Env; SEED::Int=static_SEED)
    Random.seed!(env.rng, SEED)
    empty!(env.waiting_jobs)
    foreach(empty!, env.resource_queues)
    env.sequence = env.completed = env.wait_time = env.busy_time = 0
    env.reward = 0.0
    job_manager!(env.rng, env.waiting_jobs; probability=env.advance_chance, duration_range=env.job_time, verbose=false)
    return observation(env)
end

# action: which resource gets the oldest waiting job (ignored if none is waiting)
function step!(env::Env, action::Int)
    1 <= action <= length(env.resource_queues) || throw(ArgumentError("no resource $action"))
    isempty(env.waiting_jobs) || dispatch_to!(env.waiting_jobs, env.resource_queues, action)

    # every job not being worked on is waiting; every non-empty queue is busy
    waiting = length(env.waiting_jobs) + sum(q -> max(length(q) - 1, 0), env.resource_queues)
    busy = count(!isempty, env.resource_queues)
    completed = length(resource_manager!(env.resource_queues; verbose=false))

    env.sequence += 1
    env.completed += completed
    env.wait_time += waiting
    env.busy_time += busy
    reward = completed - wait_cost * waiting + busy_bonus * busy
    env.reward += reward

    job_manager!(env.rng, env.waiting_jobs; probability=env.advance_chance, duration_range=env.job_time, verbose=false)
    return observation(env), reward, env.sequence >= env.sequence_count
end

metrics(env::Env) = (
    completed=env.completed,
    mean_wait=env.wait_time / max(env.completed, 1),
    utilization=env.busy_time / (length(env.resource_queues) * env.sequence),
    reward=env.reward,
)

# baselines: env -> action
random_policy(rng) = env -> rand(rng, 1:length(env.resource_queues))
round_robin() = (next = Ref(1); env -> (a = next[]; next[] = a % length(env.resource_queues) + 1; a))
shortest_queue(env) = argmin(length.(env.resource_queues))

function run_episode(policy, env::Env; SEED::Int=static_SEED)
    reset!(env; SEED)
    done = false
    while !done
        _, _, done = step!(env, policy(env))
    end
    return metrics(env)
end

# one episode per seed; returns each episode's metrics
evaluate(policy, env::Env; seeds=1:30) = [run_episode(policy, env; SEED) for SEED in seeds]

if abspath(PROGRAM_FILE) == @__FILE__
    env = Env(3, 200)
    for (name, policy) in ["random" => random_policy(MersenneTwister(static_SEED)),
                           "round-robin" => round_robin(),
                           "shortest-queue" => shortest_queue]
        println(rpad(name, 16), run_episode(policy, env))
    end
end
