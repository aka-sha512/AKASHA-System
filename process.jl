
using Random

const advance_chance = 0.7
const job_time = 2:5
const static_SEED = 1

# read user input via communication prompt
function prompt(prompt::AbstractString)
    while true
        print(prompt)
        input = strip(readline())
        value = tryparse(Int, input)

        if isnothing(value) | value < 1
            println("enter a natural number")
        else
            return value
        end
    end
end

# throw static states unto the console
function state_manager(waiting_jobs::Vector{Int}, resource_queues::Vector{Vector{Int}})
    if isempty(waiting_jobs)
        println("no waiting jobs...")
    else
        println("$waiting_jobs jobs waiting...")
    end
    for (resource_index, queue) in enumerate(resource_queues)
        if isempty(queue)
            println("resource $resource_index: open")
        else
            println("resource $resource_index: $queue")
        end
    end
end

# bernoulli
function job_manager!(
    rng::AbstractRNG, waiting_jobs::Vector{Int};
    probability::Float64, duration_range::UnitRange{Int})
    # A random draw below the configured probability means one job arrives.
    if rand(rng) < probability
        duration = rand(rng, duration_range)
        push!(waiting_jobs, duration)
        println("new job; it takes $duration")
        return true
    end
    println("(b) no new jobs...")
    return false
end

# round-robin
function dispatch!(waiting_jobs::Vector{Int}, resource_queues::Vector{Vector{Int}}, next_resource::Int)
    if isempty(waiting_jobs)
        println("(rr) no new jobs...")
        return next_resource
    end

    job_duration = popfirst!(waiting_jobs)
    push!(resource_queues[next_resource], job_duration)
    println("sent job to resource $next_resource.")

    return next_resource % length(resource_queues) + 1
end

# preemptive shortest
function resource_manager!(resource_queues::Vector{Vector{Int}})
    for (resource_index, queue) in enumerate(resource_queues)
        if isempty(queue)
            continue
        end

        job_index = argmin(queue)
        queue[job_index] -= 1

        if queue[job_index] == 0
            deleteat!(queue, job_index)
            println("job at resource $resource_index has completed")
        end
    end
end

# parent process to manage the task management children
function task_master(
    resource_count::Int, sequence_count::Int;
    advance_chance::Float64=advance_chance,
    duration_range::UnitRange{Int}=job_time,
    SEED::Int=static_SEED,
    )

    rng = MersenneTwister(SEED)
    waiting_jobs = Int[]
    resource_queues = [Int[] for _ in 1:resource_count]
    next_resource = 1

    for sequence in 1:sequence_count
        println("\nSequence $sequence")
        job_manager!(rng, waiting_jobs; probability=advance_chance, duration_range)
        next_resource = dispatch!(waiting_jobs, resource_queues, next_resource)
        resource_manager!(resource_queues)
        state_manager(waiting_jobs, resource_queues)
    end

    return (waiting=waiting_jobs, resources=resource_queues, next_resource=next_resource)
end

println("this is a scheduling algorithm to be used the AKASHA System")
resource_count = prompt("enter resource count: ")
sequence_count = prompt("enter sequence length: ")

deadline = task_master(resource_count, sequence_count)
overdue = length(deadline.waiting)
in_progress = sum(length, deadline.resources; init=0)

println("\nwork complete.")
println("jobs waiting: $overdue")
println("jobs in-progress: $in_progress")
