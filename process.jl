using Random

const advance_chance = 0.7
const job_time = 2:5
const static_SEED = 1

work_left(job::Integer) = job
advance(job::Integer, units::Integer=1) = job - units

# read user input via communication prompt
function prompt(prompt::AbstractString)
    while true
        print(prompt)
        input = strip(readline())
        value = tryparse(Int, input)

        if isnothing(value) || value < 1
            println("enter a natural number")
        else
            return value
        end
    end
end

# throw static states unto the console
function state_manager(waiting_jobs::AbstractVector, resource_queues::AbstractVector)
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
    rng::AbstractRNG, waiting_jobs::AbstractVector;
    probability::Float64, duration_range::UnitRange{Int},
    make_job=identity, verbose::Bool=true)
    # A random draw below the configured probability means one job arrives.
    if rand(rng) < probability
        duration = rand(rng, duration_range)
        push!(waiting_jobs, make_job(duration))
        verbose && println("new job; it takes $duration")
        return true
    end
    verbose && println("(b) no new jobs...")
    return false
end

# send the oldest waiting job to a chosen resource
function dispatch_to!(waiting_jobs::AbstractVector, resource_queues::AbstractVector, resource::Int)
    push!(resource_queues[resource], popfirst!(waiting_jobs))
    return resource
end

# round-robin
function dispatch!(
    waiting_jobs::AbstractVector, resource_queues::AbstractVector, next_resource::Int;
    verbose::Bool=true)
    if isempty(waiting_jobs)
        verbose && println("(rr) no new jobs...")
        return next_resource
    end

    dispatch_to!(waiting_jobs, resource_queues, next_resource)
    verbose && println("sent job to resource $next_resource.")

    return next_resource % length(resource_queues) + 1
end

# preemptive shortest
function resource_manager!(
    resource_queues::AbstractVector{<:AbstractVector{T}};
    speeds=nothing, verbose::Bool=true) where {T}
    completed = Tuple{Int,T}[]
    for (resource_index, queue) in enumerate(resource_queues)
        capacity = isnothing(speeds) ? 1 : speeds[resource_index]

        while capacity > 0 && !isempty(queue)
            remaining, job_index = findmin(work_left, queue)
            units = min(capacity, remaining)
            queue[job_index] = advance(queue[job_index], units)
            capacity -= units

            if work_left(queue[job_index]) == 0
                push!(completed, (resource_index, popat!(queue, job_index)))
                verbose && println("job at resource $resource_index has completed")
            end
        end
    end
    return completed
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

if abspath(PROGRAM_FILE) == @__FILE__
    println("this is a scheduling algorithm to be used the AKASHA System")
    resource_count = prompt("enter resource count: ")
    sequence_count = prompt("enter sequence length: ")

    deadline = task_master(resource_count, sequence_count)
    overdue = length(deadline.waiting)
    in_progress = sum(length, deadline.resources; init=0)

    println("\nwork complete.")
    println("jobs waiting: $overdue")
    println("jobs in-progress: $in_progress")
end
