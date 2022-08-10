using LinearAlgebra, BenchmarkTools, .Threads, PyPlot, Statistics

k(vᵢ::Integer, G::AbstractMatrix) = count(!iszero, G[vᵢ, :])
k(eᵢ::AbstractVector) = count(!iszero, eᵢ)
k(G::AbstractMatrix) = mapslices(k, G; dims = 2)

function denovoNBI(F₀::AbstractMatrix{Float64})
    # Transfer matrix
    W = F₀ ./ k(F₀)
    replace!(W, Inf => 0.0)
    replace!(W, NaN => 0.0)

    return W
end

function SDTNBI(F₀::AbstractMatrix{Float64})::Matrix
    n_nodes = size(F₀, 1)# Number of nodes
    W = zeros(n_nodes, n_nodes)# Transfer matrix

    for idx in 1:n_nodes
        W[idx, :] = k(idx, F₀) > 0 ? F₀[idx, :] ./ k(idx, F₀) : zeros(n_nodes, 1)
    end

    return W
end

function SDTNBI_threaded(F₀::AbstractMatrix{Float64})::Matrix
    n_nodes = size(F₀, 1)# Number of nodes
    W = zeros(n_nodes, n_nodes)# Transfer matrix

    @threads for idx in 1:n_nodes
        @inbounds W[idx, :] = k(idx, F₀) > 0 ? F₀[idx, :] ./ k(idx, F₀) : zeros(n_nodes, 1)
    end

    return W
end

N = [2^n for n in 0:12]
times = Dict(
    "N" => [[] for _ in N],
    "SDTNBI" => [[] for _ in N],
    "Threaded SDTNBI" => [[] for _ in N],
    "denovoNBI" => [[] for _ in N]
)

a = Matrix{Float64}(rand(10,10) .> 0.1)
SDTNBI(a)
SDTNBI_threaded(a)
denovoNBI(a)

for (i,n) in enumerate(N)
    for run in 1:10
        println("Size = $n (run #$run)")
        # Query network
        DD = rand(n, n)
        DD = @. (DD + DD') / 2
        DD[diagind(DD)] .= 1
        CD = rand(1, n)
        DT = rand(n, n) .> 0.5

        F₀ = Matrix{Float64}([
            zeros(1, 1) zeros(1, n) CD zeros(1, n)
            zeros(n, 1) zeros(n, n) DD DT
            CD' DD' zeros(n, n) zeros(n, n)
            zeros(n, 1) DT' zeros(n, n) zeros(n, n)
        ] .> 0.3
        )

        # Calculate de novo NBI
        push!(times["N"][i], n)
        push!(times["SDTNBI"][i], @elapsed SDTNBI(F₀))
        push!(times["Threaded SDTNBI"][i], @elapsed SDTNBI_threaded(F₀))
        push!(times["denovoNBI"][i], @elapsed denovoNBI(F₀))
    end
end

# Plots
f, ax = subplots()
ax.plot(mean.(times["N"]), mean.(times["SDTNBI"]), color = "tab:blue", label="SDTNBI")
ax.plot(mean.(times["N"]), mean.(times["Threaded SDTNBI"]), color = "tab:orange", label="SDTNBI (12 threads)")
ax.plot(mean.(times["N"]), mean.(times["denovoNBI"]), color = "tab:green", label="denovo NBI")
ax.scatter(times["N"], times["SDTNBI"], color = "tab:blue", alpha=0.3)
ax.scatter(times["N"], times["Threaded SDTNBI"], color = "tab:orange", alpha=0.3)
ax.scatter(times["N"], times["denovoNBI"], color = "tab:green", alpha=0.3)

xscale("log", base=2)
yscale("log", base=2)
grid(linestyle=":", color="k")
ax.legend(title = "Variant")
xlabel("Size")
ylabel("Compute time (seconds)")
xticks([2.0^n for n in -1:12])
yticks([2.0^n for n in -20:5])
xlim([1.0, 2.0^12])
ylim([2.0^-20, 2.0^5])
show()
