module NetworkBasedInference

using LinearAlgebra
using NamedArrays

export NBI,
    NWNBI,
    EWNBI,
    denovoNBI,
    simmat
"""
    NBI(F₀::AbstractMatrix{Float64})

Calculate the weighted preyaction for the row nodes of a rectangular adyacency matrix using
a resource-spreading algorithm.

# Example
```jldoctest
julia> A = rand(3,7) .> 0.5
3×7 BitMatrix:
 0  1  0  1  1  1  0
 0  1  1  0  0  0  0
 1  1  0  1  0  1  1

julia> round.(NBI(A); digits = 2)
7×7 Matrix{Float64}:
 0.2   0.2   0.0   0.2   0.0   0.2   0.2
 0.07  0.32  0.17  0.15  0.08  0.15  0.07
 0.0   0.5   0.5   0.0   0.0   0.0   0.0
 0.1   0.22  0.0   0.22  0.12  0.22  0.1
 0.0   0.25  0.0   0.25  0.25  0.25  0.0
 0.1   0.22  0.0   0.22  0.12  0.22  0.1
 0.2   0.2   0.0   0.2   0.0   0.2   0.2

julia> round.(A * NBI(A); digits = 2) # Link prediction
3×7 Matrix{Float64}:
 0.27  1.02  0.17  0.85  0.58  0.85  0.27
 0.07  0.82  0.67  0.15  0.08  0.15  0.07
 0.67  1.17  0.17  1.0   0.33  1.0   0.67
```

# Arguments
- `F₀::AbstractMatrix{Float64}`: Rectangular adjacency matrix

# Extended help
The final resource matrix obtained from product between `F₀` and `NBI(F₀)` corresponds to
the a link prediction / edge weighting for the bipartite network.

# References
1. Zhou, et al (2007). Bipartite network projection and personal recommendation.
   Physical Review E, 76(4). https://doi.org/10.1103/physreve.76.046115
2. Cheng, et al (2012). Prediction of Drug-Target Interactions and Drug Repositioning via
   Network-Based Inference. PLoS Computational Biology, 8(5), e1002503.
   https://doi.org/10.1371/journal.pcbi.1002503
3. Cheng, et al (2012). Prediction of Chemical-Protein Interactions Network with Weighted
   Network-Based Inference Method. PLoS ONE, 7(7), e41064.
   https://doi.org/10.1371/journal.pone.0041064
"""
function NBI(F₀::AbstractMatrix{Float64})
    # Resource matrices
    R = diagm(sum.(eachrow(F₀)))
    H = diagm(sum.(eachcol(F₀)))

    # Transfer matrix
    W = (F₀ * H^-1)' * (R^-1 * F₀)

    return W
end

NBI(F₀::AbstractMatrix{Bool}) = NBI(AbstractMatrix{Float64}(F₀))


"""
    NWNBI(F₀::AbstractMatrix{T}, β::T) where T<:Float64

Calculate NBI proyection taking into consideration that hub node with more resources are
more difficult to be influenced.

# Example
```jldoctest
julia> A = rand(3,7) .> 0.5
3×7 BitMatrix:
 0  1  0  1  1  1  0
 0  1  1  0  0  0  0
 1  1  0  1  0  1  1

julia> round.(NWNBI(A, 1.0); digits = 2) # Stronger hub nodes
7×7 Matrix{Float64}:
 0.2   0.07  0.0   0.1   0.0   0.1   0.2
 0.07  0.11  0.17  0.08  0.08  0.08  0.07
 0.0   0.17  0.5   0.0   0.0   0.0   0.0
 0.1   0.08  0.0   0.11  0.12  0.11  0.1
 0.0   0.08  0.0   0.12  0.25  0.12  0.0
 0.1   0.08  0.0   0.11  0.12  0.11  0.1
 0.2   0.07  0.0   0.1   0.0   0.1   0.2

julia> round.(NWNBI(A, 0.0); digits = 2) # Uniform case, equivalent to NBI
7×7 Matrix{Float64}:
 0.2   0.2   0.0   0.2   0.0   0.2   0.2
 0.07  0.32  0.17  0.15  0.08  0.15  0.07
 0.0   0.5   0.5   0.0   0.0   0.0   0.0
 0.1   0.22  0.0   0.22  0.12  0.22  0.1
 0.0   0.25  0.0   0.25  0.25  0.25  0.0
 0.1   0.22  0.0   0.22  0.12  0.22  0.1
 0.2   0.2   0.0   0.2   0.0   0.2   0.2

julia> round.(NWNBI(A, -1.0); digits = 2) # Weaker hub nodes
7×7 Matrix{Float64}:
 0.2   0.6   0.0   0.4   0.0   0.4   0.2
 0.07  0.95  0.17  0.3   0.08  0.3   0.07
 0.0   1.5   0.5   0.0   0.0   0.0   0.0
 0.1   0.68  0.0   0.45  0.12  0.45  0.1
 0.0   0.75  0.0   0.5   0.25  0.5   0.0
 0.1   0.68  0.0   0.45  0.12  0.45  0.1
 0.2   0.6   0.0   0.4   0.0   0.4   0.2
```

# Arguments
- `F₀::AbstractMatrix{Float64}`: Rectangular adjacency matrix
- `β::Float64`: Hub influence parameter

# Extended help
Some special cases exist for some β values:
- `β > 0` strengthens the influence of hub nodes
- `β = 0` corresponds to the uniform case, i.e., equivalent to NBI
- `β < 0` weakens the influence of hub nodes

The final resource matrix obtained from product between `F₀` and `NWNBI(F₀)` corresponds to 
the a link prediction / edge weighting for the bipartite network.

# References
1. Cheng, et al (2012). Prediction of Chemical-Protein Interactions Network with Weighted
   Network-Based Inference Method. PLoS ONE, 7(7), e41064.
   https://doi.org/10.1371/journal.pone.0041064
"""
function NWNBI(F₀::AbstractMatrix{T}, β::T) where {T<:Float64}
    # Resource matrices
    R = diagm(sum.(eachrow(F₀)))
    H = diagm(sum.(eachcol(F₀)))

    # Weighting matrix
    H′ = diagm(sum.(eachcol(F₀)) .^ β)

    # Transfer matrix
    W = (F₀ * H^-1)' * (R^-1 * F₀ * H′^-1)

    return W
end

NWNBI(F₀::AbstractMatrix{Bool}, β::Float64) = NWNBI(AbstractMatrix{Float64}(F₀), β)

"""
    EWNBI(F₀::AbstractMatrix{T}, λ::T) where T<:Float64

Calculate NBI proyection taking into consideration that edge weights correlate with the
importance of an edge.

# Example
```jldoctest
julia> A = cutoff.(rand(3,7), 0.5)
3×7 Matrix{Float64}:
 0.0       0.0       0.0       0.723091  0.667418  0.500017  0.830616
 0.771808  0.857922  0.578597  0.862383  0.0       0.0       0.0
 0.0       0.545525  0.516228  0.0       0.555673  0.545182  0.939333

julia> round.(EWNBI(A, 1.0); digits = 2)
7×7 Matrix{Float64}:
 0.25  0.28  0.19  0.28  0.0   0.0   0.0
 0.15  0.24  0.18  0.17  0.07  0.07  0.12
 0.13  0.23  0.18  0.15  0.08  0.08  0.14
 0.14  0.15  0.1   0.27  0.11  0.08  0.14
 0.0   0.08  0.08  0.15  0.22  0.18  0.3
 0.0   0.09  0.09  0.13  0.21  0.18  0.3
 0.0   0.09  0.09  0.12  0.21  0.18  0.3

julia> round.(EWNBI(A, 0.5); digits = 2)
7×7 Matrix{Float64}:
 0.25  0.27  0.22  0.27  0.0   0.0   0.0
 0.14  0.23  0.2   0.15  0.08  0.08  0.11
 0.13  0.23  0.2   0.14  0.09  0.09  0.12
 0.13  0.14  0.11  0.26  0.12  0.1   0.13
 0.0   0.09  0.09  0.14  0.22  0.2   0.26
 0.0   0.1   0.09  0.13  0.22  0.2   0.26
 0.0   0.1   0.09  0.13  0.22  0.2   0.26

julia> round.(EWNBI(A, 0.0); digits = 2)
7×7 Matrix{Float64}:
 0.25  0.25  0.25  0.25  0.0   0.0   0.0
 0.12  0.22  0.22  0.12  0.1   0.1   0.1
 0.12  0.22  0.22  0.12  0.1   0.1   0.1
 0.12  0.12  0.12  0.25  0.12  0.12  0.12
 0.0   0.1   0.1   0.12  0.22  0.22  0.22
 0.0   0.1   0.1   0.12  0.22  0.22  0.22
 0.0   0.1   0.1   0.12  0.22  0.22  0.22

julia> round.(EWNBI(A, -1.0); digits = 2)
7×7 Matrix{Float64}:
 0.24  0.22  0.32  0.22  0.0   0.0   0.0
 0.09  0.22  0.27  0.08  0.13  0.13  0.08
 0.11  0.22  0.27  0.1   0.11  0.11  0.07
 0.11  0.1   0.15  0.22  0.13  0.18  0.11
 0.0   0.12  0.12  0.1   0.23  0.27  0.16
 0.0   0.1   0.11  0.12  0.23  0.28  0.16
 0.0   0.1   0.11  0.12  0.23  0.28  0.16
```

# Arguments
- `F₀::AbstractMatrix{Float64}`: Rectangular adjacency matrix
- `λ::Float64`: Edge weighting parameter

# Extended help
Some special cases exist for some λ values:
- `λ = 1` corresponds to the weighted case, i.e., equivalent to EWNBI
- `0 < λ < 1` strengthens weak-links weights
- `λ = 0` corresponds to the unweighted case, i.e., equivalent to NBI
- `λ < 0` inverts the edge weighting

The final resource matrix obtained from product between `F₀` and `EWNBI(F₀)` corresponds to
the a link prediction / edge weighting for the bipartite network.

# References
1. Cheng, et al (2012). Prediction of Chemical-Protein Interactions Network with Weighted
   Network-Based Inference Method. PLoS ONE, 7(7), e41064.
   https://doi.org/10.1371/journal.pone.0041064
"""
function EWNBI(F₀::AbstractMatrix{T}, λ::T) where {T<:Float64}
    # Edge weighting procedure
    F₀′ = copy(F₀)
    F₀′[findall(!iszero, F₀′)] .^= λ
    replace!(F₀′, Inf => 0.0)
    replace!(F₀′, NaN => 0.0)

    # Weighting matrix
    W = NBI(F₀′)
    return W
end

"""
    denovoNBI(F₀::AbstractMatrix{Float64})

# Arguments
- `F₀::AbstractMatrix{Float64}`: Square adjacency matrix

# Extended help

# References
1. Wu, et al (2016). SDTNBI: an integrated network and chemoinformatics tool for systematic
   prediction of drug–target interactions and drug repositioning. Briefings in
   Bioinformatics, bbw012. https://doi.org/10.1093/bib/bbw012

"""
function denovoNBI(F₀::AbstractMatrix{Float64})
    # Degree helper function
    _k(G::AbstractMatrix) = mapslices(e -> count(!iszero, e), G; dims = 2)

    # Transfer matrix
    W = F₀ ./ _k(F₀)
    replace!(W, Inf => 0.0)
    replace!(W, NaN => 0.0)

    return W
end

denovoNBI(F₀::AbstractMatrix{Bool}) = denovoNBI(AbstractMatrix{Float64}(F₀))
function denovoNBI(namedF₀::NamedMatrix)
    namedW = copy(namedF₀)
    namedW.array = denovoNBI(Matrix{Float64}(namedF₀.array))
end

end
