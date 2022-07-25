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
