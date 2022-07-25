module NetworkBasedInference

using LinearAlgebra
using NamedArrays

export NBI,
    NWNBI,
    EWNBI,
    denovoNBI,
    simmat
function NBI(F₀::AbstractMatrix{Float64})
    # Resource matrices
    R = diagm(sum.(eachrow(F₀)))
    H = diagm(sum.(eachcol(F₀)))

    # Transfer matrix
    W = (F₀ * H^-1)' * (R^-1 * F₀)

    return W
end

NBI(F₀::AbstractMatrix{Bool}) = NBI(AbstractMatrix{Float64}(F₀))
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
