"""SDTNBI"""
function prepare(L₁::T, L₂::T, L₃::T)::T where T<:AbstractMatrix
end

"""SDTNBI for novel compounds"""
function prepare(L₁::T, L₂::Tuple{T,T}, L₃::T)::T where T<:AbstractMatrix
end

"""DDTNBI"""
function prepare(L₁::T, L₂::T, L₃::T, cutoffs::Tuple{AbstractFloat,AbstractFloat})::T where T<:AbstractMatrix
end

"""DDTNBI for novel compounds """
function prepare(L₁::T, L₂::Tuple{T,T}, L₃::T, cutoffs::Tuple{AbstractFloat,AbstractFloat})::T where T<:AbstractMatrix
end

"""NBI"""
function prepare(L₁, L₂)::AbstractMatrix
end
