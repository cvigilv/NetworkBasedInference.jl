module NetworkBasedInference
	
	using LinearAlgebra
	
	export NBI, 
			NWNBI, 
			SDTNBI

	"""
	
	"""
	function NBI(F₀::Matrix)::Matrix
		M,N = size(F₀)
		R = diagm([ sum(F₀[i,:]) for i in 1:M ])
		H = diagm([ sum(F₀[:,j]) for j in 1:N ])
		W = (F₀ * H^-1)' * (R^-1 * F₀)

		return W
    end

	NBI(SymF₀::Symmetric, Nd::Int)::Matrix = NBI(SymF₀[1:Nd, Nd+1:end])


	"""

	"""
	function NWNBI(F₀::Matrix, β::AbstractFloat)::Matrix
		M,N = size(F₀)
        R  = diagm([ sum(F₀[i,:])   for i in 1:M ])
        H  = diagm([ sum(F₀[:,j])   for j in 1:N ])
        H´ = diagm([ sum(F₀[:,j])^β for j in 1:N ])
        W´ = (F₀ * H^-1)' * (R^-1 * F₀ * H´^-1)

        return W´
    end

	NWNBI(SymF₀::Symmetric, Nd::Int, β::AbstractFloat) = NWNBI(SymF₀[1:Nd, Nd+1:end], β)

	"""

	"""
	function SDTNBI(F₀::Symmetric)::Matrix
		k(A, x) = count(r->(r != 0), A[x,:])	# Degree helper function
		n_nodes = size(F₀, 1)					# Number of nodes	
		W	    = zeros(n_nodes, n_nodes)		# Transfer matrix

		for idx in 1:n_nodes
			W[idx,:] = k(F₀, idx) > 0 ? F₀[idx,:] ./ k(F₀, idx) : zeros(n_nodes, 1)
		end

		return W
	end
end
