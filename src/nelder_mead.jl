using Statistics, LinearAlgebra

export IterativeNelderMeadOptimizer, optimize

struct IterativeNelderMeadOptimizer
    options::Dict{String, Any}
end

"""
    IterativeNelderMeadOptimizer(;options=nothing)
Construct an IterativeNelderMeadOptimizer optimizer. options is of type Dict{String, Any}.
Default options are:
- `max_fcalls = 1400 * number of varied parameters`. The number of objective calls is **not** reset after each iteration / subspace.
- `no_improve_break = 3`. For a given parameter space, the number of times the solver needs to converge in a row to officially be considered converged. This applies to all parameter spaces / iterations.
- `ftol_rel = 1E-6`. For a given parameter space, the relative change in the objective function to be considered converged. This applies to all parameter spaces / iterations.
- `penalty = 10`. The penalty applied when parameters are out of bounds. The penalty is applied as `f_new = f + penalty * abs(bound - x)` in normalized parameter units if both bounds are provided (and therefore bound = 0 or 1), or the original units if only one bound is provided.
- `n_iterations = number of varied parameters`.
"""
function IterativeNelderMeadOptimizer(;options=nothing)
    if isnothing(options)
        options = Dict{String, Any}()
    end
    return IterativeNelderMeadOptimizer(options)
end

function get_default_options!(options, p0, lower_bounds, upper_bounds, vary)
    nv = sum(vary)
    get_option!(options, "max_fcalls", 1400 * nv)
    get_option!(options, "no_improve_break", 3)
    get_option!(options, "n_iterations", nv)
    get_option!(options, "ftol_rel", 1E-6)
    get_option!(options, "penalty", 10)
end

function get_option!(options, key, default_value)
    if !haskey(options, key)
        options[key] = default_value
    end
end

# Pn = (P - Pl) / Δ
# P = Pn * Δ + Pl
function normalize_parameters(values, lower_bounds, upper_bounds, vary)
    vn = zeros(length(values))
    lbn = zeros(length(values))
    ubn = zeros(length(values))
    for i ∈ eachindex(values)
        vn[i], lbn[i], ubn[i] = normalize_parameter(values[i], lower_bounds[i], upper_bounds[i], vary[i])
    end
    return vn, lbn, ubn
end

function normalize_parameter(value, lower_bound, upper_bound, vary)
    if isfinite(lower_bound) && isfinite(upper_bound) && lower_bound != upper_bound && vary
        r = upper_bound - lower_bound
        vn = (value - lower_bound) / r
        return vn, 0.0, 1.0
    else
        return value, lower_bound, upper_bound
    end
end


function denormalize_parameters(valuesn, lower_bounds, upper_bounds, vary)
    v = zeros(length(valuesn))
    for i ∈ eachindex(valuesn)
        v[i] = denormalize_parameter(valuesn[i], lower_bounds[i], upper_bounds[i], vary[i])
    end
    return v
end

function denormalize_parameter(valuen, lower_bound, upper_bound, vary)
    if isfinite(lower_bound) && isfinite(upper_bound) && vary
        r = upper_bound - lower_bound
        v = valuen * r + lower_bound
        return v
    else
        return valuen
    end
end

struct Subspace{T<:Union{Int, Nothing}}
    index::T
    indices::Vector{Int}
    indicesv::Vector{Int}
end

function get_subspaces(vary)
    subspaces = Subspace[]
    vi = findall(vary)
    nv = length(vi)
    full_subspace = Subspace(nothing, vi, [1:nv;])
    if nv > 2
        for i=1:nv-1
            k1 = vi[i]
            k2 = vi[i+1]
            push!(subspaces, Subspace(i, [k1, k2], [i, i+1]))
        end
        k1 = vi[1]
        k2 = vi[end]
        push!(subspaces, Subspace(nv, [k1, k2], [1, nv]))
        if nv > 3
            k1 = vi[2]
            k2 = vi[end-1]
            push!(subspaces, Subspace(nv+1, [k1, k2], [2, nv-1]))
        end
    end
    return full_subspace, subspaces
end

function get_initial_simplex(p0n, lower_boundsn, upper_boundsn, vary)
    indsv = findall(vary)
    p0nv = p0n[indsv]
    nv = length(p0nv)
    simplex = repeat(p0nv, 1, nv+1)
    simplex[:, 1:end-1] .+= diagm(0.5 .* p0nv)
    return simplex
end


"""
    optimize(obj, p0::Vector{Float64}, optimizer::IterativeNelderMeadOptimizer, [lower_bounds, upper_bounds, vary])
Minimize the object function obj with initial parameters p0 using the IterativeNelderMeadOptimizer solver. Lower bounds can also be provided as additional vectors or using the Parameter API.
Returns an NamedTuple with properties:
- pbest::Vector{Float64}, the parameters corresponding to the optimized objective value fbest.
- fbest::Float64, the optimized objective value.
- fcalls::Int, the number of objective calls.
- simplex::Matrix{Float64}, the number of objective calls.
- iteration::Int, the iteration reached, typically equal to n_iterations.
"""
function optimize(obj, p0::Vector{<:Real}, optimizer::IterativeNelderMeadOptimizer; lower_bounds=nothing, upper_bounds=nothing, vary=nothing)

    # Resolve Options
    options = optimizer.options
    n = length(p0)
    if isnothing(vary)
        if isnothing(lower_bounds)
            vary = BitVector(fill(true, n))
        else
            vary = BitVector(fill(true, n))
            for i=1:n
                vary[i] = (lower_bounds[i] !== upper_bounds[i])
            end
        end
    end
    if isnothing(lower_bounds)
        lower_bounds = fill(-Inf, n)
    end
    if isnothing(upper_bounds)
        upper_bounds = fill(Inf, n)
    end
    get_default_options!(options, p0, lower_bounds, upper_bounds, vary)

    # Normalize
    p0n, lower_boundsn, upper_boundsn = normalize_parameters(p0, lower_bounds, upper_bounds, vary)

    # Varied parameters
    vi = findall(vary)
    nv = length(vi)

    # If no parameters to optimize, return
    if nv == 0
        fbest = obj(p0)
        return (;pbest=p0, fbest=fbest, fcalls=0)
    end

    # Number of iterations
    n_iterations = options["n_iterations"]

    # Subspaces
    full_subspace, subspaces = get_subspaces(vary)

    # Initial solution
    pbest = copy(p0)
    fbest = Ref(obj(p0))

    # Fcalls
    fcalls = Ref(0)

    # Full simplex
    full_simplex = get_initial_simplex(p0n, lower_boundsn, upper_boundsn, vary)

    current_iteration = 0
    
    # Loop over iterations
    for iteration=1:n_iterations

        current_iteration += 1

        # Perform Ameoba call for all parameters
        optimize_space!(obj, full_subspace, p0, lower_bounds, upper_bounds, vary, pbest, fbest, fcalls, full_simplex, copy(full_simplex), options)

        # Check x tolerance
        #x_converged = (compute_dx_rel(full_simplex) < options["xtol_rel"]) || (compute_dx_abs(full_simplex) < options["xtol_abs"])
        #if x_converged
        #    break
        #end
        
        # If there's <= 2 params, a three-simplex is the smallest simplex used and only used once.
        if nv <= 2
            break
        end
        
        # Perform Ameoba call for subspaces
        for subspace ∈ subspaces
            pbestn, _, _ = normalize_parameters(pbest, lower_bounds, upper_bounds, vary)
            initial_simplex = get_subspace_simplex(subspace, p0n, pbestn)
            optimize_space!(obj, subspace, p0, lower_bounds, upper_bounds, vary, pbest, fbest, fcalls, full_simplex, initial_simplex, options)
        end
    end
    
    # Output
    out = get_result(options, pbest, fbest, fcalls, full_simplex, current_iteration, lower_bounds, upper_bounds, vary)

    # Return
    return out

end

function get_subspace_simplex(subspace, p0n, pbestn)
    n = length(subspace.indices)
    simplex = zeros(n, n+1)
    simplex[:, 1] .= p0n[subspace.indices]
    simplex[:, 2] .= pbestn[subspace.indices]
    for i=3:n+1
        simplex[:, i] .= pbestn[subspace.indices]
        j = i - 2
        simplex[j, i] = p0n[j]
    end
    return simplex
end

function optimize_space!(obj, subspace::Subspace, p0::Vector, lower_bounds::Vector, upper_bounds::Vector, vary, pbest::Vector, fbest::Ref{Float64}, fcalls::Ref{Int}, current_full_simplex::Matrix, initial_simplex::Matrix, options::Dict{String, Any})
    
    # Simplex for this subspace
    simplex = copy(initial_simplex)
    nx, nxp1 = size(simplex)

    # Max f evals
    max_fcalls = options["max_fcalls"]
    ftol_rel = options["ftol_rel"]

    # Keeps track of the number of times the solver thinks it has converged in a row.
    no_improve_break = options["no_improve_break"]
    n_converged = 0

    # Penalty
    penalty = options["penalty"]

    # Initiate storage arrays
    fvals = zeros(nxp1)
    xr = zeros(nx)
    xbar = zeros(nx)
    xc = zeros(nx)
    xe = zeros(nx)
    xcc = zeros(nx)

    # Test parameters, normalized
    p0n, lower_boundsn, upper_boundsn = normalize_parameters(p0, lower_bounds, upper_bounds, vary)
    ptestn, _, _ = normalize_parameters(pbest, lower_bounds, upper_bounds, vary)
    
    # Generate the fvals for the initial simplex
    for i=1:nxp1
        fvals[i] = @views compute_obj(obj, simplex[:, i], subspace, ptestn, lower_bounds, upper_bounds, lower_boundsn, upper_boundsn, vary, fcalls, penalty)
    end

    # Sort the fvals and then simplex
    inds = sortperm(fvals)
    simplex .= simplex[:, inds]
    fvals .= fvals[inds]
    x1 = simplex[:, 1]
    xn = simplex[:, end-1]
    xnp1 = simplex[:, end]
    f1 = fvals[1]
    fn = fvals[end-1]
    fnp1 = fvals[end]

    # Hyper parameters
    α = 1.0
    γ = 2.0
    σ = 0.5
    δ = 0.5
    
    # Loop
    while true
            
        # Checks whether or not to shrink if all other checks "fail"
        shrink = false

        # break after max number function calls is reached.
        if fcalls[] >= max_fcalls
            break
        end
            
        # Break if f tolerance has been met no_improve_break times in a row
        if compute_df_rel(f1, fnp1) > ftol_rel
            n_converged = 0
        else
            n_converged += 1
        end
        if n_converged >= no_improve_break
            break
        end

        # Idea of NM: Given a sorted simplex; N + 1 Vectors of N parameters,
        # We want to iteratively replace the worst vector with a better vector.
        
        # The "average" vector, ignoring the worst point
        # We first anchor points off this average Vector
        xbar .= @views reshape(mean(simplex[:, 1:end-1], dims=2), (nx, 1))
        
        # The reflection point
        xr .= xbar .+ α .* (xbar .- xnp1)
        
        # Update the current testing parameter with xr
        fr = compute_obj(obj, xr, subspace, ptestn, lower_bounds, upper_bounds, lower_boundsn, upper_boundsn, vary, fcalls, penalty)

        if fr < f1
            xe .= xbar .+ γ .* (xbar .- xnp1)
            fe = compute_obj(obj, xe, subspace, ptestn, lower_bounds, upper_bounds, lower_boundsn, upper_boundsn, vary, fcalls, penalty)
            if fe < fr
                simplex[:, end] .= xe
                fvals[end] = fe
            else
                simplex[:, end] .= xr
                fvals[end] = fr
            end
        elseif fr < fn
            simplex[:, end] .= xr
            fvals[end] = fr
        else
            if fr < fnp1
                xc .= xbar .+ σ .* (xbar .- xnp1)
                fc = compute_obj(obj, xc, subspace, ptestn, lower_bounds, upper_bounds, lower_boundsn, upper_boundsn, vary, fcalls, penalty)
                if fc <= fr
                    simplex[:, end] .= xc
                    fvals[end] = fc
                else
                    shrink = true
                end
            else
                xcc .= xbar .+ σ .* (xnp1 .- xbar)
                fcc = compute_obj(obj, xcc, subspace, ptestn, lower_bounds, upper_bounds, lower_boundsn, upper_boundsn, vary, fcalls, penalty)
                if fcc < fvals[end]
                    simplex[:, end] .= xcc
                    fvals[end] = fcc
                else
                    shrink = true
                end
            end
        end
        if shrink
            for j=2:nxp1
                simplex[:, j] .= @views x1 .+ δ .* (simplex[:, j] .- x1)
                fvals[j] = @views compute_obj(obj, simplex[:, j], subspace, ptestn, lower_bounds, upper_bounds, lower_boundsn, upper_boundsn, vary, fcalls, penalty)
            end
        end

        # Sort
        inds = sortperm(fvals)
        fvals .= fvals[inds]
        simplex .= simplex[:, inds]
        x1 .= simplex[:, 1]
        xn .= simplex[:, end-1]
        xnp1 .= simplex[:, end]
        f1 = fvals[1]
        fn = fvals[end-1]
        fnp1 = fvals[end]
    end


    # Sort
    inds = sortperm(fvals)
    fvals .= fvals[inds]
    simplex .= simplex[:, inds]
    x1 .= simplex[:, 1]
    xn .= simplex[:, end-1]
    xnp1 .= simplex[:, end]
    f1 = fvals[1]
    fn = fvals[end-1]
    fnp1 = fvals[end]
    
    # Update the full simplex and best fit parameters
    pbestn, _, _ = normalize_parameters(pbest, lower_bounds, upper_bounds, vary)
    pbestn[subspace.indices] .= x1
    vi = findall(vary)
    if !isnothing(subspace.index)
        current_full_simplex[:, subspace.index] .= pbestn[vi]
    else
        current_full_simplex .= copy(simplex)
    end

    # Denormalize and store
    pbest .= denormalize_parameters(pbestn, lower_bounds, upper_bounds, vary)
    fbest[] = f1
    nothing
end

function get_result(options::Dict{String, Any}, pbest::Vector, fbest::Ref{Float64}, fcalls::Ref{Int}, simplex::Matrix, iteration::Int, lower_bounds, upper_bounds, vary)
    simplex_out = denormalize_simplex(simplex, pbest, lower_bounds, upper_bounds, vary)
    return (;pbest=pbest, fbest=fbest[], fcalls=fcalls[], simplex=simplex_out, iteration=iteration)
end

function denormalize_simplex(simplex::AbstractMatrix, pars, lower_bounds, upper_bounds, vary)
    vi = findall(vary)
    ptempn = copy(pars)
    ptemp = copy(pars)
    simplex_out = zeros(size(simplex))
    for i=1:length(vi)+1
        ptempn[vi] .= simplex[:, i]
        ptemp = denormalize_parameters(ptempn, lower_bounds, upper_bounds, vary)
        simplex_out[:, i] .= ptemp[vi]
    end
    return simplex_out
end


###################
#### TOLERANCE ####
###################
    
function compute_dx_rel(simplex::AbstractMatrix{Float64})
    a = minimum(simplex, dims=2)[:]
    b = maximum(simplex, dims=2)[:]
    c = (abs.(b) .+ abs.(a)) ./ 2
    bad = findall(c .< 0)
    c[bad] .= 1
    r = abs.(b .- a) ./ c
    return maximum(r)
end

function compute_dx_abs(simplex::AbstractMatrix{Float64})
    a = minimum(simplex, dims=2)[:]
    b = maximum(simplex, dims=2)[:]
    r = abs.(b .- a)
    return maximum(r)
end

function compute_df_rel(a, b)
    avg = (abs(a) + abs(b)) / 2
    return abs(a - b) / avg
end

function compute_df_abs(a, b)
    return abs(a - b)
end



###########################################################################
###########################################################################
###########################################################################


function penalize(f, ptestn, indices, lower_boundsn, upper_boundsn, penalty)
    for i=1:length(indices)
        j = indices[i]
        if ptestn[j] < lower_boundsn[j]
            f += penalty * (lower_boundsn[j] - ptestn[j])
        end
        if ptestn[j] > upper_boundsn[j]
            f += penalty * (ptestn[j] - upper_boundsn[j])
        end
    end
    return f
end

function compute_obj(obj, x::AbstractVector{Float64}, subspace::Subspace, ptestn::Vector, lower_bounds::Vector, upper_bounds::Vector, lower_boundsn::Vector, upper_boundsn::Vector, vary::AbstractVector, fcalls::Ref{Int}, penalty::Real)
    fcalls[] += 1
    ptestn[subspace.indices] .= x
    ptest = denormalize_parameters(ptestn, lower_bounds, upper_bounds, vary)
    f = obj(ptest)
    f = penalize(f, ptestn, subspace.indices, lower_boundsn, upper_boundsn, penalty)
    if !isfinite(f)
        f = 1E6
    end
    return f
end