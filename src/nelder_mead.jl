using Statistics, LinearAlgebra
using IterativeNelderMead

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
- `n_iterations = number of varied parameters`.
"""
function IterativeNelderMeadOptimizer(;options=nothing)
    if isnothing(options)
        options = Dict{String, Any}()
    end
    return IterativeNelderMeadOptimizer(options)
end

function get_options!(options, p0::Parameters)
    get_option!(options, "max_fcalls", 1400 * num_varied(p0))
    get_option!(options, "no_improve_break", 3)
    get_option!(options, "n_iterations", num_varied(p0))
    get_option!(options, "ftol_rel", 1E-6)
    get_option!(options, "uses_parameters", false)
end

function get_option!(options, key, default_value)
    if !haskey(options, key)
        options[key] = default_value
    end
end

# Pn = (P - Pl) / Δ
# P = Pn * Δ + Pl
function normalize_parameters(pars::Parameters)
    parsn = Parameters()
    for par ∈ values(pars)
        parsn[par.name] = normalize_parameter(par)
    end
    return parsn
end

function normalize_parameter(par::Parameter)
    if isfinite(par.lower_bound) && isfinite(par.upper_bound) && par.lower_bound != par.upper_bound
        r = par.upper_bound - par.lower_bound
        return Parameter(name=par.name, value=(par.value - par.lower_bound) / r, lower_bound=0.0, upper_bound=1.0)
    else
        return Parameter(name=par.name, value=par.value, lower_bound=par.lower_bound, upper_bound=par.upper_bound)
    end
end


function denormalize_parameters(parsn, pars::Parameters)
    pars_out = Parameters()
    for (parn, par) ∈ zip(values(parsn), values(pars))
        pars_out[par.name] = denormalize_parameter(parn, par)
    end
    return pars_out
end

function denormalize_parameter(parn, par::Parameter)
    if isfinite(par.lower_bound) && isfinite(par.upper_bound)
        r = par.upper_bound - par.lower_bound
        return Parameter(name=par.name, value=parn.value * r + par.lower_bound, lower_bound=par.lower_bound, upper_bound=par.upper_bound)
    else
        return Parameter(name=par.name, value=parn.value, lower_bound=par.lower_bound, upper_bound=par.upper_bound)
    end
end

"""
    optimize(obj, p0::Vector{<:Real}, optimizer::IterativeNelderMeadOptimizer, [lower_bounds, upper_bounds, vary])
    optimize(obj, p0::Parameters, optimizer::IterativeNelderMeadOptimizer)
Minimize the object function obj with initial parameters p0 using the IterativeNelderMeadOptimizer solver. Lower bounds can also be provided as additional vectors or using the Parameter API.
Returns an NamedTuple with properties:
- pbest::Parameters or Vector, the parameters corresponding to the optimized objective value fbest.
- fbest::Float64, the optimized objective value.
- fcalls::Int, the number of objective calls.
"""
function optimize(obj, p0::Vector{<:Real}, optimizer::IterativeNelderMeadOptimizer, lower_bounds=nothing, upper_bounds=nothing, vary=nothing)
    if !haskey(optimizer.options, "uses_parameters")
        optimizer.options["uses_parameters"] = false
    end
    p0 = Parameters(p0, ["par$i" for i=1:length(p0)], lower_bounds, upper_bounds, vary)
    return optimize(obj, p0, optimizer)
end

function optimize(obj, p0::Parameters, optimizer::IterativeNelderMeadOptimizer)

    # Options
    options = optimizer.options
    if !haskey(options, "uses_parameters")
        options["uses_parameters"] = true
    end
    get_options!(options, p0)

    # Varied parameters
    p0v = Parameters()
    for par ∈ values(p0)
        if !(par.lower_bound == par.upper_bound == par.value)
            p0v[par.name] = par
        end
    end

    # Nothing to optimize if fbest is 
    if length(p0v) == 0
        fbest = obj(p0)
        return (;pbest=p0, fbest=fbest, fcalls=0)
    end

    # Number of iterations
    n_iterations = options["n_iterations"]

    # Subspaces
    subspaces = []
    pnames = collect(keys(p0))
    pnamesv = collect(keys(p0v))
    vi = [i for (i, par) ∈ enumerate(values(p0)) if par.name ∈ pnamesv]
    full_subspace = (;names=pnamesv, index=nothing, indices=vi, indicesv=[1:length(p0v);])
    if length(p0v) > 2
        for i=1:length(p0v)-1
            k1 = vi[i]
            k2 = vi[i+1]
            push!(subspaces, (;names=[pnamesv[i], pnamesv[i+1]], index=i, indicesv=[i, i+1], indices=[k1, k2]))
        end
        k1 = findfirst(pnames .== pnamesv[1])
        k2 = findfirst(pnames .== pnamesv[end])
        push!(subspaces, (;names=[pnamesv[1], pnamesv[end]], index=length(p0v), indicesv=[1, length(p0v)], indices=[k1, k2]))
        k1 = findfirst(pnames .== pnamesv[2])
        k2 = findfirst(pnames .== pnamesv[end-1])
        push!(subspaces, (;names=[pnamesv[2], pnamesv[end-1]], index=length(p0v), indicesv=[2, length(p0v)-1], indices=[k1, k2]))
    end

    # Rescale parameters
    p0n = normalize_parameters(p0)
    p0vn = normalize_parameters(p0v)

    # Initial solution
    pbest = Ref(deepcopy(p0))
    if options["uses_parameters"]
        fbest = Ref(float(obj(p0)))
    else
        x0 = [par.value for par ∈ values(p0)]
        fbest = Ref(float(obj(x0)))
    end

    # Fcalls
    fcalls = Ref(0)

    # Full simplex
    x0vn = [par.value for par ∈ values(p0vn)]
    current_full_simplex = repeat(x0vn, 1, length(x0vn)+1)
    current_full_simplex[:, 1:end-1] .+= diagm(0.5 .* x0vn)
    
    # Loop over iterations
    for iteration=1:n_iterations

        # Perform Ameoba call for all parameters
        optimize_space!(full_subspace, p0, p0v, pbest, fbest, fcalls, options, current_full_simplex, current_full_simplex, obj)
        
        # If there's <= 2 params, a three-simplex is the smallest simplex used and only used once.
        if length(p0v) <= 2
            break
        end
        
        # Perform Ameoba call for subspaces
        for subspace ∈ subspaces
            initial_simplex = get_subspace_simplex(subspace, p0, pbest[])
            optimize_space!(subspace, p0, p0v, pbest, fbest, fcalls, options, current_full_simplex, initial_simplex, obj)
        end
    end
    
    # Output
    out = get_result(options, pbest, fbest, fcalls)

    # Return
    return out

end

function get_subspace_simplex(subspace, p0::Parameters, pbest::Parameters)
    n = length(subspace.names)
    simplex = zeros(n, n+1)
    p0n = normalize_parameters(p0)
    pbestn = normalize_parameters(pbest)
    xbestn = Float64[par.value for par ∈ values(pbestn) if par.name ∈ subspace.names]
    x0n = Float64[par.value for par ∈ values(p0n) if par.name ∈ subspace.names]
    simplex[:, 1] .= x0n
    simplex[:, 2] .= xbestn
    for i=3:n+1
        simplex[:, i] .= xbestn
        j = i - 2
        simplex[j, i] = x0n[j]
    end
    return simplex
end

function optimize_space!(subspace, p0::Parameters, p0v::Parameters, pbest::Ref{Parameters}, fbest::Ref{Float64}, fcalls::Ref{Int}, options, current_full_simplex::Matrix, initial_simplex::Matrix, obj)
    
    # Simplex for this subspace
    simplex = copy(initial_simplex)
    nx, nxp1 = size(simplex)

    # Max f evals
    max_fcalls = options["max_fcalls"]
    ftol_rel = options["ftol_rel"]

    # Initiate storage arrays
    fvals = zeros(nxp1)
    xnp1 = zeros(nx)
    x1 = zeros(nx)
    xn = zeros(nx)
    xr = zeros(nx)
    xbar = zeros(nx)
    xc = zeros(nx)
    xe = zeros(nx)
    xcc = zeros(nx)

    # Best and test parameters, normalized
    pbestn = normalize_parameters(pbest[])
    ptestn = deepcopy(pbestn)
    
    # Generate the fvals for the initial simplex
    for i=1:nxp1
        fvals[i] = @views compute_obj(simplex[:, i], subspace, ptestn, p0, obj, fcalls, options)
    end

    # Sort the fvals and then simplex
    inds = sortperm(fvals)
    simplex .= simplex[:, inds]
    fvals .= fvals[inds]
    fmin = fvals[1]
    
    # Best fit parameter is now the first column
    xmin = simplex[:, 1]
    
    # Keeps track of the number of times the solver thinks it has converged in a row.
    no_improve_break = options["no_improve_break"]
    n_converged = 0

    # Hyper parameters
    α = 1.0
    γ = 2.0
    σ = 0.5
    δ = 0.5
    
    # Loop
    while true

        # Sort the vertices according from best to worst
        # Define the worst and best vertex, and f(best vertex)
        xnp1 .= simplex[:, end]
        fnp1 = fvals[end]
        x1 .= simplex[:, 1]
        f1 = fvals[1]
        xn .= simplex[:, end-1]
        fn = fvals[end-1]
            
        # Checks whether or not to shrink if all other checks "fail"
        shrink = false

        # break after max number function calls is reached.
        if fcalls[] >= max_fcalls
            break
        end
            
        # Break if f tolerance has been met
        if compute_df_rel(fmin, fnp1) > ftol_rel
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
        fr = compute_obj(xr, subspace, ptestn, p0, obj, fcalls, options)

        if fr < f1
            xe .= xbar .+ γ .* (xbar .- xnp1)
            fe = compute_obj(xe, subspace, ptestn, p0, obj, fcalls, options)
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
                fc = compute_obj(xc, subspace, ptestn, p0, obj, fcalls, options)
                if fc <= fr
                    simplex[:, end] .= xc
                    fvals[end] = fc
                else
                    shrink = true
                end
            else
                xcc .= xbar .+ σ .* (xnp1 .- xbar)
                fcc = compute_obj(xcc, subspace, ptestn, p0, obj, fcalls, options)
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
                fvals[j] = @views compute_obj(simplex[:, j], subspace, ptestn, p0, obj, fcalls, options)
            end
        end

        inds = sortperm(fvals)
        fvals = fvals[inds]
        simplex .= simplex[:, inds]
        fmin = fvals[1]
        xmin .= simplex[:, 1]
    end

    inds = sortperm(fvals)
    fvals = fvals[inds]
    simplex .= simplex[:, inds]
    fmin = fvals[1]
    xmin .= simplex[:, 1]
    
    # Update the full simplex and best fit parameters
    for (i, pname) ∈ enumerate(subspace.names)
        pbestn[pname].value = xmin[i]
    end
    if !isnothing(subspace.index)
        current_full_simplex[:, subspace.index] .= [par.value for par ∈ values(pbestn) if par.name ∈ keys(p0v)]
    else
        current_full_simplex .= copy(simplex)
    end

    # Denormalize and store
    pbest[] = denormalize_parameters(pbestn, p0)
    fbest[] = fmin
    nothing
end

function get_result(options::Dict{String, Any}, pbest::Ref{Parameters}, fbest::Ref{Float64}, fcalls::Ref{Int})
    if options["uses_parameters"]
        return (;pbest=pbest[], fbest=fbest[], fcalls=fcalls[])
    else
        pbest_out = Float64[par.value for par ∈ values(pbest[])]
        return (;pbest=pbest_out, fbest=fbest[], fcalls=fcalls[])
    end
end


###################
#### TOLERANCE ####
###################
    
function compute_dx_rel(simplex::AbstractMatrix{Float64})
    a = nanminimum(simplex, dims=2)
    b = nanmaximum(simplex, dims=2)
    c = (abs.(b) .+ abs.(a)) ./ 2
    bad = findall(c .< 0)
    c[bad] = 1
    r = abs.(b .- a) ./ c
    return nanmaximum(r)
end


function compute_df_rel(a, b)
    avg = (abs(a) + abs(b)) / 2
    return abs(a - b) / avg
end



###########################################################################
###########################################################################
###########################################################################


function penalize(f, ptest, names)
    penalty = abs(f) * 10
    for par ∈ values(ptest)
        if par.name ∈ names
            if par.value < par.lower_bound
                f += penalty * (par.lower_bound - par.value)
            end
            if par.value > par.upper_bound
                f += penalty * (par.value - par.upper_bound)
            end
        end
    end
    return f
end

function compute_obj(x::AbstractVector{Float64}, subspace, ptestn, p0, obj, fcalls::Ref{Int}, options)
    fcalls[] += 1
    for i=1:length(subspace.names)
        ptestn[subspace.names[i]].value = x[i]
    end
    ptest = denormalize_parameters(ptestn, p0)
    if options["uses_parameters"]
        f = obj(ptest)
    else
        xtest = [par.value for par ∈ values(ptest)]
        f = obj(xtest)
    end
    f = penalize(f, ptestn, subspace.names)
    if !isfinite(f)
        f = 1E6
    end
    return f
end