import DataStructures: OrderedDict

import DataFrames: DataFrame

export Parameter, Parameters, num_varied, is_varied

mutable struct Parameter
    name::Union{String, Nothing}
    value::Float64
    lower_bound::Float64
    upper_bound::Float64
    vary::Bool
    latex_str::Union{String, Nothing}
end

struct Parameters
    dict::OrderedDict{String, Parameter}
end

"""Construct a new parameter"""
function Parameter(;name=nothing, value::Real, lower_bound::Real=-Inf, upper_bound::Real=Inf, vary=true, latex_str=nothing)
    if lower_bound == upper_bound
        vary = false
    end
    return Parameter(name, value, lower_bound, upper_bound, vary, latex_str)
end

"""Construct an empty Parameters struct"""
Parameters() = Parameters(OrderedDict{String, Parameter}())

Base.length(pars::Parameters) = length(pars.dict)
Base.merge!(pars::Parameters, pars2::Parameters) = merge!(pars.dict, pars2.dict)
Base.getindex(pars::Parameters, key::String) = getindex(pars.dict, key)
Base.firstindex(pars::Parameters) = firstindex(pars.dict)
Base.lastindex(pars::Parameters) = lastindex(pars.dict)
Base.iterate(pars::Parameters) = iterate(pars.dict)
Base.keys(pars::Parameters) = keys(pars.dict)
Base.values(pars::Parameters) = values(pars.dict)

function set_name!(par::Parameter, name::String)
    if isnothing(par.name)
        par.name = name
    end
    if isnothing(par.latex_str)
        par.latex_str = name
    end
end

function Base.setindex!(pars::Parameters, par::Parameter, key::String)
    set_name!(par, key)
    setindex!(pars.dict, par, key)
end

function Base.show(io::IO, par::Parameter)
    println(io, " $(par.name) | Value = $(par.value) | Bounds = [$(par.lower_bound), $(par.upper_bound)]")
end

function Base.show(io::IO, pars::Parameters)
    for par ∈ values(pars)
        show(io, par)
    end
end

function num_varied(pars::Parameters)
    n = 0
    for par ∈ values(pars)
        if is_varied(par)
            n += 1
        end
    end
    return n
end

function is_varied(par::Parameter)
    return (par.lower_bound == par.upper_bound) || par.vary
end

function to_df(pars::Parameters)
    df = DataFrame()
    df.names = Float64[par.name for par ∈ values(pars)]
    df.values = Float64[par.value for par ∈ values(pars)]
    df.lower_bounds = Float64[par.lower_bound for par ∈ values(pars)]
    df.upper_bounds = Float64[par.upper_bound for par ∈ values(pars)]
    df.vary = BitVector([is_varied(par) for par ∈ values(pars)])
    df.latex_str = String[par.latex_str for par ∈ values(pars)]
    return df
end

function Parameters(x::AbstractVector{<:Real}, names::AbstractVector{<:AbstractString}, lower_bounds=nothing, upper_bounds=nothing, vary=nothing)
    pars = Parameters()
    nx = length(x)
    for (i, name) ∈ enumerate(names)
        if isnothing(vary)
            _vary = true
        end
        if isnothing(lower_bounds)
            lb = -Inf
        end
        if isnothing(upper_bounds)
            ub = Inf
        end
        pars[name] = Parameter(name=name, value=x[i], lower_bound=lb, upper_bound=ub, vary=_vary)
    end
    return pars
end