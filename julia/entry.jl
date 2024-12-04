include("solve.jl")
import Base.iterate, Base.length

struct Combinations{T}
    itr::Vector{T}
    count::Int64
    itrsize::Int64
    function Combinations(itr::Vector{T}, count::Int) where {T}
        new{T}(itr, Int64(count), length(itr))
    end
end

function iterate(c::Combinations, state::Int64=0)
    if state >= length(c)
        return nothing
    end
    indices = digits(state, base=c.itrsize, pad=c.count)
    [c.itr[i] for i in (indices .+ 1)], state + 1
end

function length(c::Combinations)
    length(c.itr)^c.count
end

const stringer_positions = collect(range(0.0, 8.0, step=(1 / 8)))[1:end-1]

function position_combos()
    collect(Combinations(stringer_positions, 2))
end

@time position_combos()