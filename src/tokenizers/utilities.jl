
"""
    function count_consecutives(
        ids::AbstractVector{I}; counts::Union{Nothing,Dict{Tuple{I,I},I}}
    ) where {I<:Integer}

Return a dictionary of counts, associating each pair of consecutive integers 
in `ids` to how many times they appear.

Optionally, update the provided `counts` and return it.

# Examples
```jldoctest
julia> using Lilliput

julia> count_consecutives(UInt32[2, 3, 1, 2, 3])
Dict{Tuple{UInt32, UInt32}, UInt32} with 3 entries:
  (0x00000001, 0x00000002) => 0x00000001
  (0x00000003, 0x00000001) => 0x00000001
  (0x00000002, 0x00000003) => 0x00000002
```
"""
function count_consecutives(
        ids::AbstractVector{I}; counts::Union{Nothing,Dict{Tuple{I,I},I}} = nothing
) where {I<:Integer}
    if isnothing(counts)
        counts = Dict{Tuple{I,I},I}()
    end

    for pair in zip(ids, ids[2:end])
        counts[pair] = get(counts, pair, 0) + 1
    end

    return counts
end
