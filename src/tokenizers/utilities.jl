
"""
    function count_consecutives(
        indexes::AbstractVector{I}; counts::Union{Nothing,Dict{Tuple{I,I},I}}
    ) where {I<:Integer}

Return a dictionary of counts, associating each pair of consecutive integers 
in `indexes` to how many times they appear.

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
    indexes::AbstractVector{I};
    counts::Union{Nothing,Dict{Tuple{I,I},I}}=nothing,
) where {I<:Integer}
    if isnothing(counts)
        counts = Dict{Tuple{I,I},I}()
    end

    for pair in zip(indexes, indexes[2:end])
        counts[pair] = get(counts, pair, 0) + 1
    end

    return counts
end

"""
    function merge(
        indexes::AbstractVector{I}, pair::Tuple{I,I}, new_index::I
    ) where {I<:Integer}

Replace all the occurences of `pair` in `indexes` with `index`.

```jldoctest
julia> using Lilliput

julia> merge([2, 3, 1, 2, 3], (2, 3), 4)
3-element Vector{Int64}:
 4
 1
 4
"""
function merge(
    indexes::AbstractVector{I}, pair::Tuple{I,I}, new_index::I
) where {I<:Integer}
    indexes_length = length(indexes)
    p0, p1 = pair

    new_indexes = zeros(I, indexes_length)

    i = 1 # main iterator on indexes; jumps when a pair matches
    j = 1 # keeps track of the last available index in new_indexes

    while i <= indexes_length-1
        if indexes[i] == p0 && indexes[i + 1] == p1
            new_indexes[j] = new_index
            i += 2
        else
            new_indexes[j] = indexes[i]
            i += 1
        end
        j += 1
    end

    if i <= indexes_length
        new_indexes[j] = indexes[i]
        j += 1
    end

    return new_indexes[1:(j - 1)]
end

"""
    function render(_token::Vector{UInt8})

Pretty print the bytes in render.
This is a naive and unsafe implementation, substituting newlines with a string
`<|newline|>` and tabs with `<|tab|>`.

We should wrap every dangerous symbol in a safe control sequence.
"""
function render(_tokens::Vector{UInt8})
    replace(String(_tokens), "\n" => "<|newline|>", "\t" => "<|tab|>")
end
