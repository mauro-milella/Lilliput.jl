
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

    while i <= indexes_length - 1
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

# this could leverage Base.Unicode.iscntrl internally
function _replace_control_characters(s::AbstractString)
    # https://github.com/JuliaLang/julia/blob/master/base/strings/unicode.jl
    # http://www.unicode.org/reports/tr44/#GC_Values_Table
    # especially Cc ("control characters") are subtle when writing them
    other_categories = ["Cc", "Cf", "Cs", "Co", "Cn"]

    # stores both chars and strings (for unicodes)
    chars = []

    for c in s
        if !(Base.Unicode.category_abbrev(c) in other_categories)
            push!(chars, c)
        else
            # unicode escape, such as "\\u000a" for c = '\n'
            push!(chars, "\\u" * string(UInt32(c), base=16, pad=4))
        end
    end

    return join(chars)
end

"""
Check whether the given bytes are a valid UTF-8 string and return them.

If it is not the case, return the replacement Unicode character `\uFFFD` (�).

See `Base.isvalid`,
[replacement character](https://en.wikipedia.org/wiki/Specials_(Unicode_block)).
"""
function _utf8_or_replacementchar!(bytes::AbstractVector{UInt8})
    io = IOBuffer()
    i = firstindex(bytes)
    n = lastindex(bytes)

    while i <= n
        b = bytes[i]

        # ASCII check; of course, 0x80 is UInt8(128)
        if b < 0x80
            write(io, Char(b))
            i += 1
            continue
        end

        # to read UTF-8, we need to check the sequence length.
        # 0******* -> 1 byte sequence (ASCII, covered above)
        # 110***** -> 2 bytes 
        # 1110**** -> 3 bytes
        # 11110*** -> 4 bytes
        # 10****** -> continuation of a 2/3/4 bytes sequence
        # https://en.wikipedia.org/wiki/UTF-8
        len =
            # 0xe0 is exactly 1<<7 + 1<<6 + 1<<5, that is, 11100000
            # we use this mask over b, and see if the result is 11000000
            # (or 192, which is in fact 1<<7 + 1<<6 and no 1<<5)
            b & 0xe0 == 0xc0 ? 2 :
            # same reasoning as above
            b & 0xf0 == 0xe0 ? 3 :
            b & 0xf8 == 0xf0 ? 4 : 0

        # the default "error" scenario above is 0, so this is invalid
        if len == 0
            write(io, '\uFFFD')
            i += 1
            continue
        end

        # the sequence is tuncated, because `len` tells that we have to 
        # overpass the boundary n 
        if i + len - 1 > n
            write(io, '\uFFFD')
            break
        end

        # these are the useful bits from the first byte;
        # we consider UInt32 because we can potentially encode 4 bytes
        code = UInt32(b & (0x7f >> len))
        valid = true

        for j in 2:len
            c = bytes[i + j - 1]
            # 10****** is the continuation character
            if c & 0xc0 != 0x80
                valid = false
                break
            end
            # keep the low 6 bits with 0x3f and shift them to make room for more
            code = (code << 6) | UInt32(c & 0x3f)
        end

        valid &= (
            # if the length is 2 and you are using more less than 2 bytes, bad
            (len == 2 && code >= 0x80) ||
            (len == 3 && code >= 0x800) ||
            (len == 4 && code >= 0x10000)
        )
        # anything bigger than this is not a valid Unicode
        # see the Error handling section here
        # https://en.wikipedia.org/wiki/UTF-8
        valid &= code <= 0x10ffff

        # see the Surrogates section below; these are not legal Unicode values
        # https://en.wikipedia.org/wiki/UTF-8
        valid &= !(0xd800 <= code <= 0xdfff)

        if valid
            write(io, Char(code))
        else
            write(io, '\uFFFD')
        end

        # Consume the whole sequence, valid or not
        i += len
    end

    return String(take!(io))
end

"""
    function render(_token::Vector{UInt8})

Pretty print the bytes in render.
Bytes are converted in na string, where Unicode sequences that are dangerous
to print (e.g., '\n') are escaped.
"""
function render(_tokens::AbstractVector{UInt8})
    return _replace_control_characters(_utf8_or_replacementchar!(_tokens))
end
