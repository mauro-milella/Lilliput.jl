"""
    struct BPETokenizer{I<:Integer} <: AbstractTokenizer
        vocabulary_data::Vector{UInt8}
        vocabulary_offsets::Vector{Int}
        vocabulary_bytelengths::Vector{Int}

        merges::Dict{Tuple{I,I}, I}

        special_tokens::Vector{I}
    end

Minimal implementation of [`AbstractTokenizer`](@ref)'s interface.
Leverages the byte-pair encoding implementation of GPT-2.
"""
struct BPETokenizer{I<:Integer} <: AbstractTokenizer
    # flattened vocabulary, where each token is the following bytes sequence:
    # view(vocabulary_data, offsets[i] : offsets[i] + bytelengths[i] - 1)
    vocabulary_data::Vector{UInt8}
    vocabulary_offsets::Vector{Int}
    vocabulary_bytelengths::Vector{Int}

    # two token ids i,j, are merged in merges[(i,j,)]
    merges_data::Dict{Tuple{I,I},I}

    special_tokens::Vector{I}

    function BPETokenizer(args...; kwargs...)
        return BPETokenizer{UInt16}(args...; kwargs...)
    end

    function BPETokenizer{I}(; alphabet_size=255) where {I<:Integer}
        vocabulary_data = [UInt8(i) for i in 1:alphabet_size]
        vocabulary_offsets = [Int(i) for i in 1:alphabet_size]
        vocabulary_bytelengths = ones(Int, alphabet_size)

        merges = Dict{Tuple{I,I},I}()

        special_tokens = I[]

        return new{I}(
            vocabulary_data,
            vocabulary_offsets,
            vocabulary_bytelengths,
            merges,
            special_tokens,
        )
    end
end

data(bpet::BPETokenizer) = bpet.vocabulary_data
offsets(bpet::BPETokenizer) = bpet.vocabulary_offsets
bytelengths(bpet::BPETokenizer) = bpet.vocabulary_bytelengths
merges(bpet::BPETokenizer) = bpet.merges_data
special_tokens(bpet::BPETokenizer) = bpet.special_tokens

"""
    function train(
        bpetok::BPETokenizer{I}, vocabulary_size::Int, documents::Vector{String}
    ) where {I<:Integer}

Learn/train/build a vocabulary from the given documents, leveraging a BPE
implementation similar to the one of GPT-2.

See also:
- [gpt-2](https://github.com/openai/gpt-2/blob/master/src/encoder.py);
- [Karphaty bpe](https://github.com/karpathy/minbpe/blob/master/minbpe/basic.py).

# Examples
```jldoctest
julia> using Lilliput

julia> bpetok = BPETokenizer(); 

julia> train(bpetok, 270, ["Hello, world!", "Hello hello!"])

julia> Char.(token(bpetok, UInt16(256)))
2-element Vector{Char}:
 'l': ASCII/Unicode U+006C (category Ll: Letter, lowercase)
 'l': ASCII/Unicode U+006C (category Ll: Letter, lowercase)
```
"""
function train(
    bpetok::BPETokenizer{I}, vocabulary_size::Int, documents::Vector{String}
) where {I<:Integer}
    vocabulary_data = data(bpetok)
    vocabulary_data_length = length(vocabulary_data)
    @assert vocabulary_size >= vocabulary_data_length "No space available"

    merges_data = merges(bpetok)
    num_merges = vocabulary_size - vocabulary_data_length

    # these are token IDs; now, they seem just bytes, byt are combined later 
    # in heavier integers. You want UInt16 here, probably UInt32.
    indexes = I[]
    for doc in documents
        append!(indexes, I.(codeunits(doc)))
    end

    for _ in 1:num_merges
        counts = count_consecutives(indexes)

        # trying to call "findmax" later triggers an ArgumentError; just leave
        if isempty(counts)
            break
        end
        most_freqpair = findmax(counts)[2] # 2 returns a Tuple{I, I}

        # token ID for the new token
        new_id = I(vocabulary_lastindex(bpetok) + 1)

        # replace all the occurrences of the most frequent pair with the new ID
        indexes = merge(indexes, most_freqpair, new_id)

        merges_data[most_freqpair] = new_id

        add_token!(
            bpetok,
            vcat(
                token(bpetok, most_freqpair[1]),
                token(bpetok, most_freqpair[2]),
            ),
        )
    end

    return bpetok
end


"""
    function encode(bpetok::BPETokenizer{I}, document::String) where {I}

Encode a document into a list of token IDS.

# Examples
```jldoctest
julia> using Lilliput

julia> bpetok = BPETokenizer(); 

julia> train(bpetok, 270, ["Hello, world!", "Hello hello!"]);

julia> encoding = encode(bpetok, "Hello, ")
2-element Vector{UInt16}:
 0x0103
 0x0107

julia> decode(bpetok, encoding)
"Hello, "
```
"""
function encode(bpetok::BPETokenizer{I}, document::String) where {I}
    indexes = I.(codeunits(document))
    merges_data = merges(bpetok)
    
    while length(indexes) >= 2
        # find the pair with the lowest count
        counts = count_consecutives(indexes)

        pair = argmin(p -> get(merges_data, p, Inf), keys(counts))

        # edge case: there is only one candidate pair for merging, probably
        # with an Inf weight (see argmin above), but the pair does not exist
        if !haskey(merges_data, pair)
            break
        end

        # shrink the token IDs
        new_index = merges_data[pair]
        indexes = merge(indexes, pair, new_index)
    end

    return indexes
end

"""
    function decode(bpetok::BPETokenizer{I}, indexes::Vector{I}) where {I}

Decode a list of token IDS.

# Examples
```jldoctest
julia> using Lilliput

julia> bpetok = BPETokenizer(); 

julia> train(bpetok, 270, ["Hello, world!", "Hello hello!"]);

julia> decode(bpetok, UInt16[259, 32, 256])
"Hello ll"
```
"""
function decode(bpetok::BPETokenizer{I}, indexes::Vector{I}) where {I<:Integer}
    bytes = UInt8[]
    for index in indexes
        append!(bytes, token(bpetok, index))
    end
    return String(bytes)
end

