"""
    abstract type AbstractTokenizer end

Generic tokenizer.

Must implement the following:

- tokenizer(::AbstractTokenizer)
- data(::AbstractTokenizer)
- offsets(::AbstractTokenizer)
- bytelengths(::AbstractTokenizer)
- merges(::AbstractTokenizer)
- special_tokens(::AbstractTokenizer)
- train(::AbstractTokenizer, ::Int, ::Vector{String})
- encode(::AbstractTokenizer, ::String)
- decode(::AbstractTokenizer, Vector{<:Integer})

Offers the following minimal utilities:

- vocabulary_size(::AbstractTokenizer)
- token(::AbstractTokenizer, i::I) where {I<:Integer}
- add_token!(::AbstractTokenizer, bytes::Vector{UInt8})
"""
abstract type AbstractTokenizer end

"""
    function train(
        abstract_tokenizer::AbstractTokenizer,
        vocabulary_size::Int,
        documents::Vector{String},
    )

Train a vocabulary of size `vocabulary_size` from the given `text`.
"""
function train(
    abstract_tokenizer::AbstractTokenizer,
    vocabulary_size::Int,
    documents::Vector{String},
)
    throw(MethodError(train, (abstract_tokenizer, vocabulary_size, documents)))
end

"""
    function encode(at::AbstractTokenizer, text::String)
        throw(MethodError(encode, (at, text,)))
    end

Encode the given `text` as a list of token IDs (integers).
"""
function encode(at::AbstractTokenizer, text::String)::Vector{<:Integer}
    throw(MethodError(encode, (at, text)))
end

"""
    function decode(at::AbstractTokenizer, alphabet_size::Vector{<:Integer})
        throw(MethodError(decode, (at, alphabet_size)))
    end

Decode a list of token IDs into a string.
"""
function decode(at::AbstractTokenizer, alphabet_size::Vector{<:Integer})::String
    throw(MethodError(decode, (at, alphabet_size)))
end

"""
    function tokenizer(t::AbstractTokenizer)

Default getter for a tokenizer (itself).

This is useful if you are wrapping a tokenizer with a structure of type
[`AbstractTokenizer`](@ref).
"""
function tokenizer(t::AbstractTokenizer)
    return t
end

"""
    data(t::AbstractTokenizer) = t.vocabulary_data
"""
function data(t::AbstractTokenizer)
    throw(MethodError(data, (t,)))
end

"""
    offsets(t::AbstractTokenizer) = t.vocabulary_offsets

Return a vector whose i-th index indicates where the i-th token is encoded,
within `data(t)`.

See also [`data(t::AbstractTokenizer)`](@ref).
"""
function offsets(t::AbstractTokenizer)
    throw(MethodError(offsets, (t,)))
end

"""
    bytelengths(t::AbstractTokenizer) = t.vocabulary_bytelengths

Return a vector whose i-th index indicates how many bytes is long the i-th 
token encoded.
"""
function bytelengths(t::AbstractTokenizer)
    throw(MethodError(bytelengths, (t,)))
end

"""
    merges(t::AbstractTokenizer) = t.merges

Return the dictionary containing all the pair countings for BPE.
"""
function merges(t::AbstractTokenizer)
    throw(MethodError(merges, (t,)))
end

"""
    special_tokens(t::AbstractTokenizer) = t.special_tokens
"""
function special_tokens(t::AbstractTokenizer)
    throw(MethodError(special_tokens, (t,)))
end

"""
    function vocabulary_size(t::AbstractTokenizer)

Return how many bytes are encoded within `data(t)`.

See also [`data(t::AbstractTokenizer)`](@ref).
"""
function vocabulary_size(t::AbstractTokenizer)
    return length(offsets(t))
end

"""
    function token(t::AbstractTokenizer, i::Int)

Return the bytes associated with the i-th token ID.
"""
function token(t::AbstractTokenizer, i::I) where {I<:Integer}
    _offset_i = offsets(t)[i]
    return view(data(t), _offset_i:(_offset_i + bytelengths(t)[i] - 1))
end

"""
    function add_token!(t::AbstractTokenizer, bytes::Vector{UInt8})

Add a new token to the tokenizer, by appending the bytes to the raw `data`
collection, and incrementing by one the size of the `byteslengths` and `offsets`
collections.

See also [`data(t::AbstractTokenizer)`](@ref), 
[`bytelengths(t::AbstractTokenizer)`](@ref),
[`offsets(t::AbstractTokenizer)`](@ref).
"""
function add_token!(t::AbstractTokenizer, bytes::Vector{UInt8})
    vocabulary_data = data(t)
    vocabulary_bytelengths = bytelengths(t)
    vocabulary_offsets = offsets(t)

    push!(vocabulary_bytelengths, length(bytes))
    push!(vocabulary_offsets, length(vocabulary_data)+1)
    append!(vocabulary_data, bytes)
end
