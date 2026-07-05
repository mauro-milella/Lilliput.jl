"""
    abstract type AbstractTokenizer end

Generic tokenizer.

Must implement the following:

- tokenizer(::AbstractTokenizer)
- data(::AbstractTokenizer)
- offsets(::AbstractTokenizer)
- bytelengths(::AbstractTokenizer)
- merges(::AbstractTokenizer)
- pattern(::AbstractTokenizer)
- special_tokens(::AbstractTokenizer)
- train(::AbstractTokenizer, ::Int, ::Vector{String})
- encode(::AbstractTokenizer, ::String)
- decode(::AbstractTokenizer, Vector{<:Integer})

Offers the following minimal utilities:

- vocabulary_size(::AbstractTokenizer)
- token(::AbstractTokenizer, i::I) where {I<:Integer}
- add_token!(::AbstractTokenizer, bytes::Vector{UInt8})
- save(::AbstractTokenizer, fileprefix::String)
- load(fileprefix::String)
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
    function pattern(t::AbstractTokenizer)

Return the pattern leveraged by the tokenizer.
"""
function pattern(t::AbstractTokenizer)
    throw(MethodError(pattern, (t,)))
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

"""
    function save(t::AbstractTokenizer, fileprefix::String)

Save `<fileprefix>.model` file for loading back the tokenizer, and 
`<fileprefix>.vocab for human inspection.
"""
function save(t::AbstractTokenizer, fileprefix::String)
    # write the model: to be used in load() later
    modelfile = fileprefix + ".model"
    open(modelfile, "w") do f
        f.write("minbpe v1\n")
        f.write(f"{self.pattern}\n")
        # write the special tokens, first the number of them, then each one
        f.write(f"{len(self.special_tokens)}\n")
        for special, idx in self.special_tokens.items():
            f.write(f"{special} {idx}\n")
        # the merges dict
        for idx1, idx2 in self.merges:
            f.write(f"{idx1} {idx2}\n")

        # write the vocab: for the human to look at
        vocab_file = fileprefix + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                # note: many tokens may be partial utf-8 sequences
                # and cannot be decoded into valid strings. Here we're using
                # errors='replace' to replace them with the replacement char �.
                # this also means that we couldn't possibly use .vocab in load()
                # because decoding in this way is a lossy operation!
                s = render_token(token)
                # find the children of this token, if any
                if idx in inverted_merges:
                    # if this token has children, render it nicely as a merge
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    # otherwise this is leaf token, just print it
                    # (this should just be the first 256 tokens, the bytes)
                    f.write(f"[{s}] {idx}\n")
