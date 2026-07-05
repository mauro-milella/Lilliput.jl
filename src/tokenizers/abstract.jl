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
    function render(_token::Vector{UInt8})

Pretty print the bytes in render.
This is a naive and unsafe implementation, substituting newlines with a string
`<|newline|>` and tabs with `<|tab|>`.

We should wrap every dangerous symbol in a safe control sequence.
"""
function render(_tokens::Vector{UInt8})
    replace(String(_tokens), "\n" => "<|newline|>", "\t" => "<|tab|>")
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
    function precision(::AbstractTokenizer)

Return the integer precision within the `merges` structure of a tokenizer.
"""
function precision(t::AbstractTokenizer)
    # you could do precision(::AbstractTokenizer{I}) where {I} return I
    # but I don't want to make every child of AbstractTokenizer so constrained
    #
    # eltype().parameters returns svec(Tuple{UInt16, UInt16}, UInt16)
    # we assume merge is a dictionary 
    return eltype(merges(t)).parameters[2]
end

"""
    function save(t::AbstractTokenizer, fileprefix::String)

Save `<fileprefix>.model` file for loading back the tokenizer, and 
`<fileprefix>.vocab for human inspection.

# Examples
```jldoctest
julia> println("TODO:")
```
"""
function save(t::AbstractTokenizer, fileprefix::String; version::String="minbpe v1")
    I = precision(t)

    # write the model: to be used in load() later
    modelfile = fileprefix * ".model"

    open(modelfile, "w") do f
        # might be useful
        write(f, "$version\n")

        # print the regex used, if any
        _pattern = nothing
        try
            _pattern = pattern(t)
        catch
            _pattern = ""
        end
        write(f, "$_pattern\n")

        # print the number of special tokens (and themselves) if any 
        _special_tokens = nothing
        try 
            _special_tokens = special_tokens()
        catch
            _special_tokens = Dict{String, I}()
        end
        write(f, "$(length(special_tokens(t)))\n")
        for (token, index) in _special_tokens(t)
            write(f, "$token $index\n")
        end

        # print the trained merges        
        # remember index is just a synonym for tokenID
        for (index1, index2) in merges(t)
            f.write(f, "$index1 $index2\n")
        end
    end

    # create the human-readable file with the vocabulary
    vocabfile = fileprefix * ".vocab"
    inverted_merges = Dict{I, Pair{I,I}}([b => a for (a,b) in merges(t)])

    open(vocabfile, "w") do f
        max_index = vocabulary_size(t)
        for current_index in 1:max_index
            token1 = token(t, current_index)
            token1_string = render(token1)

            if token1 in inverted_merges
                token2, token3 = inverted_merges[token1]
                token2_string = render(token2)
                token3_string = render(token3)

                f.write(f, "($token2_string) ($token3_string) => " * 
                        "$token1_string $(current_index)\n")
            else
                f.write(f, "($token1_string) $(current_index)\n")
            end
        end
    end
end

"""
    function load(fileprefix::String)

Read the model file produced by a call to [`save`](@ref).
"""
function load(filepath::String)
    # TODO
end
