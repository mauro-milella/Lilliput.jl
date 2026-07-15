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
    push!(vocabulary_offsets, length(vocabulary_data) + 1)
    return append!(vocabulary_data, bytes)
end

"""
    function tokenprecision(::AbstractTokenizer)

Return the integer tokenprecision within the `merges` structure of a tokenizer.
"""
function tokenprecision(t::AbstractTokenizer)
    # you could do tokenprecision(::AbstractTokenizer{I}) where {I} return I
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
```julia
julia> using Lilliput

julia> bpetok = BPETokenizer(); 

julia> train(bpetok, 270, ["Hello, world!", "Hello hello!"]);

julia> mktempdir() do d 
    path = joinpath(d, "bpe-test-training")
    save(bpetok, path)
end
```
"""
function save(
    t::AbstractTokenizer, fileprefix::String; version::String="minbpe v1"
)
    I = tokenprecision(t)

    # write the model: to be used in load() later
    modelfile = fileprefix * ".model"

    open(modelfile, "w") do f
        # might be useful
        write(f, "$version\n")
        write(f, string(I))

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
            _special_tokens = special_tokens(t)
        catch
            _special_tokens = Dict{String,I}()
        end
        write(f, "$(length(special_tokens(t)))\n")
        for (token, index) in _special_tokens
            write(f, "$token $index\n")
        end

        # print the trained merges        
        # remember index is just a synonym for tokenID
        for (index1, index2) in sort(collect(merges(t)); by=x->x[2])
            write(f, "$index1 $index2\n")
        end
    end

    # create the human-readable file with the vocabulary
    vocabfile = fileprefix * ".vocab"
    inverted_merges = Dict{I,Tuple{I,I}}([b => a for (a, b) in merges(t)])

    open(vocabfile, "w") do f
        max_index = vocabulary_size(t)
        for current_index in 1:max_index
            token1 = token(t, current_index)
            token1_string = render(token1)

            if haskey(inverted_merges, current_index)
                index2, index3 = inverted_merges[current_index]

                token2 = token(t, index2)
                token3 = token(t, index3)

                token2_string = render(token2)
                token3_string = render(token3)

                write(
                    f,
                    "($token2_string) ($token3_string) => " *
                    "$token1_string $(current_index)\n",
                )
            else
                write(f, "($token1_string) $(current_index)\n")
            end
        end
    end
end

"""
    function load(fileprefix::String)

Read the model file produced by a call to [`save`](@ref).

```julia
julia> using Lilliput

julia> bpetok = BPETokenizer(); 

julia> train(bpetok, 270, ["Hello, world!", "Hello hello!"]);

julia> save(bpetok, "bpe-test-training")

julia> loadedbpetok = load("bpe-test-training")

julia> loadedbpetok.vocabulary_data == bpetok.vocabulary_data
true

julia> loadedbpetok.merges_data == bpetok.merges_data
true
```

"""
function load(filepath::String)
    @assert endswith(filepath, ".model") "Please provide a .model file " *
        "(you provided: $filepath)"

    # from this token ID (included), learned tokens are represented in merges
    initial_index = Int(256)

    # f is a file to be read, compliant with BPETokenizer's implementation;
    # when this is called, the version (the first header row) is already red.
    function _load_bpetokenizer(f)
        # first, read the integer precision;
        # this translates, say, :UInt16 in the corresponding DataType 
        I = getfield(Base, Symbol(readline(f)))

        merges_data = Dict{Tuple{I,I},I}()
        vocabulary_data = Vector{UInt8}()
        vocabulary_offsets = Vector{Int}()
        vocabulary_bytelengths = Vector{Int}()
        special_tokens = Dict{String,I}()

        # first of all, we need to reconstruct the vocabulary
        _offset = 1
        for i in 1:(initial_index-1) # (the original alphabet size)
            bytes = codeunits(string(Char(i)))
            len = length(bytes)

            append!(vocabulary_data, bytes)
            push!(vocabulary_offsets, _offset)
            push!(vocabulary_bytelengths, len)

            _offset += len
        end

        # similar to token(::AbstractTokenizer, id::Int), but without tokenizer
        function _token_bytes(id)
            offset = vocabulary_offsets[id]
            len = vocabulary_bytelengths[id]

            return view(vocabulary_data, offset:(offset + len - 1))
        end

        # another utility
        function _strip_parentheses(t)
            return strip(t, ['(', ')', ','])
        end

        # special tokens
        nspecials = parse(Int, strip(readline(f)))
        for _ in 1:nspecials
            line = readline(f)
            token, index = split(strip(line))

            token = _strip_parentheses(token)
            special_tokens[token] = parse(I, index)
        end

        # read the merges
        for line in eachline(f)
            index1, index2, new_id = split(strip(line))

            index1 = parse(I, _strip_parentheses(index1))
            index2 = parse(I, _strip_parentheses(index2))
            new_id = parse(I, new_id)

            # create a new token from existing ones
            token_bytes = vcat(_token_bytes(index1), _token_bytes(index2))
            # not correct
            # token_bytes = collect(codeunits(string(Char(index1), Char(index2))))
            
            push!(vocabulary_offsets, length(vocabulary_data) + 1)
            push!(vocabulary_bytelengths, length(token_bytes))
            append!(vocabulary_data, token_bytes)

            merges_data[(index1, index2)] = new_id
        end

        return BPETokenizer{I}(
            vocabulary_data,
            vocabulary_offsets,
            vocabulary_bytelengths,
            merges_data,
            special_tokens,
        )
    end

    open(filepath, "r") do f
        version = strip(readline(f))

        if version == "minbpe v1"
            return _load_bpetokenizer(f)
        else
            error("Unsupported version $version")
        end
    end
end
