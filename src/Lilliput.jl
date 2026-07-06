module Lilliput

include("tokenizers/tokenizers.jl")
using .Tokenizers

# yep, I am not using Reexport
export count_consecutives, merge, render

export AbstractTokenizer
export train, encode, decode
export tokenizer, data, offsets, bytelengths, merges, special_tokens
export vocabulary_size, token, add_token!
export precision
export save, load

export BPETokenizer

end
