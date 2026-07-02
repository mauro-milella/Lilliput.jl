module Tokenizers

include("utilities.jl")
export count_consecutives, merge

include("abstract.jl")
export AbstractTokenizer
export train, encode, decode
export tokenizer, data, offsets, bytelengths, merges, special_tokens
export vocabulary_lastindex, token, add_token!

include("bpe.jl")
export Tokenizer
export BasicTokenizer

end
