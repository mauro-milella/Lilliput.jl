module Tokenizers

include("utilities.jl")
export count_consecutives, merge

include("abstract.jl")
export AbstractTokenizer
export train, encode, decode

include("bpe.jl")
export Tokenizer
export data, offsets, bytelengths, merges, special_tokens
export vocabulary_lastindex, token, add_token!
export BasicTokenizer

end
