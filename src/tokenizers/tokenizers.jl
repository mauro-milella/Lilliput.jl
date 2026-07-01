module Tokenizers

include("utilities.jl")
export count_consecutives, merge

include("abstract.jl")
export AbstractTokenizer
export train, encode, decode
export Tokenizer

end
