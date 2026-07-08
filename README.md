# Lilliput.jl

My trial at implementing a transformer architecture, starting from [Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch?utm_source=raschka&utm_medium=affiliate&utm_campaign=book_raschka_build_12_12_23&a_aid=raschka&a_bid=4c2437a0&chan=mm_website) by Sebastian Raschka.

*With zero dependencies.*

### Tokenizer

At the moment of writing, the interface of each tokenizer is dscribed in the docstring of `AbstractTokenizer`.
See `src/tokenizers/abstract.jl`.

One example of GPT-2-like tokenizer is the `BPETokenizer` in `src/tokenizers/bpe.jl`.

### Testing

Just run `julia --project=docs docs/make.jl`.

If the documentation is building, it means that all the use cases appearing in
`jldoctest` environments are running.

This is a good compromise between sketching and showcasing examples of use-cases.

### Useful Resources

- [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch);
- [minbpe](https://github.com/karpathy/minbpe/);
- [Julia Base.Unicode](https://github.com/JuliaLang/julia/blob/master/base/strings/unicode.jl), which seems to be adapted from [utf8 proc](https://juliastrings.github.io/utf8proc/).
