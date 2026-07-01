using Documenter
using Lilliput

makedocs(;
    sitename="Lilliput.jl documentation",
    modules=[Lilliput, Lilliput.Tokenizers],
    remotes=nothing,
    doctest=true,
)
