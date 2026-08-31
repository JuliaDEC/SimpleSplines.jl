using SimpleSplines
using Documenter

DocMeta.setdocmeta!(SimpleSplines, :DocTestSetup, :(using SimpleSplines); recursive = true)

makedocs(;
    modules = [SimpleSplines],
    authors = "Michael Kraus",
    sitename = "SimpleSplines.jl",
    format = Documenter.HTML(;
        canonical = "https://JuliaDEC.github.io/SimpleSplines.jl",
        edit_link = "main",
        assets = String[]
    ),
    pages = [
        "Home" => "index.md",
    ]
)

deploydocs(;
    repo = "github.com/JuliaDEC/SimpleSplines.jl",
    devbranch = "main"
)
