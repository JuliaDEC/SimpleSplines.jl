using SimpleSplines
using Documenter
using DocumenterCitations

DocMeta.setdocmeta!(SimpleSplines, :DocTestSetup, :(using SimpleSplines); recursive = true)

bib = CitationBibliography(joinpath(@__DIR__, "src", "references.bib"))

makedocs(;
    plugins = [bib],
    modules = [SimpleSplines],
    authors = "Michael Kraus",
    sitename = "SimpleSplines.jl",
    checkdocs = :exports,
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://JuliaDEC.github.io/SimpleSplines.jl",
        edit_link = "main",
        # the library page collects every docstring in the package and is legitimately
        # large; the default warning threshold is aimed at pages that are large by accident
        size_threshold_warn = 400 * 1024,
        size_threshold = 800 * 1024,
        assets = String[]
    ),
    pages = [
        "Home" => "index.md",
        "Tutorial" => "tutorial.md",
        "Theory" => [
            "B-Splines" => "theory/bsplines.md",
            "Boundary Conditions" => "theory/boundary.md",
            "Tensor Products" => "theory/tensorproduct.md"
        ],
        "Usage" => [
            "Meshes" => "usage/meshes.md",
            "Bases" => "usage/bases.md",
            "Boundary Conditions" => "usage/boundary.md",
            "Assembly" => "usage/quadrature.md",
            "Tensor Products" => "usage/tensorproduct.md",
            "Splines" => "usage/splines.md"
        ],
        "Gallery" => "gallery.md",
        "Library" => "library.md",
        "References" => "references.md"
    ]
)

deploydocs(;
    repo = "github.com/JuliaDEC/SimpleSplines.jl",
    devbranch = "main"
)
