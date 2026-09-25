using Documenter
using SimpleSplines
using Test

DocMeta.setdocmeta!(SimpleSplines, :DocTestSetup, :(using SimpleSplines); recursive = true)

doctest(SimpleSplines; manual = false)
