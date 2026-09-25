using Documenter
using SimpleSplines
using Test

DocMeta.setdocmeta!(SimpleSplines, :DocTestSetup, :(using SimpleSplines); recursive = true)

# `manual = false`: every jldoctest block is in a docstring, and none is under docs/src.
# Drop the flag when a page under docs/src gets a jldoctest block.
doctest(SimpleSplines; manual = false)
