pushfirst!(LOAD_PATH,"../src/")

using Documenter
using IterativeNelderMead

makedocs(
    sitename = "IterativeNelderMead",
    format = Documenter.HTML(),
    modules = [IterativeNelderMead]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "https://github.com/astrobc1/IterativeNelderMead.jl"
)
