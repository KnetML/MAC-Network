for p in ("Knet","JLD","JSON","Images","WordTokenizers")
    Pkg.installed(p) == nothing && Pkg.add(p)
end
