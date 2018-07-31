for p in ("Knet","JLD","JSON","Images")
    Pkg.installed(p) == nothing && Pkg.add(p)
end
