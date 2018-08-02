for p in ("Knet","JLD","JSON","Images","WordTokenizers","ProgressMeter","ArgParse","MAT")
    Pkg.installed(p) == nothing && Pkg.add(p)
end
