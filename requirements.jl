for p in ("Knet","JLD","JSON","Images","WordTokenizers","ProgressMeter","ArgParse","MAT")
    Pkg.installed(p) == nothing && Pkg.add(p)
end
# Pkg.checkout("AutoGrad")
# Pkg.checkout("Knet")
# Pkg.build("Knet")

