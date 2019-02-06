import Pkg; Pkg.activate("."); Pkg.instantiate();
include("configs/config2.jl")
include("src/main.jl")
#MODEL
@show arrtype
M    = MACNetwork(o);
Knet.gc()
#FEATS
function benchmark(M::MACNetwork,o::Dict;N=10)
    B=64; L=25
    for i=1:N
        ids  = randperm(128)[1:B]
        xB   = arrtype(ones(Float32,1,B))
        xS   = arrtype(randn(Float32,14,14,1024,B))
        xQ   = [rand(1:84) for i=1:B*L]
        answers = [rand(1:28) for i=1:B]
        batchSizes = [B for i=1:L]
        xP   = nothing
        y    = @diff M(xQ,batchSizes,xS,xB,xP;answers=answers,p=o[:p],selfattn=o[:selfattn],gating=o[:gating])
    end
end

benchmark(M,o;N=100)
