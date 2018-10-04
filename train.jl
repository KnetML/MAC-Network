import Pkg; Pkg.activate("."); Pkg.instantiate();
include(ARGS[1])
include(ARGS[2])
println("Loading questions ...")
Knet.seed!(11131994)
trnqstns = getQdata(o[:dhome],"train")
valqstns = getQdata(o[:dhome],"val")
println("Loading dictionaries ... ")
qvoc,avoc,i2w,i2a = getDicts(o[:dhome],"dic")
sets = []
push!(sets,miniBatch(trnqstns;B=64))
push!(sets,miniBatch(valqstns;B=64))
trnqstns=nothing;
valqstns=nothing;
#MODEL
#gpu(0)
M    = MACNetwork(o);
Mrun = MACNetwork(o);
for (wr,wi) in zip(params(Mrun),params(M));
    wr.value[:] = wi.value[:]
end
Knet.gc()
#FEATS
trnfeats = loadFeatures(o[:dhome],"train";h5=o[:h5])
valfeats = loadFeatures(o[:dhome],"val";h5=o[:h5])
M,Mrun = train!(M,Mrun,sets,(trnfeats,valfeats),o)
