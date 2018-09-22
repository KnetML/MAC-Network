include(ARGS[1])
include(ARGS[2])
println("Loading questions ...")
trnqstns = getQdata(o[:dhome],"train")
valqstns = getQdata(o[:dhome],"val")
println("Loading dictionaries ... ")
qvoc,avoc,i2w,i2a = getDicts(o[:dhome],"dic")
sets = []
push!(sets,shuffle!(miniBatch(trnqstns;B=32,srtd=true)))
push!(sets,shuffle!(miniBatch(valqstns;B=32,srtd=true)))
trnqstns=nothing;
valqstns=nothing;

#MODEL
gpu(0)
M = MACNetwork(o);
Mrun=deepcopy(M);
#FEATS
trnfeats = loadFeatures(o[:dhome],"train";h5=o[:h5])
valfeats = loadFeatures(o[:dhome],"val";h5=o[:h5])
Knet.gc()
M,Mrun = train!(M,Mrun,sets,(trnfeats,valfeats),o) 
