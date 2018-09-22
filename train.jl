include(ARGS[1])
include(ARGS[2])
# o=Dict(:h5=>false,:mfile=>nothing,:epochs=>16,
#                 :lr=>0.0001,:p=>12,:ema=>0.999f0,:batchsize=>32,
#                 :selfattn=>false,:gating=>false,
#                 :shuffle=>true,:sorted=>false,:prefix=>string(now())[1:10],
#                 :vocab_size=>90,:embed_size=>300, :dhome=>"data/", :loadresnet=>false,:d=>512)
println("Loading questions ...")
trnqstns = getQdata(o[:dhome],"train")
valqstns = getQdata(o[:dhome],"val")
println("Loading dictionaries ... ")
qvoc,avoc,i2w,i2a = getDicts(o[:dhome],"dic")
sets = []
push!(sets,shuffle!(miniBatch(trnqstns;B=48,srtd=true)))
push!(sets,shuffle!(miniBatch(valqstns;B=48,srtd=true)))
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
