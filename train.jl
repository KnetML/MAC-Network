include("src/newmacnetwork.jl")
o=Dict(:h5=>false,:mfile=>nothing,:epochs=>10,
                :lr=>0.0001,:p=>12,:ema=>0.999f0,:batchsize=>32,
                :selfattn=>false,:gating=>false,
                :shuffle=>true,:sorted=>false,:prefix=>string(now())[1:10],
                :vocab_size=>90,:embed_size=>300, :dhome=>"data/", :loadresnet=>false,:d=>512)
info("Loading questions ...")
trnqstns = getQdata(o[:dhome],"train")
valqstns = getQdata(o[:dhome],"val")
info("Loading dictionaries ... ")
qvoc,avoc,i2w,i2a = getDicts(o[:dhome],"dic")
sets = []
push!(sets,miniBatch(trnqstns))
push!(sets,miniBatch(valqstns))
trnqstns=nothing;
valqstns=nothing;

#MODEL
gpu(0)
w,r,_,_,_ = init_network(o);
wrun=deepcopy(w)
opts = optimizers(w,Adam;lr=o[:lr])

#FEATS
trnfeats = loadFeatures(o[:dhome],"train";h5=o[:h5])
valfeats = loadFeatures(o[:dhome],"val";h5=o[:h5])
gc();Knet.gc();gc();

for i=1:o[:epochs]		
       modelrun(w,r,opts,sets[1],trnfeats,o,wrun;train=true)   
       if iseven(i)	
            modelrun(wrun,r,opts,sets[2],valfeats,o;train=false)
       end
end

#getter(id) = view(trnfeats,:,:,:,id)
#ids,questions,answers,batchSizes,pad,families = rand(sets[1])
# #MEMORY LEAK
# for i=1:250
#            ids,questions,answers,batchSizes,pad,families = rand(sets[1])
# 	   B     = batchSizes[1]
#            xB    = ones(Float32,1,B)
#            xS    = batcher(map(getter,ids))
#            xP    = pad==nothing ? nothing : pad*Float32(1e22)
#            grads = lossfun(w,r,questions,batchSizes,xS,xB,xP;answers=answers,p=o[:p])
# end

# #NO LEAK
# for i=1:250
#            ids,questions,answers,batchSizes,pad,families = rand(sets[1])
# 	   B     = batchSizes[1]
#            xB    = ones(Float32,1,B)
#            xS    = batcher(map(getter,ids))
#            xP    = pad==nothing ? nothing : pad*Float32(1e22)
#            forward_net(w,r,questions,batchSizes,copy(xS),copy(xB),xP;answers=answers,p=o[:p])
# end

# #NO LEAK
# for i=1:500
#            ids,questions,answers,batchSizes,pad,families = rand(sets[1])
# 	   B     = batchSizes[1]
#            xB    = ones(Float32,1,B)
#            xS    = batcher(map(getter,ids))
#            xP    = pad==nothing ? nothing : pad*Float32(1e22)
# end

# #NO LEAK
# for i=1:250
#            grads = lossfun(w,r,questions,batchSizes,xS,xB,xP;answers=answers,p=o[:p])
# end
