using JSON,JLD
dhome    = ARGS[1]
trgthome = dhome * "demo/"
valfile  = dhome * "val.json"
dicfile  = dhome * "dic.json"
oclvr    = dhome * "CLEVR_v1.0/images/val/"
clvrhome = dhome * "demo/CLEVR_v1.0/"
imgshome = dhome * "demo/CLEVR_v1.0/images/"
valimgs  = dhome * "demo/CLEVR_v1.0/images/val/"

function loadFeatures(dhome,set)
    feats    = reinterpret(Float32,read(open(dhome*set*".bin")))
    reshape(feats,(14,14,1024,div(length(feats),200704)))
end

function getdemo(file;part=100)
    info("Validation feats are loading...")
    valfeats  = loadFeatures(dhome,"val")
    demofeats = Any[];
    info("Validation data are loading...")
    data      = JSON.parsefile(file)
    demo      = data[randperm(length(data))[1:part]]

    info("Random selection are collecting...")
    for d in demo
        imgname = d[1]
    	cp(oclvr*imgname ,valimgs*imgname;remove_destination=true)
        id = parse(Int,imgname[end-9:end-4])+1
        push!(demofeats,valfeats[:,:,:,id])
    end
    demofeats = cat(4,demofeats...)

    info("Demo feats are writed...")
    f = open(trgthome*"demo.bin","w")
    write(f,demofeats); close(f)

    info("Demo data is writed...")
    open(trgthome * "demo.json","w") do f
        write(f,json(demo))
    end
end

info("Creating demo folders if necessary...")
!isdir(trgthome) && mkdir(trgthome)
!isdir(clvrhome) && mkdir(clvrhome)
!isdir(imgshome) && mkdir(imgshome)
if isdir(valimgs)
   rm(valimgs;recursive=true)
end
mkdir(valimgs)

getdemo(valfile)
cp(dicfile,trgthome  * "dic.json";remove_destination=true)
run(`tar -cvzf demo.tar.gz $(trgthome)`)
