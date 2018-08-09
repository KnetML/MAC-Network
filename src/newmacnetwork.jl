using JSON,JLD,HDF5,Knet,AutoGrad
include("loss.jl")
if !isdefined(Main,:atype)
    global atype = KnetArray{Float32}
end
init(o...)=atype(xavier(Float32,o...))
bias(o...)=atype(zeros(Float32,o...))

function elu(x)
    return relu(x) + (exp(min(0,x)) - 1)
end

function load_resnet(atype;stage=3)
    w,m,meta = ResNetLib.resnet101init(;trained=true,stage=stage)
    global avgimg = meta["normalization"]["averageImage"]
    global descriptions = meta["classes"]["description"]
    return w,m,meta,avgimg
end

function prepocess_image(w,m,imgurl,avgimg;stage=3)
    img = imgdata(imgurl, avgimg)
    return ResNetLib.resnet101(w,m,atype(img);stage=stage);
end

function postpocess_kb(w,x;train=false,pdrop=0.0)
    # if train
    #      x  = dropout(x,pdrop)
    # end
    x  = elu.(conv4(w[1],x;padding=1,stride=1,mode=1) .+ w[2])
    # if train
    #      x  = dropout(x,pdrop)
    # end
    x  = elu.(conv4(w[3],x;padding=1,stride=1,mode=1) .+ w[4])
    h,w,c,b = size(x)
    x  = reshape(x,h*w,c,b)
    return permutedims(x,(2,3,1))
end

function init_postkb(d)
    w = Any[];
    push!(w,init(3,3,1024,d))
    push!(w,bias(1,1,d,1))
    push!(w,init(3,3,d,d))
    push!(w,bias(1,1,d,1))
    return w
end

function process_question(w,r,words,batchSizes;train=false,qdrop=0.0,embdrop=0.0)
    wordemb      = w[1][:,words]
    # if train
    #      wordemb = dropout(wordemb,embdrop)
    # end
    y,hyout,_,_ = rnnforw(r,w[2],wordemb;batchSizes=batchSizes,hy=true,cy=false)

    q            = vcat(hyout[:,:,1],hyout[:,:,2])
    # if train
    #        q  = dropout(q,qdrop)
    # end
    indices      = batchSizes2indices(batchSizes)
    lngths       = length.(indices)
    Tmax         = maximum(lngths)
    td,B         = size(q)
    d            = div(td,2)
    cw           = Any[];
    for i=1:length(indices)
        y1 = y[:,indices[i]]
        df = Tmax-lngths[i]
        if df > 0
	    cpad = zeros(Float32,2d,df)
            kpad = atype(cpad)
            ypad = hcat(y1,kpad)
            push!(cw,ypad)
        else
            push!(cw,y1)
        end
    end
    cws_2d =  reshape(vcat(cw...),2d,B*Tmax)
    cws_3d =  reshape(w[3]*cws_2d .+ w[4],(d,B,Tmax))
    return q,cws_3d;
end

function batchSizes2indices(batchSizes)
    B = batchSizes[1]
    indices = Any[]
    for i=1:B
        ind = i.+cumsum(filter(x->(x>=i),batchSizes)[1:end-1])
        push!(indices,append!(Int[i],ind))
    end
    return indices
end

function init_rnn(inputSize,embedSize,hiddenSize;outputSize=nothing)
    w = Any[]
    wembed = atype(rand(Float32,embedSize,inputSize))
    push!(w,wembed)
    r,wrnn = rnninit(embedSize,hiddenSize; bidirectional=true, binit=zeros) #change Knet/src/rnn.jl
    #setfgbias!(r,wrnn)
    push!(w,wrnn)
    push!(w,init(hiddenSize,2hiddenSize),bias(hiddenSize,1))
    return w,r
end

function setfgbias!(r,w)
    rnnparam(r,w,1,2,2)[:] = 0.5
    rnnparam(r,w,1,6,2)[:] = 0.5
    rnnparam(r,w,2,2,2)[:] = 0.5
    rnnparam(r,w,2,6,2)[:] = 0.5
end

function control_unit(w,ci₋1,qi,cws,pad;train=false,tap=nothing)
    #cws       : d x B x T
    d,B,T      = size(cws)
    #qi,ci     : d x B
    cqi        = reshape(w[1] * vcat(ci₋1,qi) .+ w[2],(d,B,1)) # eq c1
    #cqi       : d x B x 1
    cvis       = reshape(cqi .* cws,(d,B*T)) #eq c2.1.1
    #cvis      : d x BT
    cvis_2d    = reshape(w[3] * cvis .+ w[4],(B,T)) #eq c2.1.2
    #cvis_2d   : B x T
    if pad != nothing
        cvi    = reshape(softmax(cvis_2d - pad,2),(1,B,T)) #eq c2.2
    else
        cvi    = reshape(softmax(cvis_2d,2),(1,B,T)) #eq c2.2
    end
    #tap!=nothing && get!(tap,"w_attn_$(tap["cnt"])",Array(reshape(cvi,B,T)))
    #cvi       : 1 x B x T
    ci         = reshape(sum(cvi.*cws,3),(d,B)) #eq c2.3
end

function init_control(d)
    w = Any[]
    push!(w,init(d,2d))
    push!(w,bias(d,1))
    push!(w,init(1,d))
    push!(w,bias(1,1))
    return w
end

function read_unit(w,mi₋1,ci,KBhw,KBhw′′;train=false,mdrop=0.0,attdrop=0.0,tap=nothing)
    d,B,N          = size(KBhw)
    d,BN           = size(KBhw′′)
    #KBhw'     : d x B x N  := w[3] * khw + w[4]
    #KBhw''    : d x BN     := reshape(w[5] * khw + w[7]),(d,BN))
    # if train
    #     mi₋1       = dropout(mi₋1,mdrop)
    # end

    mi_3d           = reshape(w[1]*mi₋1 .+ w[2],(d,B,1)) #eq r1.1
    #mi_3d     : d x B x 1
    ImKB            = reshape(mi_3d .* KBhw,(d,BN)) # eq r1.2
    #ImKB      : d x BN
    ImKB′           = reshape(w[5] * ImKB .+ KBhw′′,(d,B,N)) #eq r2
    #ImKB'     : d x B x N
    ci_3d           = reshape(ci,(d,B,1))
    #ci_3d     : d x B x 1
    IcmKB_pre       = reshape(ci_3d .* ImKB′,(d,BN)) #eq r3.1.1
    #IcmKB_pre : d x BN
    # if train
    #      IcmKB_pre = dropout(IcmKB_pre,attdrop)#dropout(IcmKB_pre,0.15)
    # end
    IcmKB           = reshape(w[6] * IcmKB_pre  .+ w[7],(B,N)) #eq r3.1.2
    #IcmKB     : B x N
    mvi             = reshape(softmax(IcmKB,2),(1,B,N)) #eq r3.2
    #mvi       : 1 x B x N
    #tap!=nothing && get!(tap,"KB_attn_$(tap["cnt"])",Array(reshape(mvi,B,N)))
    mnew            = reshape(sum(mvi.*KBhw,3),(d,B)) #eq r3.3
end

function init_read(d)
    w = Any[]
    push!(w,init(d,d))
    push!(w,bias(d,1))
    push!(w,init(d,d))
    push!(w,bias(d,1))
    push!(w,init(d,d))
    push!(w,init(1,d))
    push!(w,bias(1,1))
    return w
end

function write_unit(w,m_new,mi₋1,mj,ci,cj;train=false,selfattn=true,gating=true)
    d,B        = size(m_new)
    #mnew      : d x B
    mi         =  w[1] * vcat(m_new,mi₋1) .+ w[2] #eq w.1 #no such layer in code delete w6-w7
    #mi       : d x B
    !selfattn && return mi

    d,BT       = size(mj)
    #mj        : d x BT
    T          = div(BT,B)
    #iproj     =  w[6] * ci .+ w[7] !!!delete w[6]
    ci_3d      = reshape(ci,d,B,1)
    #ci_3d     : d x B x 1
    cj_3d      = reshape(cj,d,B,T)
    #cj_3d     : d x B x T
    sap        = reshape(ci_3d.*cj_3d,(d,BT)) #eq w2.1.1
    #sap       : d x BT
    sa         = reshape(w[3] * sap .+ w[4],(B,T)) #eq w2.1.2
    #sa        : B x T
    sa′        = reshape(softmax(sa,2),(1,B,T)) #eq w2.1.3
    #sa'       : 1 x B x T
    mj_3d      = reshape(mj,(d,B,T))
    #mj_3d     : d x B x T
    mi_sa      = reshape(sum(sa′ .* mj_3d ,3),(d,B)) #eq w2.2
    #m_sa      : d x B
    mi′′       = w[5] * mi_sa .+ w[6] .+ mi   #eq w2.3
    #mi′′      : d x B

    !gating && return mi′′

    σci′       = sigm.(w[7] * ci .+ w[8])  #eq w3.1
    #σci′      : 1 x B
    mi′′′      = (σci′ .* mi₋1) .+  ((1.-σci′) .* mi′′) #eq w3.2
end

function init_write(d;selfattn=false,gating=false)
    w = Any[]
    push!(w,init(d,2d))
    push!(w,bias(d,1))
    !selfattn && return w
    push!(w,init(1,d))
    push!(w,bias(1,1))
    push!(w,init(d,d))
    push!(w,bias(d,1))
    !gating  && return w
    push!(w,init(1,d))
    push!(w,bias(1,1))
    return w
end


function mac(w,cw,qi,KBhw,KBhw′′,ci₋1,mi₋1,cj,mj,pad;train=false,selfattn=false,gating=false,tap=nothing)
    ci     = control_unit(w[1:4],ci₋1,qi,cw,pad;train=train,tap=tap)
    m_new  = read_unit(w[5:11],mi₋1,ci,KBhw,KBhw′′;train=train,tap=tap)
    mi     = write_unit(w[12:end],m_new,mi₋1,mj,ci,cj;train=train,selfattn=selfattn,gating=gating)
    #tap != nothing && (tap["cnt"]+=1)
    return ci,mi
end

function init_mac(d;selfattn=false,gating=false)
   w  = Any[];
   append!(w,init_control(d))
   append!(w,init_read(d))
   append!(w,init_write(d;selfattn=selfattn,gating=gating))
   return w
end

function output_unit(w,q,mp;train=false,pdrop=0.0)
  x  = elu.(w[1] * vcat(mp,q) .+ w[2])
  # if train
  #     x  = dropout(x,pdrop) #0.15
  # end
  y  = w[3] * x .+ w[4]
end

function init_output(d;outsize=28)
    w = Any[];
    push!(w,init(d,3d))
    push!(w,bias(d,1))
    push!(w,init(28,d))
    push!(w,bias(28,1))
    return w;
end

loss_layer(y,answers) = nll(y,answers;average=true)

function forward_net(w,r,qs,batchSizes,xS,xB,xP;answers=nothing,p=12,selfattn=false,gating=false,tap=nothing)

    train         = answers!=nothing
    #STEM Processing
    KBhw          = postpocess_kb(w[1:4],xS;train=train)

    #Read Unit Precalculations
    d,B,N         = size(KBhw)
    KBhw_2d       = reshape(KBhw,(d,B*N))
    KBhw′′        = w[17]*KBhw_2d .+ w[18]

    #Question Unit
    q,cws         = process_question(w[5:8],r,qs,batchSizes;train=train)

    qi_c          = w[9]*q .+ w[10]

    #Memory Initialization
    ci            = w[end-1]*xB
    mi            = w[end]*xB

    if selfattn
        cj=ci; mj=mi
    else
        cj=nothing; mj=nothing
    end

    #PAD for Word Attention

    #MAC Units
    wmac          = w[11:end-6]
    for i=1:p
        qi        = qi_c[(i-1)*d+1:i*d,:]
        if train
            ci = dropout(ci,0.15); mi = dropout(mi,0.15)
        end
        ci     = control_unit(wmac[1:4],ci,qi,cws,xP;train=train,tap=tap)
        m_new  = read_unit(wmac[5:11],mi,ci,KBhw,KBhw′′;train=train,tap=tap)
        mi     = write_unit(wmac[12:end],m_new,mi,mj,ci,cj;train=train,selfattn=selfattn,gating=gating)
        if selfattn
            cj = hcat(cj,ci); mj = hcat(mj,mi)
        end
    end


    y = output_unit(w[end-5:end-2],q,mi;train=train)


    if answers==nothing
        predmat = Array{Float32}(y)
        tap!=nothing && get!(tap,"y",predmat)
        return mapslices(indmax,predmat,1)[1,:]
    else
        return loss_layer(y,answers)
    end

end


function init_network(o)
    if o[:loadresnet]
        rsnt,m,meta = load_resnet(o[:atype];stage=3);
    else
        rsnt,m,meta = nothing,nothing,nothing;
    end
    w           = Any[];
    wcnn        = init_postkb(o[:d])
    append!(w,wcnn);
    wrnn,r      = init_rnn(o[:vocab_size],o[:embed_size],o[:d])
    append!(w,wrnn);
    #for i=1:p
        push!(w,init(o[:p]*o[:d],2*o[:d]),bias(o[:p]*o[:d],1)) #qi embbedding
    #end
    wmac = init_mac(o[:d];selfattn=o[:selfattn],gating=o[:gating]) #!!!share weights among cells
    append!(w,wmac);
    wout = init_output(o[:d])
    append!(w,wout);
    push!(w,init_state(o[:d],1;initial=:xavier))
    push!(w,init_state(o[:d],1;initial=:randn)) # m0
    return w,r,rsnt,m,meta;
end

function init_state(d,B;initial=:zero)
    if initial == :zero
        x=zeros(Float32,d,B)
    elseif initial == :randn
        x=randn(Float32,d,B)
    elseif initial == :xavier
        x=xavier(Float32,d,B)
    end
    return atype(x)
end

lossfun = gradloss(forward_net)

function savemodel(filename,w,wrun,r,opts,o)
    save(filename,"w",w,"wrun",wrun,"r",r,"opts",opts,"o",o)
end

function loadmodel(filename;onlywrun=false)
    d = load(filename)
    if onlywrun
        wrun=d["wrun"];r=d["r"];opts=nothing;w=nothing;o=d["o"]
    else
        w=d["w"];wrun=d["wrun"];r=d["r"];opts=d["opts"];o=d["o"];
    end
    return w,wrun,r,opts,o;
end

function getQdata(dhome,set)
    JSON.parsefile(dhome*set*".json")
end


function invert(vocab)
       int2tok = Array{String}(length(vocab))
       for (k,v) in vocab; int2tok[v] = k; end
       return int2tok
end

function getDicts(dhome,dicfile)
    dic  = JSON.parsefile(dhome*dicfile*".json")
    qvoc = dic["word_dic"]
    avoc = dic["answer_dic"]
    i2w  = invert(qvoc)
    i2a  = invert(avoc)
    return qvoc,avoc,i2w,i2a
end

function loadFeatures(dhome,set;h5=false)
    if h5
        return h5open(dhome*set*".hdf5","r")["data"]
    else
        feats = reinterpret(Float32,read(open(dhome*set*".bin")))
        return reshape(feats,(14,14,1024,div(length(feats),200704)))
    end
end

function miniBatch(data;shfl=true,srtd=false,B=32)
    L = length(data)
    shfl && shuffle!(data)
    srtd && sort!(data;by=x->length(x[2]))
    batchs = [];
    for i=1:B:L
        b         = min(L-i+1,B)
        questions = Any[]
        answers   = zeros(Int,b)
        images    = Any[]
        families  = zeros(Int,b)

        for j=1:b
            crw = data[i+j-1]
            push!(questions,reverse(Array{Int}(crw[2])))
            push!(images,parse(Int,crw[1][end-9:end-4])+1)
            answers[j]  = crw[3]
            families[j] = crw[4]
        end

        lngths     = length.(questions);
        srtindices = sortperm(lngths;rev=true)

        lngths     = lngths[srtindices]
        Tmax       = lngths[1]
        questions  = questions[srtindices]
        answers    = answers[srtindices]
        images     = images[srtindices]
        families   = families[srtindices]

        qs = Int[];
        batchSizes = Int[];
        pads = falses(b,Tmax)

        for k=1:b
           pads[k,lngths[k]+1:Tmax]=true
        end

        if sum(pads)==0
           pads=nothing
        end

        while true
            batch = 0
            for j=1:b
                if length(questions[j]) > 0
                    batch += 1
                    push!(qs,pop!(questions[j]))
                end
            end
            if batch != 0
                push!(batchSizes,batch)
            else
                break;
            end
        end
        push!(batchs,(images,qs,answers,batchSizes,pads,families))
    end
    return batchs
end

function loadTrainingData(dhome="data/";h5=false)
    !h5 && info("Loading pretrained features for train&val sets.
                It requires minimum 70GB RAM!!!")
    trnfeats = loadFeatures(dhome,"train";h5=h5)
    valfeats = loadFeatures(dhome,"val";h5=h5)
    info("Loading questions ...")
    trnqstns = getQdata(dhome,"train")
    valqstns = getQdata(dhome,"val")
    info("Loading dictionaries ... ")
    qvoc,avoc,i2w,i2a = getDicts(dhome,"dic")
    return (trnfeats,valfeats),(trnqstns,valqstns),(qvoc,avoc,i2w,i2a)
end

function loadDemoData(dhome="data/demo/")
    info("Loading demo features ...")
    feats = loadFeatures(dhome,"demo")
    info("Loading demo questions ...")
    qstns = getQdata(dhome,"demo")
    info("Loading dictionaries ...")
    dics = getDicts(dhome,"dic")
    return feats,qstns,dics
end


batchindex(xs, i) = (reverse(Base.tail(reverse(indices(xs))))..., i)
function batcher(xs)
    data = first(xs) isa AbstractArray ?
    similar(first(xs), size(first(xs))..., length(xs)) :
    Vector{eltype(xs)}(length(xs))
    for (i, x) in enumerate(xs)
        data[batchindex(data, i)...] = x
    end
    return data
end

function modelrun(w,r,opts,data,feats,o,wrun=nothing;train=false)
    if train
        cumgrads = map(similar,w)
    end

    getter(id) = view(feats,:,:,:,id)

    cnt=total=0.0;
    L = length(data)
    println("Timer Starts");flush(STDOUT);tic();

    for i=1:L
        ids,questions,answers,batchSizes,pad,families = data[i]
        B    = batchSizes[1]
        xB   = atype(ones(Float32,1,B))
        xS   = atype(batcher(map(getter,ids)))
        xP   = pad==nothing ? nothing : atype(pad*Float32(1e22))

        if train
            grads,lss = lossfun(w,r,questions,batchSizes,xS,xB,xP;answers=answers,p=o[:p])

	    cnt +=lss*B
	    total += B

            for (cu,gr) in zip(cumgrads,grads)
                axpy!(0.5f0,gr,cu);
            end
            if iseven(i)
                Knet.update!(w,cumgrads,opts)
                map(x->fill!(x,0f0),cumgrads)
                if wrun != nothing
                    for (wr,wi) in zip(wrun,w);axpy!(1.0f0-o[:ema],wi-wr,wr);end
                end
            end
        else
            preds  = forward_net(w,r,questions,batchSizes,xS,xB,xP;p=o[:p])
            cnt   += sum(preds.==answers)
            total += B
        end

        if i % 100 == 0
            toc();tic();
            println(@sprintf("%.2f%% Completed & %.2f Accuracy|Loss",
                             100i/L,train ? cnt/total:100cnt/total))
            flush(STDOUT)
        end
        if i % 2250 == 0
            savemodel(o[:prefix]*".jld",w,wrun,r,opts,o);
            info("model saved");flush(STDOUT);
            gc(); Knet.gc(); gc();
        end
    end
end

function train!(w,wrun,r,opts,sets,feats,o)
    info("Training Starts....")
    for i=1:o[:epochs]
        modelrun(w,r,opts,sets[1],feats[1],o,wrun;train=true)
        if iseven(i)
            modelrun(wrun,r,opts,sets[2],feats[2],o;train=false)
        end
    end
    return w,wrun,r,opts;
end

function train(sets,feats,o)
     if o[:mfile]==nothing
         w,r,_,_,_ = init_network(o);
         wrun      = deepcopy(w)
         opts      = optimizers(w,Adam;lr=o[:lr])
     else
         w,wrun,r,opts,p = loadmodel(o[:mfile])
     end
     train!(w,wrun,r,opts,sets,feats,o)
     return w,wrun,r,opts;
end


function train(dhome="data/",o=nothing)
     if o==nothing
         o=Dict(:h5=>false,:mfile=>nothing,:epochs=>10,
                :lr=>0.0001,:p=>12,:ema=>0.999f0,:batchsize=>32,
                :selfattn=>false,:gating=>false,:d=>512,
                :shuffle=>true,:sorted=>false,:prefix=>string(now())[1:10],
                :vocab_size=>90,:embed_size=>300, :dhome=>"data/", :loadresnet=>false)
     end
     feats,qdata,dics = loadTrainingData(dhome;h5=o[:h5])
     sets = []
     for q in qdata; push!(sets,miniBatch(q;shfle=o[:shuffle],srtd=o[:sorted])); end
     qdata = nothing; gc();
     w,wrun,r,opts = train(sets,feats,o)
     return w,wrun,r,opts,sets,feats,dics;
end

function validate(wrun,r,valset,valfeats,o)
     modelrun(wrun,r,nothing,valset,valfeats,o;train=false)
end

function validate(mfile,valset,valfeats,o)
     _,wrun,r,_ = loadmodel(mfile)
     modelrun(wrun,r,nothing,valset,valfeats;train=false)
     return wrun,r
end

function validate(mfile,dhome,o)
     _,wrun,r,_,o = loadmodel(mfile)
     valfeats     = loadFeatures(dhome,"val")
     qdata        = getQdata(dhome,"val")
     dics         = getDicts(dhome,"dic")
     valset       = miniBatch(qdata;shfle=o[:shuffle],srtd=o[:sorted])
     modelrun(wrun,r,nothing,valset,valfeats,o;train=false)
     return wrun,r,valset,valfeats
end


function singlerun(w,r,feat,question;p=12,selfattn=false,gating=false)
    results        = Dict{String,Any}("cnt"=>1)
    batchSizes     = ones(Int,length(question))
    forward_net(w,r,question,batchSizes,feat,xB,nothing;tap=results,p=p,selfattn=selfattn,gating=gating)
    prediction = indmax(results["y"])
    return results,prediction
end

function visualize(img,results;p=12)
    s_y,s_x = size(img)./14
    for k=1:p
        α = results["w_attn_$(k)"][:]
        println("step_$(k) most attn. wrds: ",i2w[question[sortperm(α;rev=true)[1:2]]])
        flush(STDOUT)
        display([RGB{N0f8}(α[i],α[i],α[i]) for i=1:length(α)]);
        hsvimg = convert.(HSV,img);
        attn = results["KB_attn_$(k)"]
        for i=1:14,j=1:14
            rngy          = floor(Int,(i-1)*s_y+1):floor(Int,min(i*s_y,320))
            rngx          = floor(Int,(j-1)*s_x+1):floor(Int,min(j*s_x,480))
            hsvimg[rngy,rngx]  = scalepixel.(hsvimg[rngy,rngx],attn[sub2ind((14,14),i,j)])
        end
        display(hsvimg)
    end
end

function scalepixel(pixel,scaler)
     return HSV(pixel.h,pixel.s,pixel.v+2*scaler)
end
