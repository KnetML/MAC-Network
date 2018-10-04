using Knet,ProgressMeter
include(Knet.dir("data/imagenet.jl"))
include(Knet.dir("examples/resnet/resnet.jl"))

function loadparams(atype)
    mname       = "imagenet-resnet-101-dag"
    model       = matconvnet(mname)
    avgimg      = Array{Float32}(model["meta"]["normalization"]["averageImage"])
    w, ms       = ResNet.get_params(model["params"], atype)
    deleteat!(w,283:length(w))
    deleteat!(ms,189:length(ms))
    return w,ms,avgimg
end

function resnet101(w,x,ms; mode=1)
    # layer 1
    conv1 = ResNet.reslayerx1(w[1:3],x,ms; padding=3, stride=2, mode=mode)
    pool1 = ResNet.pool(conv1; window=3, stride=2)
    # layer 2,3
    r2 = ResNet.reslayerx5(w[4:33], pool1, ms; strides=[1,1,1,1], mode=mode)
    r3 = ResNet.reslayerx5(w[34:72], r2, ms; mode=mode)
    return ResNet.reslayerx5(w[73:282], r3, ms; mode=mode)
end

function imgdata(img,averageImage)
    a0 = RGB.(load(img))
    new_size = ntuple(i->div(size(a0,i)*224,minimum(size(a0))),2)
    a1 = Images.imresize(a0, new_size)
    i1 = div(size(a1,1)-224,2)
    j1 = div(size(a1,2)-224,2)
    b1 = a1[i1+1:i1+224,j1+1:j1+224]
    c1 = permutedims(channelview(b1),(3,2,1))
    d1 = convert(Array{Float32},c1)
    e1 = reshape(d1, (224,224,3,1))
    f1 = (255 * e1 .- averageImage)
    g1 = permutedims(f1, [2,1,3,4])
end

function extract(dhome,set,params;atype=Array{Float32})
    w,ms,avgimg = params
    sethome = joinpath(dhome,"images",set)*"/"
    binfile = "data/"*set*".bin"
    files   = readdir(sethome)
    p = Progress(length(files))
    for file in files
        if file[end-2:end] == "png"
            x = convert(atype,imgdata(sethome*file,avgimg))
            y = convert(Array{Float32},resnet101(w,x,ms; mode=1))
            open(binfile,"a+") do f
                write(f,y)
            end
        end
        next!(p)
    end
end

function extract_features(dhome)
    atype = gpu()<0 ? Array{Float32} : KnetArray{Float32}
    params = loadparams(atype)
    println("Extracting training features")
    extract(dhome,"train",params;atype=atype)
    println("Extracting validation features")
    extract(dhome,"val",params;atype=atype)
end
