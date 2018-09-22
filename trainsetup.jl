using Pkg; Pkg.activate("."); Pkg.instantiate(); #install required packages
server = "ai.ku.edu.tr/"
if length(ARGS)==0
    error("clevr home folder is not specified. Pre-processed data will be downloaded from the servers(70GB)? (yes or no)")
    if readline() == "yes"
        !isfile("data/train.bin")  && download(server*"data/mac-network/train.bin","data/train.bin")
        !isfile("data/val.bin")    && download(server*"data/mac-network/val.bin","data/val.bin")
        !isfile("data/train.json") && download(server*"data/mac-network/train.json","data/train.json")
        !isfile("data/val.json")   && download(server*"data/mac-network/val.json","data/val.json")
        !isfile("data/dic.json")   && download(server*"data/mac-network/dic.json","data/dic.json")
    else
        error("No data available!")
    end
else
    CLEVR_HOME = ARGS[1]
end

println(CLEVR_HOME)

println("Checking dependencies...")
include("requirements.jl")

include("preprocess.jl")
println("Starting to preprocess CLEVR questions @: $(CLEVR_HOME)/questions ")
println("Training and validation questions will be processed...")
if isfile("data/train.json") && isfile("data/val.json")
    println("You already had processed questions in your data folder. Do you want to re-process the data? (yes or no)")
    readline() == "yes" && preprocess([CLEVR_HOME])
else
    preprocess([CLEVR_HOME])
end

include("extract_features.jl")
println("Starting to preprocess CLEVR images @: $(CLEVR_HOME)/images ")
println("Training and validation images will be processed...")

if isfile("data/train.bin") && isfile("data/val.bin")
    println("You already had processed images in your data folder. Do you want to re-process the data? (yes or no)")
    readline() == "yes" && extract_features([CLEVR_HOME])
else
    extract_features([CLEVR_HOME])
end
