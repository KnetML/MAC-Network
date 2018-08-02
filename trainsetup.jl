if length(ARGS)==0
    error("clevr home folder is not specified!")
else
    CLEVR_HOME = ARGS[1]
end

include("preprocess.jl")
info("Starting to preprocess CLEVR questions @: $(clevrhome)/questions ")
info("Training and validation questions will be processed...")
if isfile("data/train.json") && isfile("data/val.json")
    info("You already had processed questions in your data folder. Do you want to re-process the data? (yes or no)")
    readline() == "yes" && preprocess([CLEVR_HOME])
else
    preprocess([CLEVR_HOME])
end

include("extract_features.jl")
info("Starting to preprocess CLEVR images @: $(clevrhome)/images ")
info("Training and validation images will be processed...")

if isfile("data/train.bin") && isfile("data/val.bin")
    info("You already had processed images in your data folder. Do you want to re-process the data? (yes or no)")
    readline() == "yes" && extract_features([CLEVR_HOME])
else
    extract_features([CLEVR_HOME])
end
