server="ai.ku.edu.tr/"
info("Downloading sample questions and images from CLEVR dataset...")
download(server*"data/mac-network/demo.tgz")
run(`tar -xvfz demo.tgz`)
rm("demo.tgz")
info("Downloading pre-trained model from our servers...")
download(server*"models/mac-network/demo_model.jld","models/demo.jld")
