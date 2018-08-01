server="ai.ku.edu.tr/"
info("Downloading sample questions and images from CLEVR dataset...")
download(server*"data/mac-network/demo.tar.gz","demo.tar.gz")
run(`tar -xzf demo.tar.gz`)
rm("demo.tar.gz")
info("Downloading pre-trained model from our servers...")
download(server*"models/mac-network/demo_model.jld","models/demo.jld")
