# Compostional Attention Networks for Machine Reasoning

Knet implementation of the paper "[Compositional attention networks for machine reasoning](https://arxiv.org/abs/1803.03067)." Hudson, Drew A., and Christopher D. Manning.

## Installing Dependencies
```SHELL
julia requirements.jl
```
Patch your Knet/src/ with this [PR](https://github.com/CarloLucibello/Knet.jl/commit/91cac9a382028d1f4f6f02069c3574ff7fe1372a).


## Demo
### Getting Data
`demosetup` script downloads a pre-trained model and sample CLEVR data from our servers. It makes total 352MB download.
```SHELL
sh demosetup.sh
```
### Running Demo
Open `visualize.ipynb` notebook with jupyter to see the demo.
