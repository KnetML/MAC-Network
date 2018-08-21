# Compostional Attention Networks for Machine Reasoning

Knet implementation of the paper "[Compositional attention networks for machine reasoning](https://arxiv.org/abs/1803.03067)." Hudson, Drew A., and Christopher D. Manning.

### Running Demo
Open `visualize.ipynb` notebook with jupyter to see the demo.

## Getting Data

You have two options for the data setup.

### a) Raw Data
1-Download [CLEVR](https://cs.stanford.edu/people/jcjohns/clevr/) dataset to `data/` folder.

2-Process the CLEVR data:
```SHELL
julia trainsetup.jl data/CLEVR_v1.0
```

### b) Processed Data

1-Download preprocessed data:
```SHELL
julia trainsetup.jl
```

## Training

Below configurations can achieve %98.65 accuracy on CLEVR dataset
```SHELL
julia train.jl src/macnet.jl configs/config2.jl
```

For quick training you can use below command:
```SHELL
julia train.jl src/macnet-py.jl configs/config1.jl
```
