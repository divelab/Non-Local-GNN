# Non-Local Graph Neural Networks

This is the official implementation of the non-local GNNs proposed in the following paper.

Meng Liu*, Zhengyang Wang*, and Shuiwang Ji. "[Non-Local Graph Neural Networks](https://ieeexplore.ieee.org/document/9645300)". [TPAMI]

![](https://github.com/divelab/Non-Local-GNN/blob/main/assets/NLGNN.png)

## Requirements
We include key dependencies below. The versions we used are in the parentheses.
* PyTorch (1.10.1)
* PyTorch Geometric (1.6.3)

## Run
```linux
cd NLGNN/src
bash longrange.sh
```

## Reference
```
@article{liu2021non,
  title={Non-local graph neural networks},
  author={Liu, Meng and Wang, Zhengyang and Ji, Shuiwang},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2021},
  publisher={IEEE}
}
```
