# An Efficient Implementation of Kolmogorov-Arnold Network

This repository contains an efficient implementation of Kolmogorov-Arnold Network (KAN).
The original implementation of KAN is available [here](https://github.com/KindXiaoming/pykan).

The performance issue of the original implementation is mostly because it needs to expand all intermediate variables to perform the different activation functions.
For a layer with `in_features` input and `out_features` output, the original implementation needs to expand the input to a tensor with shape `(batch_size, out_features, in_features)` to perform the activation functions.
However, all activation functions are linear combination of a fixed set of basis functions which are B-splines; given that, we can reformulate the computation as activate the input with different basis functions and then combine them linearly.
This reformulation can significantly reduce the memory cost and make the computation a straightforward matrix multiplication, and works with both forward and backward pass naturally.

The problem is in the **sparsification** which is claimed to be critical to KAN's interpretability.
The authors proposed a L1 regularization defined on the input samples, which requires non-linear operations on the `(batch_size, out_features, in_features)` tensor, and is thus not compatible with the reformulation.
I instead replace the L1 regularization with a L1 regularization on the weights, which is more common in neural networks and is compatible with the reformulation.
The author's implementation indeed include this kind of regularization alongside the one described in the paper as well, so I think it might help.
More experiments are needed to verify this; but at least the original approach is infeasible if efficiency is wanted.

Another difference is that, beside the learnable activation functions (B-splines), the original implementation also includes a learnable scale on each activation function.
I provided an option `enable_standalone_scale_spline` that defaults to `True` to include this feature; disable it will make the model more efficient, but potentially hurts results.
It needs more experiments.


## MNIST Example

The results are presented below. I have managed to initiate convergence in the model, but I'm facing challenges in reducing the loss further. This issue will likely require some hyperparameter tuning. Adding more grids hasn't improved convergence. Moreover, using `update_grid` causes the loss to turn into `NaN`, necessitating further investigation.

```
Epoch 1/5: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:07<00:00, 121.49batch/s, loss=0.483]
Epoch 1/5, Train Loss: 0.3841, Test Loss: 0.0042, Test Accuracy: 92.23%
Epoch 2/5: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:08<00:00, 109.24batch/s, loss=0.176]
Epoch 2/5, Train Loss: 0.2526, Test Loss: 0.0040, Test Accuracy: 92.88%
Epoch 3/5: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:08<00:00, 114.90batch/s, loss=0.121]
Epoch 3/5, Train Loss: 0.2240, Test Loss: 0.0037, Test Accuracy: 93.62%
Epoch 4/5: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:08<00:00, 115.41batch/s, loss=0.168]
Epoch 4/5, Train Loss: 0.2085, Test Loss: 0.0038, Test Accuracy: 93.15%
Epoch 5/5: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 938/938 [00:07<00:00, 123.03batch/s, loss=0.0955]
Epoch 5/5, Train Loss: 0.1935, Test Loss: 0.0037, Test Accuracy: 93.04%
Parameter containing:
```
