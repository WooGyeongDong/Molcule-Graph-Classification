# Molcule Graph Classification

PyTorch implementation for Graph Classification via Graph Neural Networks

## Overview
This is code that performs graph classification using basic Graph Neural Networks(`GCN`, `GIN`, and `GAT`). 
The dataset is split into training/validation/test sets in an 8:1:1 ratio. Training is conducted for 100 epochs, with validation performed at every epoch. 
The model from the epoch that achieved the highest evaluation metric (ROC-AUC/Accuracy) on the validation set is evaluated on the test set.

## Note
**Here is the code for converting the MUTAG dataset into SMILES strings.**

Molecular data is converted into graphs using the RDKit library from SMILES strings. 
In the case of the MUTAG dataset, we generated SMILES strings from node and edge information of TUData. 
At this point, for data points 82 and 187, the provided edge information leads to invalid molecules. 
Therefore, we added a modification step to ensure that these data points become valid molecules.

## Experiments

```python main.py --model GCN --dataset MUTAG```

## Acknowledgements

The backbone implementation is reference to [https://github.com/yongduosui/CAL](https://github.com/yongduosui/CAL).
