## Towards a Unified Framework for Fair and Stable Graph Representation Learning

This repository contains source code necessary to reproduce some of the main results in [the paper]():

**If you use this software, please consider citing:**
    
    @inproceedings{agarwal2021towards,
      title={Towards a Unified Framework for Fair and Stable Graph Representation Learning},
      author={Agarwal, Chirag, Lakkaraju, Himabindu, and Zitnik, Marinka},
      year={2021},
      booktitle={arXiv},
    }

<p align="center">
    <img src="revised_proposed_model.png" width=750px>
</p>
<p align="center"><i>
  Our framework NIFTY can learn node representations that are both fair and stable (i.e., invariant to the sensitive attribute value and perturbations to the graph   structure and non-sensitive attributes) by maximizing the similarity between representations from diverse augmented graphs.  
</i></p>

## 1. Setup

### Installing software
This repository is built using PyTorch. You can install the necessary libraries by pip installing the requirements text file `pip install -r ./requirements.txt`

**Note:** We ran our codes using python=3.7.9


## 2. Datasets
We ran our experiments on three highs-stakes read-world datasets. All the data are present in the './datasets' folder. Due to space constraints the edge file of the credit dataset is zipped.

## 3. Usage
The main scripts running the experiments on the state-of-the-art GNNs and their NIFTY-augmented counterparts is in [nifty_sota_gnn.py](nifty_sota_gnn.py)

### Examples
Script 1: Evaluate fairness and stability performance of GCN (for German Graph dataset)
`python nifty_sota_gnn.py --dropout 0.5 --hidden 16 --lr 1e-3 --epochs 1000 --model gcn --dataset german --seed 1`
<p align="left"><i>
  The AUCROC of estimator: 0.7605<br/>
  Parity: 0.395253682487725 | Equality: 0.2731092436974789<br/>
  F1-score: 0.807799442896936<br/>
  CounterFactual Fairness: 0.29600000000000004<br/>
  Robustness Score: 0.11599999999999999<br/>
</i></p>

Script 2: Evaluate fairness and stability performance of NIFTY-GCN (for German Graph dataset)
`python nifty_sota_gnn.py --drop_edge_rate_1 0.001 --drop_edge_rate_2 0.001 --drop_feature_rate_1 0.1 --drop_feature_rate_2 0.1 --dropout 0.5 --hidden 16 --lr 1e-3 --epochs 1000 --model ssf --encoder gcn --dataset german --sim_coeff 0.6 --seed 1`
<p align="left"><i>
  The AUCROC of estimator: 0.7205<br/>
  Parity: 0.010365521003818934 | Equality: 0.01995798319327735<br/>
  F1-score: 0.823529411764706<br/>
  CounterFactual Fairness: 0.0<br/>
  Robustness Score: 0.0<br/>
</i></p>  

Script 3: Evaluate fairness and stability performance of FairGCN baseline (for German Graph dataset)
`python baseline_fairGNN.py --dropout 0.5 --hidden 16 --lr 1e-3 --epochs 1000 --dataset german --seed 1 --model gcn`
<p align="left"><i>
  The AUCROC of estimator: 0.7549<br/>
  Parity: 0.27632296781232957 | Equality: 0.17226890756302526<br/>
  F1-score: 0.825065274151436<br/>
  CounterFactual Fairness: N/A<br/>
  Robustness Score: 0.04400000000000004<br/>
</i></p>   

Script 4: Evaluate fairness and stability performance of RobustGCN (for German Graph dataset)
`python nifty_sota_gnn.py --dropout 0.5 --hidden 16 --lr 1e-3 --epochs 1000 --model rogcn --dataset german --seed 5`
<p align="left"><i>
  The AUCROC of estimator: 0.6230<br/>
  Parity: 0.24495362793235131 | Equality: 0.2048319327731093<br/>
  F1-score: 0.614334470989761<br/>
  CounterFactual Fairness: 0.08799999999999997<br/>
  Robustness Score: 0.132<br/>
</i></p>  

## 4. Licenses
Note that the code in this repository is licensed under MIT License. Please carefully check them before use. 

## 5. Questions?
If you have questions/suggestions, please feel free to [email](mailto:chiragagarwall12@gmail.com) or create github issues.
