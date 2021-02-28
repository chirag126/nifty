# Codes for the paper "Towards a Unified Framework for Fair and Stable Graph Representation Learning".

1. Setup
This repository is built using PyTorch. You can install the necessary libraries by pip installing the requirements text file pip install -r ./requirements.txt The code was set up using python=3.7.9

2. Datasets
We ran our experiments on three highs-stakes read-world datasets. All the data are present in the './datasets' folder. Due to space constraints the edge file of the credit dataset is zipped.

3. Usage

Script 1: Evaluate fairness and stability performance of GCN (for German Graph dataset)
python nifty_sota_gnn.py --dropout 0.5 --hidden 16 --lr 1e-3 --epochs 1000 --model gcn --dataset german --seed 1

Script 2: Evaluate fairness and stability performance of NIFTY-GCN (for German Graph dataset)
python nifty_sota_gnn.py --drop_edge_rate_1 0.001 --drop_edge_rate_2 0.001 --drop_feature_rate_1 0.1 --drop_feature_rate_2 0.1 --dropout 0.5 --hidden 16 --lr 1e-3 --epochs 1000 --model ssf --encoder gcn --dataset german --sim_coeff 0.6 --seed 1

Script 3: Evaluate fairness and stability performance of FairGCN baseline (for German Graph dataset)
python baseline_fairGNN.py --dropout 0.5 --hidden 16 --lr 1e-3 --epochs 1000 --dataset german --seed 1 --model GCN

Script 4: Evaluate fairness and stability performance of RobustGCN (for German Graph dataset)
python nifty_sota_gnn.py --dropout 0.5 --hidden 16 --lr 1e-3 --epochs 1000 --model rogcn --dataset german --seed 1

