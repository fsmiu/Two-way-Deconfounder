# Two-way-Deconfounder
The source code for the paper ‘Two-way Deconfounder for Off-policy Evaluation in Causal Reinforcement Learning,’ which has been accepted for publication at NeurIPS 2024, is available in this repository


## Run the Code
### Part 1: Generate simulation datasets

```
python sim_toy.py --d_seed 11 --d_number 1000 --e_degree 1.0 --c_degree 1.0
```

### Part 2: Generate the true value of the target policy using Monte Carlo methods

```
python MCTrue_toy.py --d_seed 11 --d_number 1000 --e_degree 1.0 --c_degree 1.0 --MC 10000
```

### Part 3: train model

```
python tune_toymodel.py --d_seed 11 --d_number 1000 --e_degree 1.0 --c_degree 1.0 --method TWD
```

### Part 3: Generate the estimated value of the target policy using the above trained model

```
python toymodel_eval.py --d_seed 11 --d_number 1000 --e_degree 1.0 --c_degree 1.0 --method TWD
```

##Contact 

I will continue to update the code over the next few days. please contract 24121534R@connect.polyu.hk if you have any questions
