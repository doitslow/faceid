# FACEID: learning face recognition/identification with neural network

## Structure of the code
1. [faceid]: key components for the learning task
   - train.py: invoke training and evaluation
   - infer.py: invoke inference of trained models
   - config.py: this is where all specifiable parameters are defined
2. [script]: shell scripts that runs defined tasks
3. [dataset_info]: optionally, one could put dataset information here in .json format

## Train
- Option1 : After acquiring of the datasets, one could simply invoke train.py to train a model. 
- Option2 : Invoke training through bash command: pls refer to script/93-train.sh.

## Highlights
1. competitive accuray/TPR/FPR with large training datasets (large number of identities, i.e. millions)
2. distributed data parallel training, enables switching among non-distributed / DP / DDP training effortlessly
3. ==distributed FC layer training: enables training multiple datasets simultaneously==