# XFLT

This repository contains the code for cross lingual fact to long text generation. The code is organized as follows - 

1. ```clustering``` - This contains the code for training the fact organisation model
    - ```mT5-baseline``` - End-to-end clustering 
    - ```statistical_clustering``` - Statistical spectral clustering 
2. ```dataset_prep``` - This contains code for data preprocessing 
    - ```coverage_classifier``` - Code and data for training coverage prompt classifier 
3. ```eval_module``` - This contains the code for running evaluation using NLG metrics and the defined X-PARENT metrics. 
4. ```generation``` - This contains code for training models using different methods 
    - ```mT5-baseline``` - Training baseline mT5 method
    - ```prompt_uni``` - Training with coverage prompt 
    - ```grounded_decoding``` - Inference with grounded decoding. Requires installing the modified ```transformers``` package included in the directory 
5. ```rl_msme``` - This contains code for training with RL rewards  

The default hyperparameter settings can be found in the ```run``` bash files in the respective directories. 
The requirements for all methods in the ```generation``` directory can be found in ```generation_reqs.txt```. The same for RL can be found in ```rl_reqs.txt```. 