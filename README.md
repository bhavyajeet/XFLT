# XFLT - Exploring Techniques for Generating Cross Lingual Factually Grounded Long Text

This repository contains the code and the processed dataset for cross lingual fact to long text generation. The paper describing the methods has been accepted at the European Conference on Artificial Intelligence (ECAI 2023). 


## Dataset

The processed `XLAlign` dataset is present in the `XLAlign-Dataset` directory. The directory contains the subdirectories for each of the languages. 

## Code 
The code is present within the `XFLT-code` directory and is organised as follows.

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


## Contributors
<li> Bhavyajeet Singh
<li> Aditya Hari 
<li> Rahul Mehta
<li> Tushar Abhishek   
<li> Manish Gupta
<li> Vasudeva Varma

