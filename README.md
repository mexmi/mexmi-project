- Abstract  
  
This project is the open source code of the paper 'MExMI: Pool-based Active Model Extraction Crossover Membership Inference'. It can perform the model extraction attack on the local model as weel as the cloud model (The target cloud model needs to be deployed by the reader.).

=============  
- Requirements  
python==3.7  
cudatoolkit==10.1  
theano==1.0.4  
lasagne==0.1  
tensorboardX  
setproctitle==1.1.10  
  
if there is an error related to file 'layer.pool.pool2d.py'  
1) replace 'from theano.tensor.signal import downsample' by 'from theano.tensor.signal import pool'. 
2) replace all 'downsample.max_pool_2d' with 'pool.pool_2d'. 

=============  
- Dataset preparation  
There should be a 'data' folder in parallel with 'mexmi' folder. Please create one before start the experiments.
The dataset imagenet32 needs to be downloaded and has the following structure:

data/imagenet-32/train/train_data_batch_1  
                       train_data_batch_2  
                       train_data_batch_3  
                       ...  
It can be download from imagenet web: https://image-net.org/download-images.  
Other datasets(CIFAR10, SVHN etc.) can be downloaded automatically.

=============  
- Victim model preperation: (1) Train victim model  

Main file: mexmi/victim/train.py  
The pyperparameters can be set in this file.  
Victim path is config/VICTIM_PATH.  

============
- Parameter files  
  
Custom parameters are in the params variable in mexmi/adversary/main.py, and in the config.py file.  

=============  
- Perform MI Pre-Filter  
  
Main file: mexmi/adversary/main.py  
Set sm_set 'membership_attack,{},{},{}'.format(membership_inference_algorithm,active_algorithm,remarks)  

=============  
- Perform Post-Filter  
  
Main file: mexmi/adversary/main.py  
Set following flags in mexmi/config.py:
1. imp_vic_mem = True  
2. vic_mem_method = 'shadow_model' or 'unsupervised'  
  
=============  
- Perform semi-supervised boosting  

Main file: mexmi/adversary/main_semi_supervised_boosting.py  
This file is the step to perform semi-supervised boosting module after the attacker finished iteration training.  
notice that the config file also impacts on this step.  
1. change the num of initial seeds to the number of queried data.  
2. change the num_iter to 2.  
