-Abstract

This project is the open source code of the paper 'MExMI'. It can perform the model extraction attack on the local model.

=============
- The architecture of /data

data/imagenet-32/train/-train_data_batch_1
                  -train_data_batch_2
                  -train_data_batch_3
                  ...
download from imagenet web: https://image-net.org/download-images.
other datasets(CIFAR10, SVHN) can be downloaded automatically.

=============
-train victim model

mexmi/victim/train.py
train a victim model.
Victim path is config/VICTIM_PATH.

=============
-adjust hyperparamters

lr_scheduler, optimizer para in-> models/parser_params.py

=============
-perform MI Pre-Filter

mexmi/adversary/main.py
set sm_set 'membership_attack,{},{},{}'.format(membership_inference_algorithm,active_algorithm,remarks)

=============
-perform Post-Filter

mexmi/adversary/main.py
set following flags in mexmi/config.py
1) imp_vic_mem = True
2) vic_mem_method = 'shadow_model' or 'unsupervised'

=============
-perform semi-supervised boosting

mexmi/adversary/main_semi_supervised_boosting.py
This file is the step to perform semi-supervised boosting module after the attacker finished iteration training.
notice that the config file also impacts on this step.
1. change the num of initial seeds to the number of queried data.
2. change the num_iter to 1.