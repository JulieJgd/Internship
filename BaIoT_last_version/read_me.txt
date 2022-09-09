Dataset:
https://archive.ics.uci.edu/ml/datasets/detection_of_IoT_botnet_attacks_N_BaIoT

GAN code:
https://github.com/inoubliwissem/labs/blob/main/GANs.ipynb

create_dfs_55_91.py
Creating two datasets: 
- one for training with 50% of attacks and 50% of begnin traffic
- one for testing with 10% of attacks and 90% of begnin traffic
The training dataset represents 70% of the used data and the testing dataset represents 30% of the used data

create_dfs_90_10.py
Creating a dataset with 90% of begnin traffic and 10% of attacks

create_dfs_balanced.py
Creating a dataset with 50% of attacks and 50% of begnin traffic

main_55_91.py
Using LinearSVC model

main_90_10.py
Using LinearSVC model
Using GAN, SMOTE, Oversampling and Undersampling to balanced training dataset

main_50_50.py
Using LinearSVC model
