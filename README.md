# EC2Vec
EC2Vec is a machine learning tool that embeds Enzyme Commission (EC) numbers into vector representations.
## Dependencies
1. pytorch 1.10.0
3. numpy 1.19.2
4. sklearn 0.23.2
5. pandas 1.1.3
6. CUDA 11.1
## Input data to the model
EC2Vec takes raw EC numbers as input. 
The ```./Datasets/EC_numbers.csv``` file contains the EC numbers used for training the model.
## Get EC number embeddings using the trained model
The trained model embeds each EC number as a 1024-dim vector.
To get the EC number embeddings using the trained model, simply run ...
## Train your own EC2Vec 


## Remark
EC2Vec can process incomplete EC numbers, such as 3.4.25.-, 1.8.-.-, 6.-.-.-, and 0.0.0.0 (spontaneous reaction).
