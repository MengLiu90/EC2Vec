# EC2Vec
EC2Vec is a machine learning tool that embeds Enzyme Commission (EC) numbers into vector representations.
## Dependencies
1. pytorch 1.10.0
3. numpy 1.19.2
4. sklearn 0.23.2
5. pandas 1.1.3

## Input data to the model
EC2Vec takes raw EC numbers as input. 
The ```./Datasets/EC_numbers.csv``` file contains the EC numbers used for training the model.
## Get EC number embeddings using the trained model
The trained model embeds each EC number as a 1024-dim vector.

To get the EC number embeddings using the trained model, put your EC number data under ```./Datasets/``` directory, please follow ```./Datasets/EC_numbers.csv``` for the format. 

Then simply run ```get_ec2vec_embeddings.py```. 

The generated embedding file will be saved under ```./Embedding_Results/``` directory as ```embedded_EC_number.csv``` file.
## Train your own EC2Vec 
To train the EC2Vec model using your own data, put your EC number data under ```./Datasets/``` directory, please follow ```./Datasets/EC_numbers.csv``` for the format. 

Then simply run ```ec2vec.py```.

The trained model based on your data will be saved under ```./Trained_model/``` directory as ```model.pth``` file.

Note that, we used 1024 as the embedding size for an EC number. You can adjust this dimension by changing the ```hidden_sizes``` parameter in the code.

## Remark
EC2Vec can process incomplete EC numbers, such as 3.4.25.-, 1.8.-.-, 6.-.-.-, and 0.0.0.0 (spontaneous reaction).
