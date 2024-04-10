import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from EC_dataset_process import ECDataset
import pandas as pd
import os

class MultimodalAutoencoder(nn.Module):
    def __init__(self, input_sizes, hidden_sizes):
        super(MultimodalAutoencoder, self).__init__()
        assert len(input_sizes) == len(hidden_sizes), "Number of input sizes must match number of hidden sizes"
        self.input_sizes = input_sizes
        self.hidden_sizes = hidden_sizes

        # Embedding layer for the first input x1
        self.embedding_x1 = nn.Embedding(input_sizes[0], hidden_sizes[0])
        # Embedding layer for the second input x2
        self.embedding_x2 = nn.Embedding(input_sizes[1], hidden_sizes[1])
        # Embedding layer for the third input x3
        self.embedding_x3 = nn.Embedding(input_sizes[2], hidden_sizes[2])
        # Embedding layer for the fourth input x4
        self.embedding_x4 = nn.Embedding(input_sizes[3], hidden_sizes[3])

        # Inner encoder (1D convolution)
        self.conv1d = nn.Conv1d(sum(hidden_sizes), max(hidden_sizes), kernel_size=3, padding=1)

        # Decoder (symmetric to encoder)
        self.deconv1d = nn.ConvTranspose1d(max(hidden_sizes), sum(hidden_sizes), kernel_size=3,
                                           padding=1)

        # Decoder embedding layer for x1 reconstruction
        self.embedding_recon_x1 = nn.Linear(hidden_sizes[0], input_sizes[0])
        # Decoder embedding layer for x2 reconstruction
        self.embedding_recon_x2 = nn.Linear(hidden_sizes[1], input_sizes[1])
        # Decoder embedding layer for x3 reconstruction
        self.embedding_recon_x3 = nn.Linear(hidden_sizes[2], input_sizes[2])
        # Decoder embedding layer for x4 reconstruction
        self.embedding_recon_x4 = nn.Linear(hidden_sizes[3], input_sizes[3])

    def forward(self, *inputs):
        assert len(inputs) == len(self.input_sizes), "Number of inputs must match number of input sizes"

        # Initial embedding for each input (x1, x2, x3, x4)
        # Embedding for x1
        embedded_x1 = self.embedding_x1(inputs[0].long())
        # Embedding for x2
        embedded_x2 = self.embedding_x2(inputs[1].long())
        # Embedding for x3
        embedded_x3 = self.embedding_x3(inputs[2].long())
        # Embedding for x4
        embedded_x4 = self.embedding_x4(inputs[3].long())

        # Concatenate embeddings
        x = torch.cat([embedded_x1, embedded_x2, embedded_x3, embedded_x4], dim=1)  # Concatenate along dimension 2

        # Inner encoder
        encoder_embeddings = F.relu(self.conv1d(x.unsqueeze(2)).squeeze(2))

        # Decoder
        x = F.relu(self.deconv1d(encoder_embeddings.unsqueeze(2)).squeeze(2))

        # Decoder embedding for x1 reconstruction
        reconstructed_x1 = self.embedding_recon_x1(x[:, -self.hidden_sizes[0]:])
        # Decoder embedding for x2 reconstruction
        reconstructed_x2 = self.embedding_recon_x2(x[:, -self.hidden_sizes[1]:])
        # Decoder embedding for x3 reconstruction
        reconstructed_x3 = self.embedding_recon_x3(x[:, -self.hidden_sizes[2]:])
        # Decoder embedding for x4 reconstruction
        reconstructed_x4 = self.embedding_recon_x4(x[:, -self.hidden_sizes[3]:])

        return tuple([reconstructed_x1, reconstructed_x2, reconstructed_x3, reconstructed_x4]), encoder_embeddings

print('Getting embeddings...')
dataset = ECDataset('Datasets/EC_numbers.csv')
df_ = pd.read_csv('Datasets/EC_numbers.csv')
df_[['POS_1', 'POS_2', 'POS_3', 'POS_4']] = df_['EC_number'].str.split('.', expand=True)
# number of unique characters in each position of the EC numbers
input_sizes = [len(df_['POS_1'].unique()), len(df_['POS_2'].unique()),
               len(df_['POS_3'].unique()), len(df_['POS_4'].unique())]
hidden_sizes = [16, 64, 64, 1024]

model = MultimodalAutoencoder(input_sizes, hidden_sizes)
model.load_state_dict(torch.load("Trained_model/final_model.pth"))
model.eval()

save_dir = 'Embedding_Results'
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

criterion = nn.CrossEntropyLoss()
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

all_data = []
x1_labels = []
x1_preds = []
x2_labels = []
x2_preds = []
x3_labels = []
x3_preds = []
x4_labels = []
x4_preds = []

with torch.no_grad():
    for data in data_loader:
        x1, x2, x3, x4 = data['x1'], data['x2'], data['x3'], data['x4']
        outputs, encoder_embeddings = model(x1, x2, x3, x4)
        x1_recon, x2_recon, x3_recon, x4_recon = outputs

        x1_pred = x1_recon.data.cpu().numpy().argmax(axis=1)
        x1_label = x1.tolist()
        x1_labels += x1_label
        x1_preds += x1_pred.tolist()

        x2_pred = x2_recon.data.cpu().numpy().argmax(axis=1)
        x2_label = x2.tolist()
        x2_labels += x2_label
        x2_preds += x2_pred.tolist()

        x3_pred = x3_recon.data.cpu().numpy().argmax(axis=1)
        x3_label = x3.tolist()
        x3_labels += x3_label
        x3_preds += x3_pred.tolist()

        x4_pred = x4_recon.data.cpu().numpy().argmax(axis=1)
        x4_label = x4.tolist()
        x4_labels += x4_label
        x4_preds += x4_pred.tolist()

        # Convert tensors to numpy arrays
        x1_array = x1.numpy().reshape(-1)
        x2_array = x2.numpy().reshape(-1)
        x3_array = x3.numpy().reshape(-1)
        x4_array = x4.numpy().reshape(-1)
        embeddings_array = encoder_embeddings.numpy().reshape(-1)

        # Combine input data and embeddings into a single row
        row = list(x1_array) + list(x2_array) + list(x3_array) + list(x4_array) + list(embeddings_array)
        all_data.append(row)


# Define column names for the CSV file
column_names = ['x1_col_{}'.format(i) for i in range(x1_array.size)] + \
               ['x2_col_{}'.format(i) for i in range(x2_array.size)] + \
               ['x3_col_{}'.format(i) for i in range(x3_array.size)] + \
               ['x4_col_{}'.format(i) for i in range(x4_array.size)] + \
               ['ec2vec_{}'.format(i) for i in range(embeddings_array.size)]

# Convert the data to a DataFrame
df = pd.DataFrame(all_data, columns=column_names)

df_index = pd.read_csv('Datasets/processed_EC_data.csv')
for i in range(4):
    # Create a dictionary mapping indices to words for dictionary 1, 2, 3, 4
    index_to_word_dict = dict(zip(df_index[f'POS_{i+1}_index'], df_index[f'POS_{i+1}']))
    # Map indices to words for 'POS_i+1' in df
    df[f'x_{i+1}_word'] = df[f'x{i+1}_col_0'].map(index_to_word_dict)

# Convert columns to string type
df['x_1_word'] = df['x_1_word'].astype(str)
df['x_2_word'] = df['x_2_word'].astype(str)
df['x_3_word'] = df['x_3_word'].astype(str)
df['x_4_word'] = df['x_4_word'].astype(str)
df['EC_number'] = df['x_1_word'] + '.' + df['x_2_word'] + '.' + df['x_3_word'] + '.' + df['x_4_word']
df.drop(columns=['x1_col_0', 'x2_col_0', 'x3_col_0', 'x4_col_0', 'x_1_word', 'x_2_word', 'x_3_word', 'x_4_word'],
        inplace=True)
embedding_cols = ['ec2vec_{}'.format(i) for i in range(1024)]
df = df[['EC_number'] + embedding_cols]
df.to_csv(os.path.join(save_dir, 'embedded_EC_number.csv'), index=False)
print('Done!')

