import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from EC_dataset_process import ECDataset
from sklearn.model_selection import train_test_split
import random
from sklearn.metrics import accuracy_score
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
        x = torch.cat([embedded_x1, embedded_x2, embedded_x3, embedded_x4], dim=1)

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

        return tuple([reconstructed_x1, reconstructed_x2, reconstructed_x3, reconstructed_x4])

dataset = ECDataset('Datasets/EC_numbers.csv')
df_ = pd.read_csv('Datasets/EC_numbers.csv')
df_[['POS_1', 'POS_2', 'POS_3', 'POS_4']] = df_['EC_number'].str.split('.', expand=True)
# number of unique characters in each position of the EC numbers
input_sizes = [len(df_['POS_1'].unique()), len(df_['POS_2'].unique()),
               len(df_['POS_3'].unique()), len(df_['POS_4'].unique())]
hidden_sizes = [16, 64, 64, 1024]
model = MultimodalAutoencoder(input_sizes, hidden_sizes)

save_dir = 'Trained_model'
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
# Define batch size and number of workers for data loading
batch_size = 2
num_epochs = 200
seed = random.randint(1, 500)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.000001, weight_decay=0.00001)
# Split dataset into train and test sets
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

# Create DataLoader for train and test sets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Training loop
for epoch in range(num_epochs):
    running_loss = 0.0
    x1_acc_tr = 0
    x2_acc_tr = 0
    x3_acc_tr = 0
    x4_acc_tr = 0
    model.train()
    # Iterate over the training dataset
    for i, data in enumerate(train_loader, 0):
        x1, x2, x3, x4 = data['x1'], data['x2'], data['x3'], data['x4']
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        x1_recon, x2_recon, x3_recon, x4_recon = model(x1, x2, x3, x4)

        # Compute the loss
        loss_x1 = criterion(x1_recon, x1)
        loss_x2 = criterion(x2_recon, x2)
        loss_x3 = criterion(x3_recon, x3)
        loss_x4 = criterion(x4_recon, x4)
        loss = loss_x1 + loss_x2 + loss_x3 + loss_x4

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()

        x1_pred = x1_recon.data.cpu().numpy().argmax(axis=1)
        x1_acc_tr += accuracy_score(x1, x1_pred)
        x2_pred = x2_recon.data.cpu().numpy().argmax(axis=1)
        x2_acc_tr += accuracy_score(x2, x2_pred)
        x3_pred = x3_recon.data.cpu().numpy().argmax(axis=1)
        x3_acc_tr += accuracy_score(x3, x3_pred)
        x4_pred = x4_recon.data.cpu().numpy().argmax(axis=1)
        x4_acc_tr += accuracy_score(x4, x4_pred)
    print('[%d] train loss: %.3f' % (epoch + 1, running_loss / len(train_loader)))

    # After training, evaluate the model on the test set
    test_loss = 0.0
    x1_acc = 0
    x2_acc = 0
    x3_acc = 0
    x4_acc = 0

    x1_labels = []
    x1_preds = []
    x2_labels = []
    x2_preds = []
    x3_labels = []
    x3_preds = []
    x4_labels = []
    x4_preds = []

    model.eval()
    with torch.no_grad():
        for data in test_loader:
            x1, x2, x3, x4 = data['x1'], data['x2'], data['x3'], data['x4']
            x1_recon, x2_recon, x3_recon, x4_recon = model(x1, x2, x3, x4)

            loss_x1 = criterion(x1_recon, x1)
            loss_x2 = criterion(x2_recon, x2)
            loss_x3 = criterion(x3_recon, x3)
            loss_x4 = criterion(x4_recon, x4)
            test_loss += (loss_x1 + loss_x2 + loss_x3 + loss_x4).item()

            x1_pred = x1_recon.data.cpu().numpy().argmax(axis=1)
            x1_acc += accuracy_score(x1, x1_pred)
            x1_label = x1.tolist()
            x1_labels += x1_label
            x1_preds += x1_pred.tolist()

            x2_pred = x2_recon.data.cpu().numpy().argmax(axis=1)
            x2_acc += accuracy_score(x2, x2_pred)
            x2_label = x2.tolist()
            x2_labels += x2_label
            x2_preds += x2_pred.tolist()

            x3_pred = x3_recon.data.cpu().numpy().argmax(axis=1)
            x3_acc += accuracy_score(x3, x3_pred)
            x3_label = x3.tolist()
            x3_labels += x3_label
            x3_preds += x3_pred.tolist()

            x4_pred = x4_recon.data.cpu().numpy().argmax(axis=1)
            x4_acc += accuracy_score(x4, x4_pred)
            x4_label = x4.tolist()
            x4_labels += x4_label
            x4_preds += x4_pred.tolist()

    print('[%d] test loss: %.3f' % (epoch + 1, test_loss / len(test_loader)))

    # Save the model's state dictionary at the end of the last epoch
    if epoch == num_epochs - 1:
        # Define the file path where you want to save the model
        model_file = os.path.join(save_dir, "model.pth")

        # Save the model's state dictionary
        torch.save(model.state_dict(), model_file)

        print("Model saved at the end of the last epoch:", model_file)
