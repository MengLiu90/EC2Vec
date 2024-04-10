import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class ECDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.data[['POS_1', 'POS_2', 'POS_3', 'POS_4']] = self.data['EC_number'].str.split('.', expand=True)

        words_1th_pos = self.data['POS_1'].unique().tolist()
        word_to_index = {word: index for index, word in enumerate(words_1th_pos)}
        self.data['POS_1_index'] = self.data['POS_1'].map(word_to_index)

        words_2th_pos = self.data['POS_2'].unique().tolist()
        word_to_index = {word: index for index, word in enumerate(words_2th_pos)}
        self.data['POS_2_index'] = self.data['POS_2'].map(word_to_index)

        words_3th_pos = self.data['POS_3'].unique().tolist()
        word_to_index = {word: index for index, word in enumerate(words_3th_pos)}
        self.data['POS_3_index'] = self.data['POS_3'].map(word_to_index)

        words_4th_pos = self.data['POS_4'].unique().tolist()
        word_to_index = {word: index for index, word in enumerate(words_4th_pos)}
        self.data['POS_4_index'] = self.data['POS_4'].map(word_to_index)

        self.data.to_csv('Datasets/processed_EC_data.csv', index=False)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {
            'x1': torch.tensor(self.data.iloc[idx, -4].astype(np.int64), dtype=torch.long),
            'x2': torch.tensor(self.data.iloc[idx, -3].astype(np.int64), dtype=torch.long),
            'x3': torch.tensor(self.data.iloc[idx, -2].astype(np.int64), dtype=torch.long),
            'x4': torch.tensor(self.data.iloc[idx, -1].astype(np.int64), dtype=torch.long)
        }
        return sample

# dataset = ECDataset('Datasets/EC_numbers.csv')
# # print(len(dataset))
# for i in range(5):
#     sample = dataset[i]
#     print(f'Sample {i + 1}:')
#     print('x1:', sample['x1'])
#     print('x2:', sample['x2'])
#     print('x3:', sample['x3'])
#     print('x4:', sample['x4'])
