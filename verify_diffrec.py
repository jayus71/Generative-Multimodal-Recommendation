import sys
import os
import torch
import numpy as np
import scipy.sparse as sp

# Add src to path to simulate running from src
sys.path.append(os.path.abspath('GenMMRec/src'))

from models.diffrec import DiffRec

# Mock Config
config = {
    'embedding_size': 32,
    'steps': 100,
    'noise_scale': 0.0001,
    'noise_min': 0.0001,
    'noise_max': 0.02,
    'dims': [64],
    'activation': 'relu',
    'dropout': 0.1,
    'device': 'cpu',
    'inference_steps': 10, # Start from step 10
    'USER_ID_FIELD': 'user_id',
    'ITEM_ID_FIELD': 'item_id',
    'NEG_PREFIX': 'neg_',
    'train_batch_size': 256,
    'is_multimodal_model': False,
    'end2end': False
}

# Mock Dataset
class MockDataset:
    def __init__(self):
        self.n_users = 10
        self.n_items = 20
        self.inter_matrix_val = sp.random(10, 20, density=0.5, format='coo')
        self.dataset = self
        
    def inter_matrix(self, form='coo'):
        return self.inter_matrix_val
        
    def get_user_num(self): return 10
    def get_item_num(self): return 20

dataset = MockDataset()

print("Initializing DiffRec...")
model = DiffRec(config, dataset)
print("DiffRec Initialized.")

# Test Forward (Calculate Loss)
print("Testing Calculate Loss...")
# interaction: [users, pos_items, neg_items]
# Note: DiffRec expects interaction to be a list/tuple where index 0 is users
interaction = [torch.tensor([0, 1]), torch.tensor([0, 1]), torch.tensor([1, 0])]
loss = model.calculate_loss(interaction)
print(f"Loss: {loss}")

# Test Predict
print("Testing Predict...")
scores = model.full_sort_predict(interaction)
print(f"Scores shape: {scores.shape}")
assert scores.shape == (2, 20)
print("Verification Passed.")
