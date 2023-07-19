import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
import torch
from src.data.make_dataset import TripletVoxCeleb1ID 

@pytest.fixture
def mock_dataset():
    return [(torch.rand(10), i % 2) for i in range(100)]

def test_init_train(mock_dataset):
    dataset = TripletVoxCeleb1ID(mock_dataset, train=True)
    assert len(dataset.train_labels) == 100
    assert len(dataset.labels_set) == 2
    assert all(label in dataset.label_to_indices for label in dataset.labels_set)

def test_init_test(mock_dataset):
    dataset = TripletVoxCeleb1ID(mock_dataset, train=False)
    assert len(dataset.test_labels) == 100
    assert len(dataset.labels_set) == 2
    assert all(label in dataset.label_to_indices for label in dataset.labels_set)
    assert len(dataset.test_triplets) == 100

def test_getitem_train(mock_dataset):
    dataset = TripletVoxCeleb1ID(mock_dataset, train=True)
    for i in range(len(dataset)):
        (spec1, spec2, spec3), labels = dataset[i]
        assert spec1.shape == torch.Size([10])
        assert spec2.shape == torch.Size([10])
        assert spec3.shape == torch.Size([10])

def test_getitem_test(mock_dataset):
    dataset = TripletVoxCeleb1ID(mock_dataset, train=False)
    for i in range(len(dataset)):
        (spec1, spec2, spec3), labels = dataset[i]
        assert spec1.shape == torch.Size([10])
        assert spec2.shape == torch.Size([10])
        assert spec3.shape == torch.Size([10])

def test_len(mock_dataset):
    dataset = TripletVoxCeleb1ID(mock_dataset, train=True)
    assert len(dataset) == 100



def test_TripletVoxCeleb1ID_triplet_generation():
    # Mock dataset with 10 samples, each having an index as data and label
    mock_data = [(i, i%2) for i in range(10)]  # Labels are alternating 0 and 1
    
    # Initialize TripletVoxCeleb1ID with mock_data in test mode
    triplet_dataset = TripletVoxCeleb1ID(mock_data, train=False)
    
    # Test triplet generation
    for triplet in triplet_dataset.test_triplets:
        indices = triplet['indices']
        labels = triplet['labels']
        
        # Check if anchor and positive samples have the same label
        assert labels[0] == labels[1], "Anchor and positive samples do not have the same label"
        
        # Check if anchor and negative samples have different labels
        assert labels[0] != labels[2], "Anchor and negative samples have the same label"
        
        # Check if all indices in a triplet are unique
        assert len(set(indices)) == len(indices), "All indices in a triplet are not unique"



