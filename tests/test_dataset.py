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
