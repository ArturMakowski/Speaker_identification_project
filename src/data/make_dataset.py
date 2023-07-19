import torch
import torchaudio
from torch.utils.data import Subset, Dataset
import numpy as np
import json



class VoxCeleb1IdentificationUnified(Dataset):
    
    def __init__(self, voxceleb1_dataset, present_audio_files = []):
        self.voxceleb1_dataset = voxceleb1_dataset
        self.voxceleb1_dataset = Subset(self.voxceleb1_dataset, present_audio_files)
        
        self.num_samples_per_clip = 40000   
        self.mel_spectrogram_transformation = torchaudio.transforms.MelSpectrogram(
            sample_rate=4000,
            n_fft=1024,
            hop_length=512,
            n_mels=64)

    def __getitem__(self, idx):
        waveform, _, target, _ = self.voxceleb1_dataset[idx]
        waveform = torchaudio.transforms.Resample(16000, 4000)(waveform)
        waveform = self._right_zero_pad(self._cut_if_necessary(waveform))
        mel_spec = self.mel_spectrogram_transformation(waveform)
        return mel_spec, target
    
    def __len__(self):
        return len(self.voxceleb1_dataset)
    
    def _right_zero_pad(self, signal):
      length_signal = signal.shape[1]
      if length_signal < self.num_samples_per_clip:
          num_missing_samples = self.num_samples_per_clip - length_signal
          signal = torch.nn.functional.pad(signal, (0, num_missing_samples))
      return signal
    
    def _cut_if_necessary(self, signal):
      if signal.shape[1] > self.num_samples_per_clip:
          signal = signal[:, :self.num_samples_per_clip]
      return signal
  
  
class TripletVoxCeleb1ID(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, voxceleb1_dataset, train=True):
        self.voxceleb1_dataset = voxceleb1_dataset
        self.train = train

        if self.train:
            self.train_labels = torch.tensor([self.voxceleb1_dataset[i][1]
                                              for i in range(len(self.voxceleb1_dataset))])
            
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}

        else:
            self.test_labels = torch.tensor([self.voxceleb1_dataset[i][1] 
                                              for i in range(len(self.voxceleb1_dataset))])
        
            # generate fixed triplets for testing
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                    for label in self.labels_set}

            random_state = np.random.RandomState(29)

            triplets = [{'indices': [i,
                                     random_state.choice([idx for idx in self.label_to_indices[self.test_labels[i].item()] if idx != i]),
                                     random_state.choice(self.label_to_indices[
                                                             np.random.choice(
                                                                 list(self.labels_set - set([self.test_labels[i].item()]))
                                                             )
                                                         ])
                                    ],
                         'labels': [self.test_labels[i].item(),
                                    self.test_labels[random_state.choice([idx for idx in self.label_to_indices[self.test_labels[i].item()] if idx != i])].item(),
                                    self.test_labels[random_state.choice(self.label_to_indices[
                                                                            np.random.choice(
                                                                                list(self.labels_set - set([self.test_labels[i].item()]))
                                                                            )
                                                                        ])].item()
                                    ]
                        }
                        for i in range(len(self.test_labels))]
            self.test_triplets = triplets


    def __getitem__(self, index):
        if self.train:

            audio1, label1 = self.voxceleb1_dataset[index][0], self.train_labels[index].item()
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
                #! 99% that for one particular speaker there is just 1 recording 
                #TODO Create histogram of number of samples assigned to one speaker
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])

            audio2 = self.voxceleb1_dataset[positive_index][0]
            audio3 = self.voxceleb1_dataset[negative_index][0]
            
            return (audio1, audio2, audio3), [label1, label1, negative_label]
        else:
            audio1 = self.voxceleb1_dataset[self.test_triplets[index]['indices'][0]][0]
            audio2 = self.voxceleb1_dataset[self.test_triplets[index]['indices'][1]][0]
            audio3 = self.voxceleb1_dataset[self.test_triplets[index]['indices'][2]][0]
            
            return (audio1, audio2, audio3), self.test_triplets[index]["labels"]


    def __len__(self):
        return len(self.voxceleb1_dataset)
    
if __name__ == "__main__":
    
    train_dataset = torchaudio.datasets.VoxCeleb1Identification('/mnt/d/VoxCeleb1Identification/data', subset='train', download=False)
    test_dataset = torchaudio.datasets.VoxCeleb1Identification('/mnt/d/VoxCeleb1Identification/data', subset='test', download=False)

    present_train_audio_files = json.load(open("present_train_audio_files.json", 'r'))
    present_test_audio_files = json.load(open("present_test_audio_files.json", 'r'))
    
    train_subset = VoxCeleb1IdentificationUnified(train_dataset, present_train_audio_files[:10000])
    test_subset = VoxCeleb1IdentificationUnified(train_dataset, present_test_audio_files[:1024])
    
    cuda = torch.cuda.is_available()
    batch_size = 128
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    
    triplet_dataset_train = TripletVoxCeleb1ID(train_subset, train=True)
    triplet_dataset_test = TripletVoxCeleb1ID(test_subset, train=False)
    triplet_train_loader = torch.utils.data.DataLoader(triplet_dataset_train, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs) # type: ignore
    triplet_test_loader = torch.utils.data.DataLoader(triplet_dataset_test, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs) # type: ignore
