import torch.utils.data as data

class PatternDataset(data.Dataset):
    def __init__(self, dyadics, labels):
        self.dyadics = dyadics
        self.labels = labels

    def __getitem__(self, index):
        dyadics, target = self.dyadics[index], self.labels[index]
        return dyadics, target, index

    def __len__(self):
        return len(self.dyadics)

    def size(self):
        return len(self.dyadics)