from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import UPFD


class MultiFeatureDataset(Dataset):

    def __init__(self,
                 root='./datasets/UPFD',
                 name='politifact',
                 split='train'):
        super().__init__(transform=None, pre_transform=None)

        self.content_dataset = UPFD(root=root, name=name, feature='content', split=split)
        self.bert_dataset = UPFD(root=root, name=name, feature='bert', split=split)
        self.profile_dataset = UPFD(root=root, name=name, feature='profile', split=split)
        self.spacy_dataset = UPFD(root=root, name=name, feature='spacy', split=split)

    def len(self):
        return len(self.content_dataset)

    def get(self, idx) -> tuple[Data, Data, Data, Data]:
        content_data = self.content_dataset[idx]
        bert_data = self.bert_dataset[idx]
        profile_data = self.profile_dataset[idx]
        spacy_data = self.spacy_dataset[idx]

        return content_data, bert_data, profile_data, spacy_data


def get_data(name='politifact',
             root='./datasets/UPFD',
             feature='content') -> tuple[DataLoader, DataLoader, DataLoader]:
    train_data = UPFD(root=root, name=name, feature=feature, split='train')
    val_data = UPFD(root=root, name=name, feature=feature, split='val')
    test_data = UPFD(root=root, name=name, feature=feature, split='test')

    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

    return train_loader, val_loader, test_loader
