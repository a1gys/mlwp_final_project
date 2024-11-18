from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import UPFD


class MultiFeatureDataset(Dataset):

    def __init__(self,
                 features,
                 root='./datasets/UPFD',
                 name='politifact',
                 split='train'):
        super().__init__(transform=None, pre_transform=None)

        self.features = features
        if 'content' in features:
            self.content_dataset = UPFD(root=root, name=name, feature='content', split=split)
        if 'bert' in features:
            self.bert_dataset = UPFD(root=root, name=name, feature='bert', split=split)
        if 'profile' in features:
            self.profile_dataset = UPFD(root=root, name=name, feature='profile', split=split)
        if 'spacy' in features:
            self.spacy_dataset = UPFD(root=root, name=name, feature='spacy', split=split)

    def len(self):
        if 'content' in self.features:
            return len(self.content_dataset)
        elif 'bert' in self.features:
            return len(self.bert_dataset)
        elif 'profile' in self.features:
            return len(self.profile_dataset)
        elif 'spacy' in self.features:
            return len(self.spacy_dataset)
        else:
            raise NotImplementedError('Not implemented yet')

    def get(self, idx) -> list[Data]:
        result = []
        if 'content' in self.features:
            content_data = self.content_dataset[idx]
            result.append(content_data)
        if 'bert' in self.features:
            bert_data = self.bert_dataset[idx]
            result.append(bert_data)
        if 'profile' in self.features:
            profile_data = self.profile_dataset[idx]
            result.append(profile_data)
        if 'spacy' in self.features:
            spacy_data = self.spacy_dataset[idx]
            result.append(spacy_data)

        return result


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
