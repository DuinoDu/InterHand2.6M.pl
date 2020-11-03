import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from interhand import data
from torchvision import transforms


def test_dataset():
    annot_subset = 'all'
    dataset = data.Dataset(transforms.ToTensor(), "train", annot_subset)
    print(f'{annot_subset} train dataset length: {len(dataset)}')
    dataset = data.Dataset(transforms.ToTensor(), "test", annot_subset)
    print(f'{annot_subset} test dataset length: {len(dataset)}')

    annot_subset = 'human_annot'
    dataset = data.Dataset(transforms.ToTensor(), "train", annot_subset)
    print(f'{annot_subset} train dataset length: {len(dataset)}')
    dataset = data.Dataset(transforms.ToTensor(), "test", annot_subset)
    print(f'{annot_subset} test dataset length: {len(dataset)}')

    annot_subset = 'machine_annot'
    dataset = data.Dataset(transforms.ToTensor(), "train", annot_subset)
    print(f'{annot_subset} train dataset length: {len(dataset)}')
    dataset = data.Dataset(transforms.ToTensor(), "val", annot_subset)
    print(f'{annot_subset} val dataset length: {len(dataset)}')
    dataset = data.Dataset(transforms.ToTensor(), "test", annot_subset)
    print(f'{annot_subset} test dataset length: {len(dataset)}')
    
    for item in dataset:
        inputs, targets, meta_info = item
        break


if __name__ == "__main__":
    test_dataset()
