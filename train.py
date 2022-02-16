import torch 
import torchvision 
from torchvision import transforms
from torch.utils.data import DataLoader
import utils
from tqdm import trange

FOLDER_NAME = 'mobilenetv3_training'
NUM_CLASSES = 200

TRAIN_PATH = '/home/ubuntu/birdnet/cub200data/CUB_200_2011/train' #'./../birdnet/cub200data/CUB_200_2011/train/' 
TEST_PATH = '/home/ubuntu/birdnet/cub200data/CUB_200_2011/test' #'./../birdnet/cub200data/CUB_200_2011/test/'

TEST_SIZE = 2024

def train_epoch(model, opt, train_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()

    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        # We set this just for the example to run quickly.

        data, target = data.to(device), target.to(device)

        opt.zero_grad()
        output = model(data)

        _, predicted = torch.max(output.data, 1)

        loss = loss_fn(output, target)
        loss.backward()

        opt.step()

        total += target.size(0)
        correct += (predicted == target).sum().item()

def test_epoch(model, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            # We set this just for the example to run quickly.
            if batch_idx * len(data) > TEST_SIZE:
                break

            data, target = data.to(device), target.to(device)

            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)

            total += target.size(0)
            correct += (predicted == target).sum().item()

    return correct / total


def train_cub(epochs):
    # transforms 

    train_transforms = transforms.Compose([
                        transforms.RandomResizedCrop(224, scale = (0.8, 1.2)),
                        transforms.ToTensor(),
                        transforms.RandomPerspective(distortion_scale = 0.4, p = 1.0),
                        transforms.Normalize(mean=[0, 0, 0], std=[0.225, 0.225, 0.225]),
                        transforms.RandomRotation(degrees=(0, 20)),
                        transforms.RandomHorizontalFlip(p=0.5)
                        ])

    test_transforms = transforms.Compose([
                        transforms.Resize(224),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0, 0, 0], std=[0.225, 0.225, 0.225])
                        ])

    train_data = torchvision.datasets.ImageFolder(root = TRAIN_PATH, transform = train_transforms)
    test_data = torchvision.datasets.ImageFolder(root = TEST_PATH, transform = test_transforms)

    train_loader = DataLoader(train_data, batch_size = 128, shuffle = True)
    test_loader = DataLoader(test_data, batch_size = 128, shuffle = True)

    # define model
    
    #model = timm.create_model('mobilenetv3_large_100', pretrained = True, num_classes = NUM_CLASSES)
    model = utils.load_model('0.74561_220216183458', foldername= 'mobilenetv3_hpsearch')

    # define opt 
    opt = torch.optim.SGD(model.parameters(), 
                    lr = 1e-5
                    momentum = .9,
                    weight_decay = .003
                    )

    t = trange(epochs, leave=True)

    for epoch in t:
        train_acc = train_epoch(model, opt, train_loader)
        val_acc = test_epoch(model, test_loader)
        t.set_description(f"Val Acc: {val_acc:.5f}, Train Acc: {train_acc:.5f}")

    runname = utils.make_runname(prefix = f'{val_acc:.5f}')

    utils.save_model(runname, model, foldername = FOLDER_NAME)

if __name__ == "__main__":
    train_cub(25)