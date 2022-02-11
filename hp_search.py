from sys import prefix
from ray import tune
import torch 
import torchvision 
from torchvision import transforms
import utils
import timm
from torch.utils.data import DataLoader

FOLDER_NAME = 'mobilenetv3_hpsearch'
NUM_CLASSES = 200

EPOCHS = 12
NUM_SAMPLES = 32

TRAIN_PATH = '/home/ubuntu/birdnet/cub200data/CUB_200_2011/train' #'./../birdnet/cub200data/CUB_200_2011/train/' 
TEST_PATH = '/home/ubuntu/birdnet/cub200data/CUB_200_2011/test' #'./../birdnet/cub200data/CUB_200_2011/test/'

TEST_SIZE = 2024

TRIAL_NAME = "init"


def train_epoch(model, opt, train_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        # We set this just for the example to run quickly.

        data, target = data.to(device), target.to(device)

        opt.zero_grad()
        output = model(data)

        loss = loss_fn(output, target)
        loss.backward()

        opt.step()

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

def train_cub(config):
    # base transforms 
    base_transforms = [
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0, 0, 0], std=[0.225, 0.225, 0.225])
    ]

    # train transforms 
    train_transforms = transforms.Compose([*base_transforms,
                        transforms.RandomRotation(degrees=(0, 20)),
                        transforms.RandomHorizontalFlip(p=0.5),
                        ])

    test_transforms = transforms.Compose([*base_transforms])

    # test transforsm 
    train_data = torchvision.datasets.ImageFolder(root = TRAIN_PATH, transform = train_transforms)
    test_data = torchvision.datasets.ImageFolder(root = TEST_PATH, transform = test_transforms)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

    # define model
    model = model = timm.create_model('mobilenetv3_large_100', pretrained = True, num_classes = NUM_CLASSES)

    # define opt 
    opt = torch.optim.SGD(model.parameters(), 
                    lr = config["lr"], 
                    momentum = config["momentum"],
                    weight_decay = config['l2']
                    )

    for epoch in range(EPOCHS):
        train_epoch(model, opt, train_loader)
        acc = test_epoch(model, test_loader)

        tune.report(mean_accuracy=acc)

    runname = utils.make_runname(prefix = f'{acc:.5f}')

    utils.save_model(runname, model, foldername = FOLDER_NAME)

if __name__ == "__main__":
    search_space = {
        'lr': tune.loguniform(1e-6, .1),
        'momentum': tune.choice([.6, .9, .99]),
        'l2': tune.loguniform(1e-6, 1e-3)}

    analysis = tune.run(
        train_cub,
        num_samples = NUM_SAMPLES,
        scheduler = tune.schedulers.ASHAScheduler(metric="mean_accuracy", mode="max"),
        config = search_space, 
        verbose = 1,
        resources_per_trial={"gpu": 1})

    utils.save_analysis('analysisinit', analysis, foldername = FOLDER_NAME)