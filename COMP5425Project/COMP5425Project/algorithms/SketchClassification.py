import torch
import torchvision
from torchvision import transforms, datasets, models
import torch.nn as nn
import numpy as np

import os
os.environ['TORCH_HOME'] = 'Cache'
import sys
import pickle


class SketchTransferClassifier(nn.Module):
    def __init__(self, n_classes = 50):
        super(SketchTransferClassifier, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.base = models.resnet50(pretrained=True)

        for param in self.base.parameters():
            param.requires_grad = False

        num_features = self.base.fc.in_features
        #self.base.fc = nn.Linear(num_features, n_classes)
        self.base.fc = nn.Sequential(
            nn.Linear(num_features, 100),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(100, n_classes)
        )
        for param in self.base.fc.parameters():
            param.requires_grad = True

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.to(self.device)

    def forward(self, x):
        out = self.base(x)
        return out


class SketchClassifier(nn.Module):
    def __init__(self, n_classes = 50):
        super(SketchClassifier, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(5, 5, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14x5
            nn.Conv2d(5, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 7x7x8
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU()  # 7x7x16
        )
        self.fc = nn.Sequential(
            nn.Linear(224 * 224, 100),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(100, n_classes)
        )
        self.to(self.device)

    def forward(self, x):
        out = self.layer1(x)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        return out


def GetTrainedClassifier():
    with open("./trained_models/model_20_classes_30.pth", "rb") as f:
        model = torch.load(f)
        model.eval()

    with open("./data/idx_to_class_20_classes.pkl", 'rb') as f:
        idx_to_class =  pickle.load(f)
    return model, idx_to_class



def train_loop(dataloader, model, loss_fn, optimizer, device = "cuda"):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X.to(device))
        loss = loss_fn(pred, y.to(device))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), batch * len(X)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn, device = "cuda"):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X.to(device))
            test_loss += loss_fn(pred, y.to(device)).item()
            correct += (pred.argmax(1) == y.to(device)).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct

class SketchDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None, device = 'cuda'):
        self.dataset = dataset
        self.transform = transform
        self.device = device

    def __getitem__(self, index):
        if self.transform:
            x = self.transform(self.dataset[index][0])
        else:
            x = self.dataset[index][0]
        y = self.dataset[index][1]
        return x.to(self.device), y
    
    def __len__(self):
        return len(self.dataset)

# get data_loaders
def GetSketchData(sketch_dir, device):
    # transform
    trans = transforms.Compose([ #transforms.Grayscale(1),
                                 #transforms.RandomRotation(25),
                                 #transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                                 transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor()])

    trans_without_aug = transforms.Compose([ #transforms.Grayscale(1),
                                             transforms.Resize(255), 
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor()])

    # get data
    full_set = datasets.ImageFolder(sketch_dir)
    traindataset = SketchDataset(full_set, trans, device = device)
    valdataset = SketchDataset(full_set, trans_without_aug, device = device)
    testdataset = SketchDataset(full_set, trans_without_aug, device = device)

    train_size = 0.6
    num_train = len(full_set)
    indices = list(range(num_train))
    split = int(np.floor(train_size * num_train))
    #split2 = int(np.floor((train_size+(1-train_size)/2) * num_train))
    split2 = len(indices) - 1
    np.random.shuffle(indices)
    train_idx, valid_idx, test_idx = indices[:split], indices[split:split2], indices[split2:]

    train_data = torch.utils.data.Subset(traindataset, indices = train_idx)
    valid_data = torch.utils.data.Subset(valdataset, indices = valid_idx)
    test_data = torch.utils.data.Subset(testdataset, indices = test_idx)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size = BATCH_SIZE, drop_last = False, num_workers = NUM_WORKERS)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size = BATCH_SIZE, drop_last = False, num_workers = NUM_WORKERS)
    test_loader  = torch.utils.data.DataLoader(test_data,  batch_size = BATCH_SIZE, drop_last = False, num_workers = NUM_WORKERS)

    idx_to_class = dict((v,k) for k,v in full_set.class_to_idx.items())
    return train_loader, test_loader, valid_loader, idx_to_class



if __name__ == "__main__":
    
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    EPOCHS = 200

    NUM_WORKERS = 0

    save_dir = "./trained_models/"

    sketch_dir = './data/sketchnet_selected/'

    idx_to_class_dir = "./data/idx_to_class_20_classes.pkl"
    accs_dir = "./data/saved_acc_20_classes.pkl"

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    train_loader, test_loader, valid_loader, idx_to_class = GetSketchData(sketch_dir, device = device)

    with open(idx_to_class_dir, 'wb') as f:
        pickle.dump(idx_to_class, f)

    loss_fn = nn.CrossEntropyLoss()
    model = SketchTransferClassifier(n_classes = len(idx_to_class))
    #model = SketchClassifier(n_classes = len(idx_to_class))
    #model = torch.load("./trained_models/model_20_classes_82.pth")
    #optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)



    ## debug
    #from matplotlib import pyplot as plt
    #for batch, (X, y) in enumerate(train_loader):
    #    for i in range(BATCH_SIZE):
    #        plt.imshow(X[i].detach().cpu().permute(1, 2, 0).numpy())
    #        plt.show()
    #        print(idx_to_class[y[i].item()])
        
    #with open("./data/saved_acc.pkl", 'rb') as f:
    #    accs = pickle.load(f)

    #for t in range(100, 114):
    #    print(f"Testing Epoch {t}\n-------------------------------")
    #    model = torch.load(os.path.join(save_dir, 'model_{}.pth'.format(t)))
    #    model.eval()
    #    accuracy = test_loop(test_loader, model, loss_fn, device = device)
    #    accs.append(accuracy)

    #print("Testing Done!")

    #idx_epoch = list(range(len(accs)))
  
    #plt.plot(idx_epoch, accs)
    #plt.title('Acc VS Epoch')
    #plt.xlabel('Epoch')
    #plt.ylabel('Accuracy')
    #plt.show()

    #sys.exit()
    from matplotlib import pyplot as plt
    accs = []

    for t in range(EPOCHS):
        print(f"Epoch {t}\n-------------------------------")
        train_loop(train_loader, model, loss_fn, optimizer, device = device)
        accuracy = test_loop(valid_loader, model, loss_fn, device = device)

        if t % 10 == 0:
            torch.save(model, os.path.join(save_dir, 'model_20_classes_{}.pth'.format(t)))
        accs.append(accuracy)
    print("Done!")

    with open(accs_dir, 'wb') as f:
        pickle.dump(accs, f)


    # show acc
    from matplotlib import pyplot as plt
    with open(accs_dir, 'rb') as f:
        accs = pickle.load(f)

    idx_epoch = list(range(len(accs)))
    plt.plot(idx_epoch, accs)
    plt.title('Acc VS Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()


