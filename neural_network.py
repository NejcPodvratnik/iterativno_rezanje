import torch
import copy
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from early_stopper import EarlyStopper

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Features(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(1, 6, (5, 5), padding=(2, 2))
        self.conv_2 = torch.nn.Conv2d(6, 16, (5, 5))
        self.avg_pool = torch.nn.AvgPool2d((2, 2), (2, 2))
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):  
        x = self.conv_1(x)   
        x = self.sigmoid(x)  
        x = self.avg_pool(x) 
        x = self.conv_2(x)    
        x = self.sigmoid(x)   
        x = self.avg_pool(x)  

        return x

class Classifier(torch.nn.Module):

    def __init__(self, n_classes) -> None:
        super().__init__()
        self.linear_1 = torch.nn.Linear(400, 120)
        self.linear_2 = torch.nn.Linear(120, 84)
        self.linear_3 = torch.nn.Linear(84, n_classes)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):    
        x = self.linear_1(x) 
        x = self.sigmoid(x) 
        x = self.linear_2(x) 
        x = self.sigmoid(x)
        x = self.linear_3(x) 
        return x            

class LeNet(torch.nn.Module):
    def __init__(self, n_classes) -> None:
        super().__init__()
        self.features = Features()
        self.flatten = torch.nn.Flatten()
        self.classifier = Classifier(n_classes)

    def forward(self, x):      
        x = self.features(x)   
        x = self.flatten(x)    
        x = self.classifier(x)
        return x               

class LeNet300(torch.nn.Module):
    def __init__(self, input_dim, n_classes):
        super(LeNet300, self).__init__()
        self.linear_1 = torch.nn.Linear(input_dim, 300)  
        self.linear_2 = torch.nn.Linear(300, 100)
        self.linear_3 = torch.nn.Linear(100, n_classes)
        self.relu = torch.nn.ReLU()
        self.input_dim = input_dim
    
    def forward(self, x):
        x = x.view(-1, self.input_dim) 
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        x = self.relu(x)
        x = self.linear_3(x)
        return x


def create_mask(model):
    mask = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            mask += [torch.from_numpy(np.ones_like(param.data.cpu())).to(device)]
    return mask

def freeze_pruned_weights(model, mask):
    i = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            param.grad.data = param.grad.data * mask[i]
            i += 1

def copy_initial_weights(model):
    init_weights = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            init_weights += [copy.deepcopy(param.data)]
    return init_weights


def train_with_iterative_pruning():
    pass


def train_2(model, optimizer, loss_fn, train_loader, val_loader, num_epochs, early_stopper):
    for epoch in range(num_epochs):
        model.train()
        losses = 0.0
        for image, target in tqdm(train_loader, total=len(train_loader.dataset)//train_loader.batch_size):
            image = image.to(device)
            target = target.to(device)

            pred = model(image)
            loss = loss_fn(pred, target)
            losses += loss

            optimizer.zero_grad()
            loss.backward()

            freeze_pruned_weights(model, mask)

            optimizer.step()

        model.eval()
        val_losses = 0.0
        with torch.no_grad():
            for image, target in tqdm(val_loader, total=len(val_loader.dataset)//val_loader.batch_size):
                image = image.to(device)
                target = target.to(device)
                
                pred = model(image)
                val_loss = loss_fn(pred, target)
                val_losses += val_loss

        losses /= (len(train_loader.dataset)//train_loader.batch_size)
        val_losses /= (len(val_loader.dataset)//val_loader.batch_size)

        writer.add_scalar("Loss/train", losses, epoch)
        writer.add_scalar("Loss/val", val_losses, epoch)
        print(f"Epoch: {epoch + 1}/{num_epochs}, loss: {losses :.4f}, val_loss: {val_losses :.4f}")

        if early_stopper.early_stop(val_losses):
            print(f"Training has ended due to early stoppage at epoch {epoch + 1}.")             
            break

def train(model, optimizer, loss_fn, train_loader, val_loader, num_epochs, patience, min_delta):
    writer = SummaryWriter("./logs")
    model = model.to(device)
    early_stopper = EarlyStopper(patience, min_delta)

    mask = create_mask(model)
    init_weights = copy_initial_weights(model)

    

    i = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            alive_weights = param.data[param.data.nonzero(as_tuple=True)] # s tem ustvarimo 1d tenzor uteži, ki še niso bile odstranjene          
            alive_weights = torch.abs(alive_weights)
            sorted_weights = torch.argsort(alive_weights)
            pruned_weight_threshold = alive_weights[sorted_weights[int(len(sorted_weights) * 0.1)]]
            mask[i] = torch.where(torch.abs(param.data) < pruned_weight_threshold, 0., mask[i])

            param.data = init_weights[i] * mask[i]
            i += 1
            print(param.data)

                

    writer.flush()
    writer.close()

def test(model, test_loader, test_data_len):
    model = model.to(device)
    model.eval()

    correct = 0
    for image, target in tqdm(test_loader, total=test_data_len//test_loader.batch_size):
        image = image.to(device)
        target = target.to(device)

        pred = model(image)
        predicted = pred.argmax(1)

        for i in range(len(predicted)):
            if predicted[i] == target[i]:
                correct += 1
    
    print(f"Točnost: {correct / test_data_len :.3f}")