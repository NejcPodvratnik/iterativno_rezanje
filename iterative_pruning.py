import torch
import copy
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from early_stopper import EarlyStopper

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class IterativePruning():
    def __init__(self, model):
        super().__init__()

        self.model = model
        self.model = self.model.to(device)

        self.model.apply(self.weights_init)

        self.mask = self.create_mask()
        self.init_weights, self.init_biases = self.copy_initial_state()

    def weights_init(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, mean = 0.0, std = 0.1)
            if m.bias is not None:
                torch.nn.init.normal_(m.bias, mean = 0.0, std = 0.1)

    def create_mask(self):
        self.mask = []
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                self.mask += [torch.from_numpy(np.ones_like(param.data.cpu())).to(device)]
        return self.mask

    def freeze_pruned_weights(self):
        i = 0
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                param.grad.data = param.grad.data * self.mask[i]
                i += 1

    def copy_initial_state(self):
        init_weights = []
        init_biases = []
        for name, param in self.model.named_parameters():
            if "weight" in name:
                init_weights += [copy.deepcopy(param.data)]
            elif "bias" in name:
                init_biases += [copy.deepcopy(param.data)]
        return init_weights, init_biases

    def start(self, loss_fn, train_loader, val_loader, test_loader, lr, num_epochs, patience, min_delta, per):
        #writer = SummaryWriter("./logs")
        acc = []
        sp = []

        steps = 30

        for step in range(steps):
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            self.early_stopper = EarlyStopper(patience, min_delta)

            #for init in self.init_weights:
            #    print(init)

            print(f" ===| Prune iteration {step + 1}/{steps} |=== ")

            sum_zeros, sum_nonzeros, sum_all = 0, 0, 0
            for name, param in self.model.named_parameters():
                if "weight" in name or "bias" in name:
                    zeros = param.data[param.data == 0.0].numel()
                    nonzeros = param.data[param.data != 0.0].numel()
                    all = param.data.numel()
                    pruned = (nonzeros / all) * 100
                    sum_zeros, sum_nonzeros, sum_all = sum_zeros + zeros, sum_nonzeros + nonzeros, sum_all + all
                    print(f"{name:20} Zeros: {zeros:8} Nonzeros: {nonzeros:8} All: {all:8} Remaining: {pruned:8.2f}%")

            sum_pruned = (sum_nonzeros / sum_all) * 100
            name = "all"
            print(f"{name:20} Zeros: {sum_zeros:8} Nonzeros: {sum_nonzeros:8} All: {sum_all:8} Remaining: {sum_pruned:8.2f}%")

            self.train(optimizer, loss_fn, train_loader, val_loader, num_epochs)
            self.test(test_loader, acc, sp)

            ## Tukaj bi lahko shranil model
            
            i = 0
            for name, param in self.model.named_parameters():
                if "weight" in name:
                    alive_weights = param.data[param.data.nonzero(as_tuple=True)] # s tem ustvarimo 1d tenzor uteži, ki še niso bile odstranjene          
                    alive_weights = torch.abs(alive_weights)
                    sorted_weights = torch.argsort(alive_weights)
                    cut_per =  per if len(self.mask) - 1 != i else per / 2
                    pruned_weight_threshold = alive_weights[sorted_weights[int(len(sorted_weights) * cut_per)]]
                    self.mask[i] = torch.where(torch.abs(param.data) < pruned_weight_threshold, 0., self.mask[i])
                    param.data = copy.deepcopy(self.init_weights[i]) * self.mask[i]
                    i += 1
                    #print(param.data)
            i = 0
            for name, param in self.model.named_parameters():
                if "bias" in name:
                    param.data = copy.deepcopy(self.init_biases[i])
                    i += 1
        return acc, sp
        #writer.flush()
        #writer.close()

    def train(self, optimizer, loss_fn, train_loader, val_loader, num_epochs):
        bar = tqdm()
        text = f"Epoch: -, loss: -.----, val_loss: -.----"
        for epoch in range(num_epochs):
            self.model.train()
            losses = 0.0

            bar.total = len(train_loader.dataset)//train_loader.batch_size
            bar.refresh()
            bar.reset()
            bar.set_description(text + f" Training")

            for image, target in train_loader:
                image = image.to(device)
                target = target.to(device)

                pred = self.model(image)
                loss = loss_fn(pred, target)
                losses += loss

                optimizer.zero_grad()
                loss.backward()

                self.freeze_pruned_weights()

                optimizer.step()
                bar.update()
            
            self.model.eval()
            val_losses = 0.0

            bar.total = len(val_loader.dataset)//val_loader.batch_size
            bar.refresh()
            bar.reset()
            bar.set_description(text + f" Validating")

            with torch.no_grad():
                for image, target in val_loader:
                    image = image.to(device)
                    target = target.to(device)
                    
                    pred = self.model(image)
                    val_loss = loss_fn(pred, target)
                    val_losses += val_loss
                    bar.update()

            losses /= (len(train_loader.dataset)//train_loader.batch_size)
            val_losses /= (len(val_loader.dataset)//val_loader.batch_size)

            text = f"Epoch: {epoch + 1}, loss: {losses :.4f}, val_loss: {val_losses :.4f}"

            if self.early_stopper.early_stop(val_losses):
                #print(f"Training has ended due to early stoppage at epoch {epoch + 1}.")             
                break

    def test(self, test_loader, acc, sp):
        self.model.eval()

        correct = 0
        for image, target in tqdm(test_loader, total=len(test_loader.dataset)//test_loader.batch_size, desc="Testing"):
            image = image.to(device)
            target = target.to(device)

            pred = self.model(image)
            predicted = pred.argmax(1)

            for i in range(len(predicted)):
                if predicted[i] == target[i]:
                    correct += 1
        acc += [correct / len(test_loader.dataset)]
        sp += [torch.sum((self.mask[0]/150) * 100).cpu().numpy()]
        print(f"Accuracy: {correct / len(test_loader.dataset) :.3f}\n\n")