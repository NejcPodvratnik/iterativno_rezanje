import os
import torch
import copy
import datetime
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from early_stopper import EarlyStopper
from stats_tracker import StatsTracker

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class IterativePruning():
    def __init__(self, model):
        super().__init__()

        self.model = model
        self.model = self.model.to(device)

        #self.model.apply(self.weights_init)

        self.mask = self.create_mask()
        self.init_weights, self.init_biases = self.copy_initial_state()
        
        if not os.path.exists("./models"):
            os.makedirs("./models")

        self.model_filepath = f"./models/{self.model.__class__.__name__}_{datetime.datetime.now():%Y%m%d%H%M%S}"
        os.makedirs(self.model_filepath)

    def weights_init(self, m):
        if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d): ## BATCH NORM ŠE!
            torch.nn.init.normal_(m.weight, mean = 0.0, std = 0.1)
            #torch.nn.init.xavier_normal_(m.weight)
            #if m.bias is not None:
                #torch.nn.init.normal_(m.bias, mean = 0.0, std = 0.1)

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

        steps = 10
        pm = "global"

        self.stats_tracker = StatsTracker(self.model.__class__.__name__, lr, steps, patience, min_delta, per)

        for step in range(steps):

            self.stats_tracker.add_iteration()

            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            self.early_stopper = EarlyStopper(patience, min_delta)

            print(f" ===| Prune iteration {step + 1}/{steps} |=== ")
            print(f"Name                              Zeros Nonzeros      All Nonzeros(%)")

            sum_zeros, sum_nonzeros, sum_all = 0, 0, 0
            for name, param in self.model.named_parameters():
                if "weight" in name : #or "bias" in name:
                    zeros = param.data[param.data == 0.0].numel()
                    nonzeros = param.data[param.data != 0.0].numel()
                    all = param.data.numel()
                    pruned = (nonzeros / all) * 100
                    sum_zeros, sum_nonzeros, sum_all = sum_zeros + zeros, sum_nonzeros + nonzeros, sum_all + all
                    self.stats_tracker.add_layer(name, zeros, nonzeros, all, pruned)
                    #print(f"{name:30} {zeros:8} {nonzeros:8} {all:8} {pruned:10.2f}%")

            sum_pruned = (sum_nonzeros / sum_all) * 100
            name = "all"
            self.stats_tracker.add_layer(name, sum_zeros, sum_nonzeros, sum_all, sum_pruned)
            print(f"{name:30} {sum_zeros:8} {sum_nonzeros:8} {sum_all:8} {sum_pruned:10.2f}%")

            self.train(optimizer, loss_fn, train_loader, val_loader, num_epochs)
            acc = self.test(test_loader)

            self.stats_tracker.add_test_acc(acc * 100)
            filename = f"model_{step + 1}_a{acc * 100 :.1f}_p{sum_pruned :.2f}".replace(".","_")
            torch.save(self.model, self.model_filepath + "/" + filename + ".pt")

            if pm == "local":            
                i = 0
                for name, param in self.model.named_parameters():
                    if "weight" in name:
                        alive_weights = param.data[param.data.nonzero(as_tuple=True)] # s tem ustvarimo 1d tenzor uteži, ki še niso bile odstranjene          
                        alive_weights = torch.abs(alive_weights)
                        sorted_weights = torch.argsort(alive_weights)
                        cut_per =  per #if len(self.mask) - 1 != i else per / 2  ## POMEMBNO: ODSTRANIKL SEM CUTANJE ZA ZADNJO PLAST
                        pruned_weight_threshold = alive_weights[sorted_weights[int(len(sorted_weights) * cut_per)]]
                        self.mask[i] = torch.where(torch.abs(param.data) < pruned_weight_threshold, 0., self.mask[i])
                        param.data = copy.deepcopy(self.init_weights[i]) * self.mask[i]
                        i += 1
            elif pm == "global":
                all_alive_weights = torch.empty(0).to(device)
                for name, param in self.model.named_parameters():
                    if "weight" in name:
                        all_alive_weights = torch.cat((all_alive_weights, param.data[param.data.nonzero(as_tuple=True)]), dim = 0)

                all_alive_weights = torch.abs(all_alive_weights)
                all_sorted_weights = torch.argsort(all_alive_weights)
                pruned_weight_threshold = all_alive_weights[all_sorted_weights[int(len(all_sorted_weights) * per)]]
                i = 0
                for name, param in self.model.named_parameters():
                    if "weight" in name:
                        self.mask[i] = torch.where(torch.abs(param.data) < pruned_weight_threshold, 0., self.mask[i])
                        param.data = copy.deepcopy(self.init_weights[i]) * self.mask[i]
                        i += 1
            i = 0
            for name, param in self.model.named_parameters():
                if "bias" in name:
                    param.data = copy.deepcopy(self.init_biases[i])
                    i += 1
        self.stats_tracker.save_to_file(self.model_filepath + "/stats.json")

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
            self.stats_tracker.add_epoch(epoch + 1, losses.item(), val_losses.item())

            if self.early_stopper.early_stop(val_losses):
                bar.set_description(text + f" Validating")
                #print(f"Training has ended due to early stoppage at epoch {epoch + 1}.")             
                break

    def test(self, test_loader):
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
        acc = (correct / len(test_loader.dataset))
        print(f"Accuracy: {acc :.3f}\n\n")
        return acc