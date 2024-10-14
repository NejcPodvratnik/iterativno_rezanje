import os
import torch
import copy
import numpy as np
from tqdm import tqdm

from early_stopper import EarlyStopper
from stats_tracker import StatsTracker

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class IterativePruning():
    def __init__(self, model_name, model, apply_weights):
        super().__init__()

        self.model_name = model_name
        self.model = model
        self.model = self.model.to(device)

        if apply_weights:
            self.model.apply(self.weights_init)

        self.mask = self.create_mask()
        self.init_weights, self.init_biases = self.copy_initial_state()

    def weights_init(self, m):
        if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
            #torch.nn.init.normal_(m.weight, mean = 0.0, std = 0.1)
            torch.nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        if isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight, mean = 0.0, std = 0.1)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

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
            elif 'bias' in name:
                param.grad.data = param.grad.data * 0

    def copy_initial_state(self):
        init_weights = []
        init_biases = []
        for name, param in self.model.named_parameters():
            if "weight" in name:
                init_weights += [copy.deepcopy(param.data)]
            elif "bias" in name:
                init_biases += [copy.deepcopy(param.data)]
        return init_weights, init_biases

    def prune_weights_resnet18(self, per):
        i = 0
        all_alive_weights = torch.empty(0).to(device)
        for name, param in self.model.named_parameters():
            if "weight" in name and "bn" not in name and "fc" not in name:
                all_alive_weights = torch.cat((all_alive_weights, param.data[param.data.nonzero(as_tuple=True)]), dim = 0)
                i += 1
        all_alive_weights = torch.abs(all_alive_weights)
        all_sorted_weights = torch.argsort(all_alive_weights)
        pruned_weight_threshold = all_alive_weights[all_sorted_weights[int(len(all_sorted_weights) * per)]]
        i = 0
        for name, param in self.model.named_parameters():
            if "weight" in name and "bn" not in name and "fc" not in name:
                self.mask[i] = torch.where(torch.abs(param.data) < pruned_weight_threshold, 0., self.mask[i])
                param.data = copy.deepcopy(self.init_weights[i]) * self.mask[i]
                i += 1
            elif "weight" in name and ("bn" in name or "fc" in name):
                param.data = copy.deepcopy(self.init_weights[i]) * self.mask[i]
                i += 1
        i = 0
        for name, param in self.model.named_parameters():
            if "bias" in name:
                param.data = copy.deepcopy(self.init_biases[i])
                i += 1

    def prune_weights_alexnet(self, per):
        i = 0
        all_alive_weights = torch.empty(0).to(device)
        for name, param in self.model.named_parameters():
            if "weight" in name and "features.0" not in name and "classifier.6" not in name:
                all_alive_weights = torch.cat((all_alive_weights, param.data[param.data.nonzero(as_tuple=True)]), dim = 0)
                i += 1
        all_alive_weights = torch.abs(all_alive_weights)
        all_sorted_weights = torch.argsort(all_alive_weights)
        pruned_weight_threshold = all_alive_weights[all_sorted_weights[int(len(all_sorted_weights) * per)]]
        i = 0
        for name, param in self.model.named_parameters():
            if "weight" in name and "features.0" not in name and "classifier.6" not in name:
                self.mask[i] = torch.where(torch.abs(param.data) < pruned_weight_threshold, 0., self.mask[i])
                param.data = copy.deepcopy(self.init_weights[i]) * self.mask[i]
                i += 1
            elif "weight" in name and ("features.0" in name or "classifier.6" in name):
                param.data = copy.deepcopy(self.init_weights[i]) * self.mask[i]
                i += 1
        i = 0
        for name, param in self.model.named_parameters():
            if "bias" in name:
                param.data = copy.deepcopy(self.init_biases[i])
                i += 1

    def prune_weights_vgg11(self, per):
        i = 0
        all_alive_weights = torch.empty(0).to(device)
        for name, param in self.model.named_parameters():
            if "weight" in name and "features.0" not in name and "classifier.6" not in name:
                all_alive_weights = torch.cat((all_alive_weights, param.data[param.data.nonzero(as_tuple=True)]), dim = 0)
                i += 1
        all_alive_weights = torch.abs(all_alive_weights)
        all_sorted_weights = torch.argsort(all_alive_weights)
        pruned_weight_threshold = all_alive_weights[all_sorted_weights[int(len(all_sorted_weights) * per)]]
        i = 0
        for name, param in self.model.named_parameters():
            if "weight" in name and "features.0" not in name and "classifier.6" not in name:
                self.mask[i] = torch.where(torch.abs(param.data) < pruned_weight_threshold, 0., self.mask[i])
                param.data = copy.deepcopy(self.init_weights[i]) * self.mask[i]
                i += 1
            elif "weight" in name and ("features.0" in name or "classifier.6" in name):
                param.data = copy.deepcopy(self.init_weights[i]) * self.mask[i]
                i += 1
        i = 0
        for name, param in self.model.named_parameters():
            if "bias" in name:
                param.data = copy.deepcopy(self.init_biases[i])
                i += 1

    def prune_weights_densenet121(self, per):
        i = 0
        all_alive_weights = torch.empty(0).to(device)
        for name, param in self.model.named_parameters():
            if "weight" in name and np.prod(np.shape(param.data)) > 500:
                all_alive_weights = torch.cat((all_alive_weights, param.data[param.data.nonzero(as_tuple=True)]), dim = 0)
                i += 1
        all_alive_weights = torch.abs(all_alive_weights)
        all_sorted_weights = torch.argsort(all_alive_weights)
        pruned_weight_threshold = all_alive_weights[all_sorted_weights[int(len(all_sorted_weights) * per)]]
        i = 0
        for name, param in self.model.named_parameters():
            if "weight" in name and np.prod(np.shape(param.data)) > 500:
                self.mask[i] = torch.where(torch.abs(param.data) < pruned_weight_threshold, 0., self.mask[i])
                param.data = copy.deepcopy(self.init_weights[i]) * self.mask[i]
                i += 1
            elif "weight" in name and not np.prod(np.shape(param.data)) > 500:
                param.data = copy.deepcopy(self.init_weights[i]) * self.mask[i]
                i += 1
        i = 0
        for name, param in self.model.named_parameters():
            if "bias" in name:
                param.data = copy.deepcopy(self.init_biases[i])
                i += 1

    def prune_weights_shufflenet_v2_x1_0(self, per):
        i = 0
        all_alive_weights = torch.empty(0).to(device)
        for name, param in self.model.named_parameters():
            if "weight" in name and np.prod(np.shape(param.data)) > 1000:
                all_alive_weights = torch.cat((all_alive_weights, param.data[param.data.nonzero(as_tuple=True)]), dim = 0)
                i += 1
        all_alive_weights = torch.abs(all_alive_weights)
        all_sorted_weights = torch.argsort(all_alive_weights)
        pruned_weight_threshold = all_alive_weights[all_sorted_weights[int(len(all_sorted_weights) * per)]]
        i = 0
        for name, param in self.model.named_parameters():
            if "weight" in name and np.prod(np.shape(param.data)) > 1000:
                self.mask[i] = torch.where(torch.abs(param.data) < pruned_weight_threshold, 0., self.mask[i])
                param.data = copy.deepcopy(self.init_weights[i]) * self.mask[i]
                i += 1
            elif "weight" in name and not np.prod(np.shape(param.data)) > 1000:
                param.data = copy.deepcopy(self.init_weights[i]) * self.mask[i]
                i += 1
        i = 0
        for name, param in self.model.named_parameters():
            if "bias" in name:
                param.data = copy.deepcopy(self.init_biases[i])
                i += 1

    def start(self, run, num_of_runs,test_filename, loss_fn, train_loader, val_loader, test_loader, lr, num_epochs, num_prune_iter, prune_per, patience):

        self.stats_tracker = StatsTracker(self.model.__class__.__name__, lr, num_prune_iter, patience, prune_per)

        for step in range(num_prune_iter):

            self.stats_tracker.add_iteration()

            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            self.early_stopper = EarlyStopper(patience)

            print(f" ===| Run {run + 1}/{num_of_runs} Prune iteration {step + 1}/{num_prune_iter} |=== ")
            print(f"Name                            Freezed   Active    Total   Active(%)")

            sum_freezed, sum_active, sum_total = 0, 0, 0
            i = 0
            for name, param in self.model.named_parameters():
                if "weight" in name : #or "bias" in name:
                    active = int(torch.sum(self.mask[i]))
                    total = int(np.prod(np.shape(self.mask[i])))
                    freezed = total - active
                    active_per = (active / total) * 100
                    sum_freezed, sum_active, sum_total = sum_freezed + freezed, sum_active + active, sum_total + total
                    self.stats_tracker.add_layer(name,  freezed, active, total, active_per)
                    #print(f"{name:30} {freezed:8} {active:8} {total:8} {active_per:10.2f}%")
                    i += 1

            sum_active_per = (sum_active / sum_total) * 100
            name = "total"
            self.stats_tracker.add_layer(name, sum_freezed, sum_active, sum_total, sum_active_per)
            print(f"{name:30} {sum_freezed:8} {sum_active:8} {sum_total:8} {sum_active_per:10.2f}%")

            self.train(optimizer, loss_fn, train_loader, val_loader, num_epochs)
            acc = self.test(test_loader)

            self.stats_tracker.add_test_acc(acc)
            filename = f"model_{step + 1}_a{acc :.1f}_p{sum_active_per :.2f}".replace(".","_")
            #torch.save(self.model, self.model_filepath + "/" + filename + ".pt")

            if self.model_name == "resnet18":
                self.prune_weights_resnet18(prune_per)
            elif self.model_name == "alexnet":
                self.prune_weights_alexnet(prune_per)
            elif self.model_name == "vgg11":
                self.prune_weights_vgg11(prune_per)
            elif self.model_name == "shuffle_net":
                self.prune_weights_shufflenet_v2_x1_0(prune_per)
            elif self.model_name == "densenet121":
                self.prune_weights_densenet121(prune_per)
            else:
                print("Wrong model used.")
                return
        self.stats_tracker.save_to_file(test_filename)

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

            if self.early_stopper.early_stop(val_losses, self.model):
                bar.set_description(text + f" Validating")
                self.model.load_state_dict(torch.load("./temp_checkpoint.pt"))
                os.remove("./temp_checkpoint.pt")
                #print(f"Training has ended due to early stoppage at epoch {epoch + 1}.")             
                break

    def test(self, test_loader):
        self.model.eval()
        test_losses = 0.0
        loss_fn = torch.nn.L1Loss()
        correct = 0
        with torch.no_grad():
            for image, target in tqdm(test_loader, total=len(test_loader.dataset)//test_loader.batch_size, desc="Testing"):
                image = image.to(device)
                target = target.to(device)

                pred = self.model(image)

                if np.shape(pred)[1] == 1:
                    test_loss = loss_fn(pred, target)
                    test_losses += test_loss
                else:
                    predicted = pred.argmax(1)
                    for i in range(len(predicted)):
                        if predicted[i] == target[i]:
                            correct += 1

        if test_losses != 0.0:
            test_losses /= (len(test_loader.dataset)//test_loader.batch_size)
            print(f"MAE: {test_losses :.4f}")
            return test_losses.item()
        else:
            acc = (correct / len(test_loader.dataset))
            print(f"Accuracy: {acc :.3f}\n\n")
            return acc * 100
        

'''
        if prune_mode == "local":            
            i = 0
            for name, param in self.model.named_parameters():
                if "weight" in name:
                    alive_weights = param.data[param.data.nonzero(as_tuple=True)] # s tem ustvarimo 1d tenzor uteži, ki še niso bile odstranjene          
                    alive_weights = torch.abs(alive_weights)
                    sorted_weights = torch.argsort(alive_weights)
                    cut_per =  per if len(self.mask) - 1 != i else per / 2  ## POMEMBNO: ODSTRANIKL SEM CUTANJE ZA ZADNJO PLAST
                    pruned_weight_threshold = alive_weights[sorted_weights[int(len(sorted_weights) * cut_per)]]
                    self.mask[i] = torch.where(torch.abs(param.data) < pruned_weight_threshold, 0., self.mask[i])
                    param.data = copy.deepcopy(self.init_weights[i]) * self.mask[i]
                    i += 1
'''