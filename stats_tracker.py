import json

class Layer():
    def __init__(self, name, freezed, active, total, freezed_per):
        self.name = name
        self.freezed = freezed
        self.active = active
        self.total = total
        self.freezed_per = freezed_per

    def to_dict(self):
        return {
            "name": self.name,
            "freezed": self.freezed,
            "active": self.active,
            "total": self.total,
            "freezed_per": self.freezed_per
        }

class Epoch():
    def __init__(self, index, loss, val_loss):
        self.index = index
        self.loss = loss
        self.val_loss = val_loss

    def to_dict(self):
        return {
            "index": self.index,
            "loss": self.loss,
            "val_loss": self.val_loss
        }


class Iteration():
    def __init__(self, index):
        self.index = index
        self.test_acc = 0
        self.model_layers = []
        self.epochs = []

    def to_dict(self):
        return {
            "index": self.index,
            "test_acc": self.test_acc,
            "model_layers": [layer.to_dict() for layer in self.model_layers],
            "epochs": [epoch.to_dict() for epoch in self.epochs]
        }
    

class StatsTracker():
    def __init__(self, model_name, lr, num_iterations, patience, min_delta, per):
        super().__init__()
        self.model_name = model_name
        self.lr = lr
        self.num_iterations = num_iterations
        self.patience = patience
        self.min_delta = min_delta
        self.per = per
        self.iterations = []

    def to_dict(self):
        return {
            "model_name": self.model_name,
            "lr": self.lr,
            "num_iterations": self.num_iterations,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "per": self.per,
            "iterations": [iteration.to_dict() for iteration in self.iterations]
        }

    def add_iteration(self):
        self.iterations += [Iteration(len(self.iterations) + 1)]

    def add_layer(self, name,  freezed, active, total, freezed_per):
        self.iterations[len(self.iterations) - 1].model_layers += [Layer(name,  freezed, active, total, freezed_per)]
    
    def add_epoch(self, index, loss, val_loss):
        self.iterations[len(self.iterations) - 1].epochs += [Epoch(index, loss, val_loss)]
    
    def add_test_acc(self, test_acc):
        self.iterations[len(self.iterations) - 1].test_acc = test_acc

    def save_to_file(self, filename):
        stats_dict = self.to_dict()
        stats_json = json.dumps(stats_dict, indent=4)
        with open(filename, "w") as file:
            file.write(stats_json)
    