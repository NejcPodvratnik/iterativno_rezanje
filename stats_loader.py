import json

class Layer:
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

    @classmethod
    def from_dict(cls, data):
        return cls(
            name=data["name"],
            freezed=data["freezed"],
            active=data["active"],
            total=data["total"],
            freezed_per=data["freezed_per"]
        )


class Epoch:
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

    @classmethod
    def from_dict(cls, data):
        return cls(
            index=data["index"],
            loss=data["loss"],
            val_loss=data["val_loss"]
        )


class Iteration:
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

    @classmethod
    def from_dict(cls, data):
        iteration = cls(index=data["index"])
        iteration.test_acc = data["test_acc"]
        iteration.model_layers = [Layer.from_dict(layer) for layer in data["model_layers"]]
        iteration.epochs = [Epoch.from_dict(epoch) for epoch in data["epochs"]]
        return iteration
    

class StatsTracker:
    def __init__(self, model_name, lr, num_iterations, patience, per):
        super().__init__()
        self.model_name = model_name
        self.lr = lr
        self.num_iterations = num_iterations
        self.patience = patience
        self.per = per
        self.iterations = []

    def to_dict(self):
        return {
            "model_name": self.model_name,
            "lr": self.lr,
            "num_iterations": self.num_iterations,
            "patience": self.patience,
            "per": self.per,
            "iterations": [iteration.to_dict() for iteration in self.iterations]
        }

    def add_iteration(self):
        self.iterations += [Iteration(len(self.iterations) + 1)]

    def add_layer(self, name, freezed, active, total, freezed_per):
        self.iterations[-1].model_layers.append(Layer(name, freezed, active, total, freezed_per))
    
    def add_epoch(self, index, loss, val_loss):
        self.iterations[-1].epochs.append(Epoch(index, loss, val_loss))
    
    def add_test_acc(self, test_acc):
        self.iterations[-1].test_acc = test_acc

    def save_to_file(self, filename):
        stats_dict = self.to_dict()
        stats_json = json.dumps(stats_dict, indent=4)
        with open(filename, "w") as file:
            file.write(stats_json)
    

class StatsLoader:
    def __init__(self):
        super().__init__()

    def load_from_file(self, filename):
        with open(filename, "r") as file:
            data = json.load(file)
        return self.from_dict(data)

    def from_dict(self, data):
        tracker = StatsTracker(
            model_name=data["model_name"],
            lr=data["lr"],
            num_iterations=data["num_iterations"],
            patience=data["patience"],
            per=data["per"]
        )
        tracker.iterations = [Iteration.from_dict(iteration) for iteration in data["iterations"]]
        return tracker

