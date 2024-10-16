import torch

class EarlyStopper:
    def __init__(self, patience):
        self.patience = patience
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss, model):
        if validation_loss < self.min_validation_loss:
            torch.save(model.state_dict(), "./temp_checkpoint.pt")
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > self.min_validation_loss:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False