import torch
import torch.nn.utils.prune as prune

from eval import accuracy
from train import fit, evaluate

def prune_and_finetune(model : torch.nn.Module, train_loader, test_loader, device, learning_rate, prune_epochs, train_epochs):
    for i in range(prune_epochs):
        params_to_prune = [
            (module, "weight") for module in filter(lambda m: type(m) == torch.nn.Conv2d, model.modules())
        ]

        prune.global_unstructured(
            params_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=0.3
        )
        test_loss, total, test_acc = evaluate(model, torch.nn.functional.cross_entropy, test_loader, metric=accuracy)
        print(f"Loss: {test_loss:.4f}, accuracy: {test_acc:.4f}\n")
        fit(train_epochs, model, torch.nn.functional.cross_entropy, torch.optim.SGD(model.parameters(), lr=learning_rate), train_loader, test_loader, accuracy)
        test_loss, total, test_acc = evaluate(model, torch.nn.functional.cross_entropy, test_loader, metric=accuracy)
        print(f"Loss: {test_loss:.4f}, accuracy: {test_acc:.4f}\n")
