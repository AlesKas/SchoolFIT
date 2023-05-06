import torch

import numpy as np

if torch.cuda.is_available():  
    dev = "cuda:0" 
else:  
    dev = "cpu" 

def loss_batch(model, loss_fun, xb, yb, opt=None, metric=None):
    preds = model(xb)

    loss = loss_fun(preds, yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    metric_result = None
    if metric is not None:
        metric_result = metric(preds, yb)

    return loss.item(), len(xb), metric_result

def evaluate(model, loss_fun, valid_dl, metric=None):
    model.eval()
    with torch.no_grad():
        results = [loss_batch(model, loss_fun, xb.to(dev), yb.to(dev), metric=metric) for xb, yb in valid_dl]
        losses, nums, metric = zip(*results)

        total = np.sum(nums)

        avg_loss = np.sum(np.multiply(losses, nums)) / total
        avg_metric = None
        if metric is not None:
            avg_metric = np.sum(np.multiply(metric, nums)) / total
        
    return avg_loss, total, avg_metric

def fit(epochs, model, loss_fun, opt, train_dl, valid_dl, metric=None):
    model.train()
    for epoch in range(epochs):
        for xb, yb in train_dl:
            xb, yb = xb.to(dev), yb.to(dev)
            loss, _, _ = loss_batch(model, loss_fun, xb, yb, opt, metric)

        result = evaluate(model, loss_fun, valid_dl, metric)
        val_loss, total, val_metric = result

        if metric is None:
            print(f'Epoch {epoch+1}, loss: {val_loss:.4f}')
        else:
            print(f'Epoch {epoch+1}, loss: {val_loss:.4f}, {metric.__name__}: {val_metric:.4f}')