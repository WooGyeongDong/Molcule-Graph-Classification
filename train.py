from tqdm import tqdm
import copy
import torch
import torch.nn.functional as F
import wandb
from torch_geometric.loader import DataLoader

from sklearn.metrics import roc_auc_score

from utils import num_graphs

def train_function(dataset, split_idx, model, args, device):
    best_valid_auc, best_epoch = 0, 0
    
    train_dataset = dataset[split_idx["train"]]
    valid_dataset = dataset[split_idx["valid"]]
    test_dataset = dataset[split_idx["test"]]
    
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay)
    
    for epoch in tqdm(range(1, args.epochs + 1)):

        train_loss = train_causal_epoch(
            model, optimizer, train_loader, device, args)
        if args.eval_metric == 'auc':
            valid_auc = eval_auc(
                model, valid_loader, device, args)
        elif args.eval_metric == 'acc':
            valid_auc = eval_acc(
                model, valid_loader, device, args)
        else:
            raise Exception("metric option not valid")
        
        try:
            wandb.log({
                'train_loss': train_loss,
                'valid_auc': valid_auc,
            })
        except: pass

        if valid_auc >= best_valid_auc:
            best_valid_auc = valid_auc
            best_epoch = epoch
            best_model = copy.deepcopy(model)
        
    print("Dataset:[{}] Model:[{}] | Best Valid:[{:.2f}] at epoch [{}]"
            .format(args.dataset,
                    args.model,
                    best_valid_auc * 100, 
                    best_epoch))
    if args.eval_metric == 'auc':
        last_test_auc = eval_auc(
            model, test_loader, device, args)
        test_auc = eval_auc(
            best_model, test_loader, device, args)
    elif args.eval_metric == 'acc':
        last_test_auc = eval_acc(
            model, test_loader, device, args)
        test_auc = eval_auc(
            best_model, test_loader, device, args)
    else:
        raise Exception("metric option not valid")
    
    print("Dataset:[{}] Model:[{}] | Best Test:[{:.2f}]"
            .format(args.dataset,
                    args.model,
                    test_auc * 100))
    print("Dataset:[{}] Model:[{}] | Final Test:[{:.2f}]"
            .format(args.dataset,
                    args.model,
                    last_test_auc * 100))
    
    return test_auc, best_model

def  train_causal_epoch(model, optimizer, train_loader, device, args):
    
    model.train()
    total_loss = 0
    for it, data in enumerate(train_loader):
        
        optimizer.zero_grad()
        data = data.to(device)
        target = data.y.view(-1)
        mask = ~target.isnan()
        target = target[mask].to(torch.long)
        logit = model(data)
        
        loss = torch.nn.CrossEntropyLoss()(logit.view(-1,args.num_classes)[mask], target)

        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
    
    num = len(train_loader.dataset)
    total_loss = total_loss / num
    return total_loss

def eval_auc(model, loader, device, args):
    
    model.eval()
    correct = 0
    pred, y = torch.tensor([]), torch.tensor([])
    for data in loader:
        data = data.to(device)
        y = torch.cat([y, data.y.view(-1, args.num_task).to('cpu')], dim=0)
        with torch.no_grad():
            logit = model(data)
            pred = torch.cat([pred, logit.detach().to('cpu')], dim=0)
    for i in range(args.num_task):
        mask = ~y[:,i].isnan()
        if args.num_classes == 2:
            y_score = F.softmax(pred[:,i*2:i*2+2], dim=-1)
            correct += roc_auc_score(y[mask,i], y_score[mask,1])
        else:
            y_score = F.softmax(pred[:, i*args.num_classes:(i+1)*args.num_classes], dim=-1)
            correct += roc_auc_score(y[mask,i], y_score, multi_class='ovr')
            
    return correct / args.num_task

def eval_acc(model, loader, device, args):
    
    model.eval()
    pred, y = torch.tensor([]), torch.tensor([])
    for data in loader:
        data = data.to(device)
        y = torch.cat([y, data.y.view(-1).to('cpu')], dim=0)
        with torch.no_grad():
            logit = model(data)
            logit = logit.view(-1, args.num_classes).max(1)[1]
            pred = torch.cat([pred, logit.detach().to('cpu')], dim=0)

    correct = pred.eq(y).sum().item()

    acc = correct /len(loader.dataset) / args.num_task
    return acc