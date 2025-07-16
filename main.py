#%%
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import argparse
import importlib
import torch
import wandb

from utils import set_seed
dataset_module = importlib.import_module('dataset')
importlib.reload(dataset_module)
model_module = importlib.import_module('model')
importlib.reload(model_module)
train_module = importlib.import_module('train')
importlib.reload(train_module)
#%%
parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--layers', type=int, default=2)
parser.add_argument('--hidden', type=int, default=256)

parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0)

parser.add_argument('--model', type=str, default="GCN", help="GCN, GIN, GAT")

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--dataset', type=str, default='mutag',
                        help='mutag, bbbp, bace, clintox, tox21')
parser.add_argument('--target_col', type=int, default=None, help='Options for multi-target dataset')

parser.add_argument('--eval_metric', type=str, default='auc')

try:
    args = parser.parse_args()
except:
    args = parser.parse_args([])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.device = device

name = f'{args.dataset}_{args.model}'
#%%
wandb.init(project = f'Basic_GNN')
wandb.run.name = name
wandb.run.save()
wandb.config.update(args)

#%%
def main(args):
    set_seed(args.seed)

    if args.dataset.upper()=='MUTAG':
        dataset, split_idx = dataset_module.get_MUTAG()   
    else:
        dataset, split_idx = dataset_module.get_MolculeNetData(args.dataset, target_col=args.target_col) 

    args.num_task = dataset[0].y.view(1,-1).shape[1]
    args.num_classes = 2
    
    model = model_module.BasicGNN(args).to(device)
    
    test_auc, best_model = train_module.train_function(
        dataset, split_idx, model, args, device)
    
    model_dir = f"./assets/model/{args.dataset}/target_{args.target_col}"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_name = f"{name}_{args.seed}"
    torch.save(best_model.state_dict(), f"./{model_dir}/{model_name}.pt")
    
    try:
        artifact = wandb.Artifact(
            f'{args.model}_{args.dataset}', 
            type='model',
            metadata=vars(args))
        artifact.add_file(f'./{model_dir}/{model_name}.pt')
        wandb.log_artifact(artifact)
    except: pass
    
    logs = {
        'AUC': test_auc,
    }
    
    try:
        wandb.log(logs)
    except: pass
    
    return logs 
#%%
if __name__ == "__main__":
    result = []
    for i in range(10):
        args.seed = i
        res = main(args)
        result_name = list(res.keys())
        result.append(list(res.values()))
    result = torch.tensor(result)*100
    res_mean = result.mean(dim=0)
    res_std = result.std(dim=0)
    print("=" * 150)
    print(f'{args.dataset}')
    for i in range(len(res_mean)):    
        print(f'& ${res_mean[i]:.2f}_{{\\pm {res_std[i]:.2f}}}$ ', end='')
    print('')
    print("=" * 150)   
    try:
        for i in range(len(res_mean)): 
            wandb.log({
                f'{result_name[i]}_mean': res_mean[i],
                f'{result_name[i]}_std': res_std[i]})
        wandb.run.finish()
    except:pass 
#%%
