import wandb
from utils import *
from tqdm.notebook import tqdm
from models import create_model
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import pandas as pd


def get_criterion(args):
    if args.train_method in ["erm","pair_erm","encoder_erm"]:
        return CrossEntropyLoss()
    elif args.train_method in ['task_jepa','ijepa']:
        return F.smooth_l1_loss


def step_encoder_erm(args, models, x, y, criterion, train=True):
    # models in encoder, target_encoder
    encoder, predictor = models # encoder extracts features, predictor predicts on task given features
                                # if running from precalculated features, encoder is just the Identity
    
    
    bs, *_ = x[0].shape
    
    x = torch.cat([x[0] , x[1]], dim=0) # concatenate along batch dimension for efficiency
    features = encoder(x) 
    x_1 = features[:bs]
    x_2 = features[bs:]
    #x_1 = encoder(x[0])
    #x_2 = encoder(x[1])

    features = torch.cat([x_1, x_2], dim=-1)

    output = predictor(features)
    
    
    loss = []
    # decompose loss per task
    for i, fov in enumerate(args.fovs_tasks):
        loss.append(criterion(output[i], y[:,i]))

    loss = torch.stack(loss)
    avg_loss = loss.mean()

    if train:
        args.optimizer.zero_grad()
        avg_loss.backward()       # Backprop
        args.optimizer.step()      # Update Parameters
    
    return (encoder, predictor), [l.detach().cpu() for l in loss], [o.detach().cpu() for o in output]

def step_ijepa():
    pass
# Method for training the jepa encoder

def step_task_jepa(args, models, x, y, criterion, train=True):
    # models in encoder, target_encoder
    encoder, target_encoder = models # target_encoder should have gradients set to off
    n_fovs = y.shape[-1]
    # create all pairs of input and target
    latents_zeros = torch.zeros_like(y)
    # Make image pairs (input, target) and assign correct latents
    x_input = x[0]
    x_target = x[1]
    # 4 cases 
    # Same rep:
    # x_0 with latent vs x_1
    # x_0 vs x_1 with -latent
    # Different rep
    # x_0 vs x_1
    # x_0 with latent vs x_1 with latent! 

    model_device = next(target_encoder.parameters()).device
    targets_1 = target_encoder(x_target, latents_zeros)
    output_1 = encoder(x_input, y)
    targets_2 = target_encoder(x_target, -y)
    output_2 = encoder(x_input, latents_zeros)

    avg_loss = criterion(output_1, targets_1) + criterion(output_2, targets_2)      # Should have same reps
    avg_loss -= criterion(output_2, targets_1) + criterion(output_1, targets_2)     # Should have different reps

    #output = [o.detach().cpu() for o in output]
    
    if train:
        args.optimizer.zero_grad()
        avg_loss.backward()       # Backprop
        args.optimizer.step()      # Update Parameters
    
    return (encoder, target_encoder), avg_loss.detach().cpu(), [output.detach().cpu(), targets.detach().cpu()]

def step_pair_erm(args, models, x, y, criterion, train=True):
    # models in encoder, target_encoder
    encoder = models[0] # target_encoder should have gradients set to off
    x = torch.cat([x[0] , x[1]], dim=2) # concatenate data along sequence
    # model_device = next(target_encoder.parameters()).device
    # targets = target_encoder(x_target, latents_zeros)
    output = encoder(x) # concatenate along H dimension
    #print(output.shape)
    #avg_loss = criterion(output, latents_diff)
    loss = []
    # decompose loss per task
    for i, fov in enumerate(args.fovs_tasks):
        loss.append(criterion(output[i], y[:,i]))

    loss = torch.stack(loss)
    avg_loss = loss.mean()
    output = [o.detach().cpu() for o in output]
    
    if train:
        args.optimizer.zero_grad()
        avg_loss.backward()       # Backprop
        args.optimizer.step()      # Update Parameters
    
    return (encoder,), [l.detach().cpu() for l in loss], [o.detach().cpu() for o in output]

def step_erm(args, model, x, y, criterion, train=True):
    output = model[0](x)
    loss = []

    # decompose loss per task
    for i, fov in enumerate(args.fovs):
        idx = args.task_to_label_index[fov]
        loss.append(criterion(output[i], y[:,idx]))

    loss = torch.stack(loss)
    avg_loss = loss.mean()
    output = [o.detach().cpu() for o in output]
    
    if train:
        args.optimizer.zero_grad()
        avg_loss.backward()       # Backprop
        args.optimizer.step()      # Update Parameters
    return model, loss.detach().cpu(), output.detach().cpu()

def create_pairs(args, x, y):
    # define pairs and labels for pairwise training
    n_fovs = y.shape[-1]
    # create all pairs of input and target
    B, C, H, W = x.shape
    # only keep pairs where shapes are equal
    
    # Expand dimensions to create all pair combinations of images
    # x.unsqueeze(1) makes the shape (B, 1, C, H, W)
    # x.unsqueeze(0) makes the shape (1, B, C, H, W)
    image_pairs_1 = x.unsqueeze(1).expand(B, B, C, H, W)  # First element of the pairs (B, B, C, H, W)
    image_pairs_2 = x.unsqueeze(0).expand(B, B, C, H, W)  # Second element of the pairs (B, B, C, H, W)
    
    # Stack along a new dimension (2), so each pair is represented by (B, B, 2, C, H, W)
    all_image_pairs = torch.stack((image_pairs_1, image_pairs_2), dim=2)
    latents_diff = y.unsqueeze(1) - y.unsqueeze(0)
    latents_diff = latents_diff.view(-1, n_fovs)
    #equal_shape  = latents_diff[:,4] == 0 # for selecting pairs where the shape is the same
    # relabel diffs for classification
    # first for shape, floor color, wall color, object color, two values = 0 for same, 1 for different

    # Get the relevant columns
    cols = [0, 1, 2, 4]
    # Modify the tensor in place: set values != 0 to 1 for the selected columns
    latents_diff[:, cols] = torch.where(latents_diff[:, cols] != 0, torch.tensor(1), latents_diff[:, cols])
    # For dimensions [3, 5]: set values > 0 to 1 and values < 0 to 2
    latents_diff[:, [3, 5]] = torch.where(latents_diff[:, [3, 5]] > 0, torch.tensor(1), latents_diff[:, [3, 5]])
    latents_diff[:, [3, 5]] = torch.where(latents_diff[:, [3, 5]] < 0, torch.tensor(2), latents_diff[:, [3, 5]])

    #for i in range(6):
    #    print(torch.unique(latents_diff[:,i], return_counts=True))

    latents_diff = latents_diff[:,args.fovs_ids].long()
    all_image_pairs = all_image_pairs.view(-1, 2, C, H, W)
    #all_image_pairs = all_image_pairs[equal_shape]
    # latents_diff = latents_diff[equal_shape]
    # latents_zeros = torch.zeros_like(latents_diff)
    # Make image pairs (input, target) and assign correct latents
    x_input = all_image_pairs[:,0].squeeze()
    x_target = all_image_pairs[:,1].squeeze()
    x = [x_input , x_target]
    y = latents_diff
    
    return x, y
        
def run_epoch(args, model, dl, criterion, train=True, split = "train", print_every=100):
    # set up metric logging
    meters = setup_meters(args)
    # define train method
    for m in model:
        if m is not None:
            if train:
                m.train()
            else:
                m.eval()
    step_functions = {"encoder_erm": step_encoder_erm,
                      "erm": step_erm,
                      "pair_erm": step_pair_erm,
                      "task_jepa": step_task_jepa,
                      "ijepa": step_ijepa
                      }
    step_function = step_functions[args.train_method]
    # Loop
    with torch.set_grad_enabled(train):
        for n_batch, (x,y) in tqdm(enumerate(dl), total=len(dl), mininterval=10.0):

            if args.scheduler is not None:
                args.current_lr = args.scheduler.step()

            if args.wd_scheduler is not None:
                args.current_wd = args.wd_scheduler.step()

            x = x.cuda()
            y = y.cuda()

            if args.train_method in ["pair_erm","encoder_erm","task_jepa",'ijepa']:
                x, y = create_pairs(args, x, y)
            model, loss, output = step_function(args, model, x, y, criterion, train=train)
             # log metrics for wandb
            meters = update_meters(args, loss, output, y.detach().cpu(), meters)
            
            if (n_batch+1) % print_every == 0 or n_batch == 0:
                print(f"Iteration: {n_batch+1}")
                pprint_metrics(meters)
            # update target_encoder if training using JEPA
            if args.train_method in ["task_jepa", "ijepa"] and n_batch % args.iters_per_ema == 0 and train:
                # Step 3. momentum update of target encoder
                encoder, target_encoder = model
                with torch.no_grad():
                    m = next(args.momentum_scheduler)
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                        param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)
                model = (encoder, target_encoder)
    return model, meters


def remove_meters(meters):
    for k, v in meters.items():
        if isinstance(v, dict):
            for k1, v1 in v.items():
                meters[k][k1]=v1.avg.item()
        else:
            meters[k] = v.avg.item()
    return meters
    
def train(args, dls):
    
    set_seed(args.seed) # make experiments deterministic!


    exp_name = get_exp_name(args)

    if args.experiment_id is not None and args.resume: # resume wandb experiment
        wandb.init(settings=wandb.Settings(start_method="thread"),
            # set the wandb project where this run will be logged
            project="task_jepa",
            # track hyperparameters and run metadata
            config=args,
            name = exp_name,
            resume = "must",
            id = args.experiment_id
        )
    else:
        wandb.init(settings=wandb.Settings(start_method="thread"),
                # set the wandb project where this run will be logged
                project="task_jepa",
                # track hyperparameters and run metadata
                config=args,
                name = exp_name
            )
        args.experiment_id = wandb.run.id
    print(f"Experiment id is {args.experiment_id}")

    if args.train_method in ['task_jepa', 'ijepa']:
        
        args.ema = [args.ema_start, 1.0]      # exponential moving average
        args.ipe = len(dls['train']) # iterations per epoch
        args.momentum_scheduler = [args.ema[0] + i*(args.ema[1]-args.ema[0])/(args.ipe*args.num_epochs*args.ipe_scale/args.iters_per_ema)
                          for i in range(int(args.ipe*args.num_epochs*args.ipe_scale)+1)]
        args.momentum_scheduler = iter(args.momentum_scheduler)

    # define optimizer
    models = create_model(args)

    models = tuple(model.cuda() for model in models if model is not None)

    # Check if model is frozen or not

    frozen = True
    for p in models[0].parameters():
        if p.requires_grad:
            frozen = False
    
    if frozen:
        print("Encoder is frozen!")
    else:
        print("Encoder is unfrozen!")


    #best_model = models
    #best_metrics = None

    # optimizers

    args.optimizer, args.scheduler, args.wd_scheduler = init_opt(args, models)

    criterion = get_criterion(args)
    # start W&B experiment
    all_metrics = {'train': [], 'val': [], 'test': []}
    
    args.epoch = 1

    if args.resume: # reload last checkpoint: (epoch, model weights, scheduler, wd_scheduler, logged metrics)
        args, models, all_metrics = resume_last_checkpoint(args, models)

    for epoch in tqdm(range(args.epoch, args.num_epochs+1)):
        args.epoch = epoch
        full_metrics = dict()
        print(f"EPOCH {epoch}")
        models, train_metrics = run_epoch(args, models, dls['train'], criterion, train=True, split = "train")
        print("[TRAIN]")
        pprint_metrics(train_metrics)
        train_metrics = remove_meters(train_metrics)
        wandb.log(format_metrics_wandb(train_metrics, split="train"), step=epoch)

        train_metrics['epoch'] = epoch
        all_metrics['train'].append(format_metrics_wandb(train_metrics))
        # write metrics to disk

        _, val_metrics = run_epoch(args, models, dls['val'], criterion, train=False, split = "val")
        print("[VAL]")
        pprint_metrics(val_metrics)
        val_metrics = remove_meters(val_metrics)
        wandb.log(format_metrics_wandb(val_metrics, split="val"), step=epoch)
        val_metrics['epoch'] = epoch
        all_metrics['val'].append(format_metrics_wandb(val_metrics))

        _, test_metrics = run_epoch(args, models, dls['test'], criterion, train=False, split = "test")
        print("[TEST]")
        pprint_metrics(test_metrics)
        test_metrics = remove_meters(test_metrics)
        wandb.log(format_metrics_wandb(test_metrics, split="test"), step=epoch)
        test_metrics['epoch'] = epoch
        all_metrics['test'].append(format_metrics_wandb(test_metrics))

        # Save current model and metrics
        if args.save_weights and epoch % args.save_every == 0:
            save_model(args, models)
        if args.save_metrics:
            save_metrics(args, all_metrics)

        # For resuming!
        create_checkpoint(args, models, all_metrics)

    wandb.finish()
    model_dict = {k: args[k] for k in ['train_method', 'dataset', 'seed']}
    model_dict['model'] = tuple(m.state_dict() for m in best_model)

