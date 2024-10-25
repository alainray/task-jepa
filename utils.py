import torch
import numpy as np

def set_seed(seed):
    """Sets seed"""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets all statistics."""
        self.count = 0
        self.sum = 0.0
        self.avg = 0.0

    def update(self, value, n=1):
        """Updates the average meter with new value.

        Args:
            value (float): New value to add.
            n (int): Number of occurrences of the new value (default: 1).
        """
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        """Returns a string representation of the average meter."""
        return f'Average: {self.avg:.4f} (Count: {self.count})'

def get_exp_name(args):
    return f"{args.dataset}_{args.train_method}_{args.seed}"

def setup_meters(args):
    
    metrics =  dict()
    
    if args.train_method in ["erm","pair_erm", "encoder_erm"]:
        
        args.metrics.append("loss")
       
        for fov in args.fovs_tasks:
            metrics[fov] = dict()
            metrics[fov] = {metric: AverageMeter() for metric in args.metrics}
    else:
      
        metrics = {"loss": AverageMeter()}
        
    return metrics

def update_meters(args, loss, output, expected_output, all_meters):
    if args.train_method in ["erm","pair_erm", "encoder_erm"]:
        #print(all_meters, loss)
        for i, (fov, meters) in enumerate(all_meters.items()):
            for metric, meter in meters.items():
                if metric == "loss":
                    val = loss[i]
                elif metric == "acc":
                    idx = args.task_to_label_index[fov]
                    preds = output[i].argmax(dim=1)
                    y = expected_output[:,idx]
                    correct = (preds == y).float().sum()
                    print(preds.shape, y.shape)
                    print("corr", correct)
                    print("numel", y.numel())
                    total = y.numel()
                    acc = correct/total
                    val = acc 
                all_meters[fov][metric].update(val) # update task metric
    else:
        all_meters["loss"].update(loss)
    
    return all_meters

def get_best_model(best_model, current_model, best_metrics, current_metrics, method="val_avg_acc"):
    # parse method
    split, task, metric = tuple(method.split("_"))
    if task == "avg":
        # get avg of all metrics
        mean = 0.0
        for task_metric, v in current_metrics[split].items():
            if isinstance(v, dict):
                for met, value in v.items():
                    if metric == met:
                        mean +=value
        mean/= len(current_metrics[split]) - 1
        new_metric = mean
        mean = 0.0
        for task_metric, v in best_metrics[split].items():
            if isinstance(v, dict):
                for met, value in v.items():
                    if metric == met:
                        mean +=value
        mean/= len(best_metrics[split]) - 1
        old_metric = mean 
    else:
        new_metric = current_metrics[split][f"{task}_{metric}"] 
        old_metric = best_metrics[split][f"{task}_{metric}"] 
    if new_metric > old_metric:
        return current_model, current_metrics
    else:
        return best_model, best_metrics
        
def format_metrics_wandb(metrics, split=""):
    full_metrics = dict()
    for k, v in metrics.items():
        if not isinstance(v, dict):
            v = {"repr_pred": v}
        for metric, value in v.items():
            if split != "":
                name = f"{split}_{k}_{metric}"
            else:
                name = f"{k}_{metric}"
            full_metrics[name] = value
    return full_metrics

def pprint_metrics(meter_dict):

    """Pretty prints a dictionary of AverageMeter objects."""
    for obj, metrics in meter_dict.items():
        print(f"{obj.capitalize()}:")

        if isinstance(metrics, dict):
            for metric, meter in metrics.items():
                # Assuming the AverageMeter has a method that returns average and count
                print(f"  - {metric.capitalize()}: Avg = {meter.avg:.4f}")
        else:
            print(f"  - Loss: Avg = {metrics.avg:.4f}")
        print()  # Add an empty line for better readability