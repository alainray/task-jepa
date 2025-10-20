import torch
import torch.nn as nn
from skorch import NeuralNetRegressor, NeuralNetClassifier
from skorch.callbacks import EpochScoring, Callback
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix
from skorch.dataset import ValidSplit
from collections import defaultdict
import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description="Train regression or classification model with skorch")
parser.add_argument('--arch', type=str, required=True, help='Architecture to extract features from')
parser.add_argument('--method', type=str, choices=['regression', 'classification'], required=True, help='Task type')
parser.add_argument('--num_blocks', type=int, default=1, help='Number of Linear Blocks in classifier')
args = parser.parse_args()

# Load data
data = torch.load("idsprites/idsprites.pth")
model = args.arch
reps = torch.load(f"idsprites/idsprites_images_feats_{model}.pth")

torch.manual_seed(42)
X = reps
print("Recentering representations!")
X = X - X.mean(dim=0)
print("Normalizing reps!")
X = torch.nn.functional.normalize(X, p=2.0, dim=1, eps=1e-12)

# Model definitions
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

class LinearClassificationModel(nn.Module):
    def __init__(self, input_dim, num_classes, num_blocks=1):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        modules = []

        for i in range(num_blocks-1):
            modules.append(nn.ReLU())
            modules.append(nn.Linear(input_dim, input_dim))
        modules.append(nn.Linear(input_dim, num_classes))

        self.linear = nn.Sequential(*modules)

    def forward(self, x):
        return self.linear(x)

# Custom callback to store confusion matrices for the best model
class BestConfusionMatrixCallback(Callback):
    def __init__(self):
        self.best_cm = None
        self.best_score = -np.inf

    def on_epoch_end(self, net, dataset_train=None, dataset_valid=None, **kwargs):
        valid_loader = net.get_iterator(dataset_valid, training=False)
        y_true, y_pred = [], []
        for batch in valid_loader:
            Xi, yi = batch
            preds = net.predict(Xi)
            y_true.extend(yi.cpu().numpy())
            y_pred.extend(preds)
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        current_acc = accuracy_score(y_true, y_pred)
        if current_acc > self.best_score:
            self.best_score = current_acc
            self.best_cm = confusion_matrix(y_true, y_pred)

input_dims = {    
    "vit_b_16": 768,    
    "vit_b_32": 768,
    "vit_l_16": 1024,
    "vit_l_32": 1024
}

all_results = dict()
attributes = ['shape','scale','orientation','x','y']

for i, attribute in enumerate(attributes):
    y = data['latent_ids'][:, i]

    cm_callback = BestConfusionMatrixCallback() if args.method == 'classification' else None

    if args.method == 'classification':
        y = y.long()  # Targets for classification must be integer
        model_class = LinearClassificationModel
        loss_fn = nn.CrossEntropyLoss
        accuracy_cb = EpochScoring(
            accuracy_score, name="accuracy", lower_is_better=False, on_train=False
        )
        net = NeuralNetClassifier(
            model_class,
            module__input_dim=input_dims[model],
            module__num_classes=len(y.unique()),
            module__num_blocks = args.num_blocks,
            max_epochs=50,
            lr=0.01,
            device="cuda",
            optimizer=torch.optim.Adam,
            criterion=loss_fn,
            batch_size=2048,
            train_split=ValidSplit(0.2),
            callbacks=[accuracy_cb, cm_callback],
            iterator_train__shuffle=True,
            verbose=1
        )
    else:
        y = y.float()
        model_class = LinearRegressionModel
        loss_fn = nn.MSELoss
        r2_cb = EpochScoring(
            r2_score, name="r2", lower_is_better=False, on_train=False
        )
        net = NeuralNetRegressor(
            model_class,
            module__input_dim=input_dims[model],
            max_epochs=50,
            lr=0.01,
            device="cuda",
            optimizer=torch.optim.Adam,
            criterion=loss_fn,
            batch_size=2048,
            train_split=ValidSplit(0.2),
            callbacks=[r2_cb],
            iterator_train__shuffle=True,
            verbose=1
        )

    print(f"Fitting the {args.method} for '{attribute}'")
    net.fit(X, y if args.method == 'regression' else y)

    results = defaultdict(list)
    metric_key = 'r2' if args.method == 'regression' else 'accuracy'
    for metrics in net.history:
        for key in ['epoch','train_loss','valid_loss', metric_key]:
            results[key].append(metrics[key])

    df = pd.DataFrame(results)

    if args.method == 'classification' and cm_callback.best_cm is not None:
        best_cm_df = pd.DataFrame(cm_callback.best_cm)
        best_cm_df.to_csv(f"confusion_matrix_best_{model}_{attribute}.csv", index=False)

    all_results[attribute] = df

# Save results
for attr, df in all_results.items():
    df.to_csv(f"results_{args.method}_{model}_{attr}.csv", index=False)

print("Training complete.")
