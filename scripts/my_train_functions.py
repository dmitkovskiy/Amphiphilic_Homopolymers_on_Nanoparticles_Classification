import torch
import torch.nn as nn
from torch import optim
import torchmetrics

from tbparse import SummaryReader
import lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim.lr_scheduler import CosineAnnealingLR

import statistics
from sklearn.metrics import confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

class Lit(L.LightningModule):
    def __init__(self, model: nn.Module, optimizer_type: str = 'SGD', learning_rate: float = 0.001, 
                 momentum: float = 0.9, scheduler_type: str = 'cosine'):
        """
        Initializes the Lit class with the given parameters.

        Parameters:
        - model (nn.Module): The neural network model to be trained.
        - optimizer_type (str): The type of optimizer to use ('SGD' or 'Adam'). Default is 'SGD'.
        - learning_rate (float): The learning rate for the optimizer. Default is 0.001.
        - momentum (float): The momentum for the SGD optimizer. Default is 0.9.
        - scheduler_type (str): The type of learning rate scheduler to use ('cosine' for CosineAnnealingLR).
        """
        super().__init__()
        self.model = model
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=6)
        self.valid_acc = torchmetrics.Accuracy(task="multiclass", num_classes=6)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=6)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.scheduler_type = scheduler_type

        # Define ModelCheckpoint callback
        self.checkpoint_callback = ModelCheckpoint(
            monitor='val_acc_epoch',
            mode='max',
            filename='best_model',
            save_top_k=1,
            verbose=True
        )

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler.

        Returns:
        - dict: Contains the optimizer and the learning rate scheduler if 'cosine' scheduler type is selected.
        - optimizer: Returns only the optimizer if no scheduler is used.
        """        
        if self.optimizer_type == 'SGD':
            optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        elif self.optimizer_type == 'Adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")

        if self.scheduler_type == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=10)  # Adjust T_max as needed
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        else:
            return optimizer

    def training_step(self, batch, batch_idx):
        """
        Defines a single step in the training loop.

        Parameters:
        - batch (tuple): A tuple containing the inputs and labels for the current batch.
        - batch_idx (int): The index of the current batch.

        Returns:
        - loss (torch.Tensor): The computed loss for the current batch.
        """
        inputs, labels = batch
        pred = self.model(inputs)
        loss = self.criterion(pred, labels)

        self.log("train_loss", loss, prog_bar=True)
        self.train_acc(pred, labels)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Defines a single step in the validation loop.

        Parameters:
        - batch (tuple): A tuple containing the inputs and labels for the current batch.
        - batch_idx (int): The index of the current batch.
        """
        inputs, labels = batch
        pred  = self.model(inputs)
        loss = self.criterion(pred, labels)

        self.log("val_loss", loss, prog_bar=True)
        self.valid_acc(pred, labels)

    def on_train_epoch_end(self):
        """
        Computes and logs the training accuracy at the end of an epoch.
        """
        train_acc_epoch = self.train_acc.compute()
        self.log("train_acc_epoch", train_acc_epoch, prog_bar=True)
        self.train_acc.reset()

    def on_validation_epoch_end(self):
        """
        Computes and logs the validation accuracy at the end of an epoch.
        """
        val_acc_epoch = self.valid_acc.compute()
        self.log("val_acc_epoch", val_acc_epoch, prog_bar=True)
        self.valid_acc.reset()

    def test_step(self, batch, batch_idx):
        """
        Defines a single step in the test loop.

        Parameters:
        - batch (tuple): A tuple containing the inputs and labels for the current batch.
        - batch_idx (int): The index of the current batch.
        """
        inputs, labels = batch
        pred = self.model(inputs)
        loss = self.criterion(pred, labels)

        self.log("test_loss", loss, prog_bar=True)
        self.test_acc(pred, labels)

    def on_test_epoch_end(self):
        """
        Computes and logs the test accuracy at the end of an epoch.
        """
        test_acc_epoch = self.test_acc.compute()
        self.log("test_acc_epoch", test_acc_epoch, prog_bar=True)
        self.test_acc.reset()

def drop_nan(df, tag):
    """
    Drops NaN values from the specified column in the DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - tag (str): The column from which to drop NaN values.

    Returns:
    - pd.Series: The column with NaN values removed.
    """
    return df[~df[tag].isna()].loc[:, tag]

def plot_training_metrics(trainer):
    """
    Plot training and validation loss and accuracy from TensorBoard logs.
    
    Args:
        trainer: The training object that contains the logger with the experiment data.
        drop_nan (function): A function to drop NaN values from the DataFrame.
    """
    # Get the log directory
    log_dir = trainer.logger.experiment.get_logdir()
    
    # Read the summary data
    reader = SummaryReader(log_dir, pivot=True)
    df = reader.scalars

    # Create the plot
    plt.figure(figsize=(12, 3))
    
    # Plot the loss
    plt.subplot(1, 2, 1)
    plt.plot(drop_nan(df, "train_loss"), label="train", color="blue")
    plt.plot(drop_nan(df, "val_loss"), label="validation", color="orange")
    plt.title("Loss")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    # Plot the accuracy
    plt.subplot(1, 2, 2)
    plt.plot(drop_nan(df, "train_acc_epoch"), label="train", color="blue")
    plt.plot(drop_nan(df, "val_acc_epoch"), label="validation", color="orange")
    plt.title("Accuracy")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    
    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(model_best, test_loader, test_set, n_shift=10):
    """
    Evaluate the model on the test set and plot the confusion matrix.
    
    Args:
        model_best: Trained model to be evaluated.
        test_loader: DataLoader for the test set.
        test_set: Test dataset containing class labels.
        n_shift (int): Number of predictions to aggregate. Default is 10.
    """
    model_best.eval()
    
    all_labels = []
    all_predictions = []
    
    av_labels = []
    av_predictions = []
    
    samples_counter = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model_best(inputs)
            predictions = torch.argmax(outputs, axis=1).cpu().numpy()
    
            av_labels.extend(labels.numpy())
            av_predictions.extend(predictions)
    
            samples_counter += 1
            if samples_counter == n_shift:
                samples_counter = 0
    
                l = statistics.mode(av_labels)
                p = statistics.mode(av_predictions)
                all_labels.extend([l])
                all_predictions.extend([p])
                av_labels = []
                av_predictions = []

    # Compute the confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_predictions)

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=test_set.classes, yticklabels=test_set.classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()