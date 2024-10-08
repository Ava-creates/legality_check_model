import copy
import math
import os
import random
import time
import torch
import wandb
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
# from focal_loss.focal_loss import FocalLoss
# from focal_loss_pytorch.focal_loss_pytorch.focal_loss import BinaryFocalLos

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
# Calculate the mean absolute percentage error


# class FocalLoss(torch.nn.Module):
#     '''
#     Multi-class Focal Loss
#     '''
#     def __init__(self, gamma=2, weight=None):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.weight = weight
#         self.reduction = reduction

#     def forward(self, input, target):
#         """
#         input: [N, C], float32
#         target: [N, ], int64
#         """
#         logpt = F.log_softmax(input, dim=1)
#         pt = torch.exp(logpt)
#         logpt = (1-pt)**self.gamma * logpt
#         loss = F.nll_loss(logpt, target, self.weight)
#         return loss
    
    

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
#         super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probs = torch.sigmoid(inputs)
        loss = self.alpha * (1 - probs) ** self.gamma * bce_loss
        return loss.mean()
    
    def backward(self, grad_output):
        # Custom backward pass implementation (if needed)
        pass
    
def Focal_Loss(inputs, target):
    gamma=2.0
    alpha=0.25
    bce_loss = F.binary_cross_entropy_with_logits(inputs, target, reduction='none')
    probas = torch.sigmoid(inputs)
    loss = alpha * (1 - probas) ** gamma * bce_loss
    return loss.mean()

    
def mape_criterion(inputs, targets):
    eps = 1e-5
    return 100 * torch.mean(torch.abs(targets - inputs) / (targets + eps))


def ddp_setup(rank: int, world_size: int):
  """
  Args:
      rank: Unique identifier of each process
     world_size: Total number of processes
  """
  os.environ["MASTER_ADDR"] = "localhost"
  os.environ["MASTER_PORT"] = "12355"
  init_process_group(backend="nccl", rank=rank, world_size=world_size)


def train_model(
    config,
    model,
    criterion,
    optimizer,
    max_lr,
    dataloader,
    num_epochs,
    log_every,
    logger,
    train_device,
    validation_device,
):

    since = time.time()
    losses = []
    train_loss = 0
    best_loss = math.inf
    best_model = None
    hash = random.getrandbits(16)
    dataloader_size = {"train": 0, "val": 0}
    # Calculate data size for each dataset. Used to caluclate the loss duriing the training  
    for item in dataloader["train"]:
        label = item[1]
        dataloader_size["train"] += label.shape[0]
    for item in dataloader["val"]:
        label = item[1]
        dataloader_size["val"] += label.shape[0]
    print(len(label))
    # Use the 1cycle learning rate policy
#     scheduler = OneCycleLR(
#         optimizer,
#         max_lr=max_lr,
#         steps_per_epoch=len(dataloader["train"]),
#         epochs=num_epochs,
#     )
    criterion = torch.nn.BCELoss()
    for epoch in range(num_epochs):
        epoch_start = time.time()
        for phase in ["train", "val"]:
            if phase == "train":
                # Enable gradient tracking for training
                model.train()
                device = train_device
            else:
                # Disable gradient tracking for evaluation
                model.eval()
                # If the user wants to run the validation on another GPU
                if (validation_device != "cpu"):
                    device = validation_device
            # Send model to the training device
            model = model.to(device)
            model.device = device
            
            running_loss = 0.0
            pbar = tqdm(dataloader[phase])
            
            # For each batch in the dataset
            for inputs, labels in pbar:

                if(labels.shape == (1,)):
                    print("batch_size 1")
                    continue
                # Send the labels and inputs to the training device
                original_device = labels.device
                inputs = (
                    inputs[0],
                    inputs[1].to(device),
                    inputs[2].to(device),
                    inputs[3].to(device),
                    inputs[4].to(device),
                    inputs[5].to(device),
                )
                labels = labels.to(device)
                
                # Reset the gradients for all tensors
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    # Get the model predictions
                    outputs = model(inputs)
#                     print(len(inputs))
#                     print("output_shape", outputs.shape)
#                     print(labels.shape)
                    try:
                        assert outputs.shape == labels.shape
                    except:
                            print(len(inputs))
                            print("output_shape", outputs.shape)
                            print(labels.shape)
                    # Calculate the loss
                    loss = criterion(outputs, labels.float())
                    if phase == "train":
                        # Backpropagation
                        loss.backward()
                        optimizer.step()
#                         scheduler.step()
                pbar.set_description("Loss: {:.3f}".format(loss.item()))
                running_loss += loss.item() * labels.shape[0]
                # Send the labels back to the original device 
                labels = labels.to(original_device)
                epoch_end = time.time()
            epoch_loss = running_loss / dataloader_size[phase]
            if phase == "val":
                # Append loss to the list of validation losses
                losses.append((train_loss, epoch_loss))
                # If we reached a new minimum loss
                if epoch_loss <= best_loss:
                    best_loss = epoch_loss
                    # Save the model weights at this checkpoint
                    best_model = copy.deepcopy(model)
                    saved_model_path = os.path.join(config.experiment.base_path, "weights/")
                    if not os.path.exists(saved_model_path):
                        os.makedirs(saved_model_path)
                    full_path = os.path.join(
                            saved_model_path,
                            f"best_model_{config.experiment.name}_{hash:4x}.pt",
                        )
                    logger.debug(f"Saving checkpoint to {full_path}")
                    torch.save(
                        model.state_dict(),
                        full_path,
                    )
                # Track progress using the wandb platform
                if config.wandb.use_wandb:
                    wandb.log(
                        {
                            "best_msle": best_loss,
                            "train_msle": train_loss,
                            "val_msle": epoch_loss,
                            "epoch": epoch,
                        }
                    )
                print(
                    "Epoch {}/{}:  "
                    "train Loss: {:.4f}   "
                    "val Loss: {:.4f}   "
                    "time: {:.2f}s   "
                    "best: {:.4f}".format(
                        epoch + 1,
                        num_epochs,
                        train_loss,
                        epoch_loss,
                        epoch_end - epoch_start,
                        best_loss,
                    )
                )
                if epoch % log_every == 0:
                    logger.info(
                        "Epoch {}/{}:  "
                        "train Loss: {:.4f}   "
                        "val Loss: {:.4f}   "
                        "time: {:.2f}s   "
                        "best: {:.4f}".format(
                            epoch + 1,
                            num_epochs,
                            train_loss,
                            epoch_loss,
                            epoch_end - epoch_start,
                            best_loss,
                        )
                    )
            else:
                train_loss = epoch_loss
    time_elapsed = time.time() - since

    print(
        "Training complete in {:.0f}m {:.0f}s   "
        "best validation loss: {:.4f}".format(
            time_elapsed // 60, time_elapsed % 60, best_loss
        )
    )
    logger.info(
        "-----> Training complete in {:.0f}m {:.0f}s   "
        "best validation loss: {:.4f}\n ".format(
            time_elapsed // 60, time_elapsed % 60, best_loss
        )
    )
#     destroy_process_group()
    return losses, best_model
