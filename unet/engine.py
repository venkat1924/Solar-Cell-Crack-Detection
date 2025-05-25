# elpv_segmentation/engine.py
import torch
from tqdm import tqdm
from utils import calculate_metrics # Assuming utils.py is in the same directory

def train_epoch_segmentation(model, loader, optimizer, criterion, device, grad_clip_norm):
    model.train()
    epoch_loss_sum = 0.0
    progress_bar = tqdm(loader, desc="Training", leave=False)
    for images, masks in progress_bar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        
        epoch_loss_sum += loss.item() * images.size(0)
        progress_bar.set_postfix(loss=loss.item())
        
    return epoch_loss_sum / len(loader.dataset)


def eval_epoch_segmentation(model, loader, criterion, device, num_classes):
    model.eval()
    epoch_loss_sum = 0.0
    all_true = []
    all_pred = []
    progress_bar = tqdm(loader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            epoch_loss_sum += loss.item() * images.size(0)
            
            preds = torch.argmax(outputs, dim=1)
            all_true.append(masks.cpu())
            all_pred.append(preds.cpu())
            progress_bar.set_postfix(loss=loss.item())

    avg_loss = epoch_loss_sum / len(loader.dataset)
    true_combined = torch.cat(all_true, dim=0)
    pred_combined = torch.cat(all_pred, dim=0)
    
    iou, f1, iou_cls, f1_cls = calculate_metrics(true_combined, pred_combined, num_classes)
    return avg_loss, iou, f1, iou_cls, f1_cls