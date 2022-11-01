import torch
import config
from tqdm import tqdm
from dataset import CCDataset, collate_fn
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def train_one_epoch(loader, model, optimizer, device):
    loop = tqdm(loader)

    for batch_idx, (images, targets) in enumerate(loop):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        print(f"Total loss: {losses.item()}")

def main():
    train_dataset = CCDataset(mode='train')
    valid_dataset = CCDataset(mode='valid')
    test_dataset = CCDataset(mode='test')

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=config.BATCH_SIZE,
                              num_workers=config.NUM_WORKERS,
                              shuffle=True,
                              pin_memory=config.PIN_MEMORY,
                              collate_fn=collate_fn)

    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=config.BATCH_SIZE,
                              shuffle=True,
                              pin_memory=config.PIN_MEMORY,
                              collate_fn=collate_fn)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=config.BATCH_SIZE,
                             shuffle=True,
                             collate_fn=collate_fn)

    model = maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, num_classes=2)
    model.to(device)

    optimizer = torch.optim.AdamW(params=model.parameters(),
                                  lr=config.LEARNING_RATE,
                                  weight_decay=config.WEIGHT_DECAY)
    
    model.train()

    for epoch in range(config.NUM_EPOCHS):
        print(f"Epoch: {epoch}")
        train_one_epoch(train_loader, model, optimizer, config.DEVICE)

        if config.SAVE_MODEL:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=f"mask_rcnn_{epoch}.pth.tar")

if __name__ == "__main__":
    main()