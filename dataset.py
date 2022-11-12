import os
import cv2
import numpy as np
import config
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class CCDataset(Dataset):
  def __init__(self, mode = 'train'):
    if mode == 'train':
      self.dataset_path = config.TRAIN_DIR
      ann_path = os.path.join(config.TRAIN_DIR, '_annotations.coco.json')
    if mode == 'valid':
      self.dataset_path = config.VALID_DIR
      ann_path = os.path.join(config.VALID_DIR, '_annotations.coco.json')
    if mode == 'test':
      self.dataset_path = config.TEST_DIR
      ann_path = os.path.join(config.TEST_DIR, '_annotations.coco.json')
    
    self.coco = COCO(ann_path)
    self.cat_ids = self.coco.getCatIds()

  def __len__(self):
      return len(self.coco.imgs)
  
  def get_masks(self, index):
      ann_ids = self.coco.getAnnIds([index])
      anns = self.coco.loadAnns(ann_ids)
      masks=[]

      for ann in anns:
            mask = self.coco.annToMask(ann)
            masks.append(mask)

      return masks

  def get_boxes(self, masks):
      num_objs = len(masks)
      boxes = []

      for i in range(num_objs):
          x,y,w,h = cv2.boundingRect(masks[i])
          boxes.append([x, y, x+w, y+h])

      return np.array(boxes)

  def __getitem__(self, index):
      # Load image
      img_info = self.coco.loadImgs([index])[0]
      image = cv2.imread(os.path.join(self.dataset_path,
                                    img_info['file_name']))
      masks = self.get_masks(index)

      if self.augmentation:
        augmented = self.augmentation(image=image, masks=masks)
        image, masks = augmented['image'], augmented['masks']

      image = image.transpose(2,0,1) 

      # Load masks
      masks = np.array(masks)
      boxes = self.get_boxes(masks)

      # Create target dict
      num_objs = len(masks)
      boxes = torch.as_tensor(boxes, dtype=torch.float32)
      labels = torch.ones((num_objs,), dtype=torch.int64)
      masks = torch.as_tensor(masks, dtype=torch.uint8)
      image = torch.as_tensor(image, dtype=torch.float32)
      data = {}
      data["boxes"] =  boxes
      data["labels"] = labels
      data["masks"] = masks

      return image, data


def collate_fn(batch):
  images = list()
  targets = list()
  for b in batch:
        images.append(b[0])
        targets.append(b[1])
  images = torch.stack(images, dim=0)
  return images, targets

if __name__ == "__main__":
    # Test
    dataset = CCDataset(mode = 'valid')
    
    loader = DataLoader(dataset = dataset,
                    batch_size = 2,
                    num_workers = 2,
                    shuffle = True,
                    pin_memory = True,
                    collate_fn = collate_fn)

    for images, targets in tqdm(loader):
        print(images.shape) # [2, 3, 640, 640]
        print(len(targets)) # 2
        import sys
        sys.exit()