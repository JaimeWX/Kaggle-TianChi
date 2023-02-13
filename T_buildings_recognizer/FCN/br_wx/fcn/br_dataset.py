import numpy as np
import torch.utils.data as data
import cv2
import random
from PIL import Image

def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = im.flatten(order = 'F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle_decode(mask_rle, shape=(512, 512)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')

class BRSegmentation(data.Dataset):

    def __init__(self,images,rles,transform=None):
        self.images = images
        self.rles = rles
        self.len = len(images)
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        mask = Image.fromarray(rle_decode(self.rles[index]))

        if self.transform is not None:
            img, mask = self.transform(img, mask)
        
        return img, mask
    
    def __len__(self):
        return self.len

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    # 计算该batch数据中，channel, h, w的最大值
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs

def split_BR_dataset(len_dataset, val_split_rate = 0.2):
    random.seed(0)
    dataset = set([x for x in range(len_dataset)])
    valid_idx = set(random.sample(dataset, int(len_dataset * val_split_rate)))
    train_idx = dataset - valid_idx
    return list(train_idx), list(valid_idx)
    