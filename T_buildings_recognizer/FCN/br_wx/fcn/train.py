import torch
import pandas as pd
import os
import time
import datetime

import transforms as T
from br_dataset import BRSegmentation, split_BR_dataset
from src import fcn_resnet50
from train_utils import train_one_epoch, evaluate, create_lr_scheduler

# -------------------------------------------- pre-process -------------------------------------------------------------------
class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size)
        max_size = int(2.0 * base_size)

        trans = [T.RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        trans.extend([
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.RandomResize(base_size, base_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train):
    base_size = 520
    crop_size = 480

    return SegmentationPresetTrain(base_size, crop_size) if train else SegmentationPresetEval(base_size)
# -------------------------------------------- pre-process -------------------------------------------------------------------

def create_model(aux, num_classes, pretrain=True):
    model = fcn_resnet50(aux=aux, num_classes=num_classes)

    if pretrain:
        weights_dict = torch.load("./fcn_resnet50_coco.pth", map_location='cpu')

        if num_classes != 21:
            # 官方提供的预训练权重是21类(包括背景)
            # 如果训练自己的数据集，将和类别相关的权重删除，防止权重shape不一致报错
            for k in list(weights_dict.keys()):
                if "classifier.4" in k:
                    del weights_dict[k]

        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys: ", missing_keys)
            print("unexpected_keys: ", unexpected_keys)

    return model

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    num_classes = args.num_classes + 1

    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    train_mask = pd.read_csv(args.rles, sep='\t', names=['name', 'mask'])
    train_mask['name'] = train_mask['name'].apply(lambda x: args.images + x)
    images = train_mask['name'].values
    rles = train_mask['mask'].fillna('').values
    print(f"the size of images is {len(images)}")
    print(f"the size of rles is {len(rles)}")

    train_idx, valid_idx = split_BR_dataset(len_dataset=len(train_mask['name']), val_split_rate=0.3)
    train_imges, val_images, train_rles, val_rles = [],[],[],[]
    for idx,i in enumerate(images):
        if idx in train_idx:
            train_imges.append(images[idx])
            train_rles.append(rles[idx])
        else:
            val_images.append(images[idx])
            val_rles.append(rles[idx])

    train_br_dataset = BRSegmentation(
        train_imges,
        train_rles,
        transform = get_transform(train=True)
    )

    val_br_dataset = BRSegmentation(
        val_images,
        val_rles,
        transform = get_transform(train=False)
    )

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

    train_loader = torch.utils.data.DataLoader(train_br_dataset,
                                            batch_size=batch_size,
                                            num_workers=num_workers,
                                            shuffle=True,                                           
                                            pin_memory=True,
                                            collate_fn=train_br_dataset.collate_fn
                                            )

    val_loader = torch.utils.data.DataLoader(val_br_dataset,
                                            batch_size=1,
                                            num_workers=num_workers,
                                            pin_memory=True,
                                            collate_fn=val_br_dataset.collate_fn
                                            )                                         
                                            
    model = create_model(aux=args.aux, num_classes=num_classes)
    print(model)
    model.to(device)

    params_to_optimize = [
        {"params": [p for p in model.backbone.parameters() if p.requires_grad]},
        {"params": [p for p in model.classifier.parameters() if p.requires_grad]}
    ]

    if args.aux:
        params = [p for p in model.aux_classifier.parameters() if p.requires_grad]
        params_to_optimize.append({"params": params, "lr": args.lr * 10})

    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # 创建学习率更新策略，这里是每个step更新一次(不是每个epoch)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)

        confmat = evaluate(model, val_loader, device=device, num_classes=num_classes)
        val_info = str(confmat)
        print(val_info)
        # write into txt
        with open(results_file, "a") as f:
            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n"
            f.write(train_info + val_info + "\n\n")

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()
        torch.save(save_file, "save_weights/model_{}.pth".format(epoch))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch fcn training")

    parser.add_argument("--images", default="/data/", help="br")
    parser.add_argument("--rles", default="/csv/", help="rle")
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=8, type=int)
    parser.add_argument("--num-classes", default=20, type=int)

    parser.add_argument("--aux", default=True, type=bool, help="auxilier loss")
    parser.add_argument("--epochs", default=30, type=int, metavar="N",
                        help="number of total epochs to train")
    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    main(args)
