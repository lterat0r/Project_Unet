import os
import time
import datetime
import torch
from src import FCN8s
from train_utils import train_one_epoch, evaluate, create_lr_scheduler
from my_dataset import DriveDataset
import transforms as T
import pandas as pd


def get_transform(args,mean = (0.485, 0.456, 0.406),std = (0.229, 0.224, 0.225)):
    train_transform = T.Compose([
        T.CenterCrop(512),
        T.RandomHorizontalFlip(0.5),
        T.RandomVerticalFlip(0.5),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    val_transform = T.Compose([
        T.CenterCrop(512),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    return train_transform,val_transform


def create_model(num_classes):
    # create model
    model = FCN8s( nclass=num_classes)
    return model


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_transform,val_transform = get_transform(args)

    train_dataset = DriveDataset(args.data_path,
                                 train=True,
                                 transforms=train_transform)

    val_dataset = DriveDataset(args.data_path,
                               train=False,
                               transforms=val_transform)

    num_workers = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True,
                                               drop_last = True
                                               )

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=num_workers,
                                             pin_memory=True)

    model = create_model(num_classes=args.num_classes)
    model.to(device)

    #收集参数
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(params_to_optimize,lr=args.lr,weight_decay = args.weight_decay)
    #混合精度训练
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    #创建学习率更新策略，这里是每个step更新一次(不是每个epoch)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)


    best_dice = 0.
    start_time = time.time()

    df = pd.DataFrame(columns = ['train_loss', 'val_loss','miou','recall','precision','glob_acc','acc','dice'])

    if not os.path.exists('log/'):
        os.makedirs('log/')

    for epoch in range(args.start_epoch, args.epochs):
        train_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch, args.num_classes,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)

        confmat,dice,val_loss= evaluate(model, optimizer,val_loader, device=device, num_classes=args.num_classes)

        val_info = str(confmat)
        val_info += f"\ndice coefficient: {dice*100:.3f}\n" \
                    f"train loss : {train_loss:.3f}\n" \
                    f"==========================================\n"
        print(val_info)

        matrix = confmat.compute()
        df.loc[epoch] = [
            f'{train_loss:.5f}',
            f'{val_loss:.5f}',
            f'{matrix[0]:.5f}',  # miou
            f'{matrix[1][1:].mean():.5f}',  # recall
            f'{matrix[2][1:].mean():.5f}',  # precision
            f'{matrix[3]:.5f}',
            f'{matrix[4][1:].mean():.5f}',
            f'{dice:5f}'
        ]

        df.to_csv('log/log.csv.', index = False)

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}

        if best_dice < dice:
            best_dice = dice
            torch.save(save_file, "save_weights/best_model.pth")
        else:
            continue


        if args.amp:
            save_file["scaler"] = scaler.state_dict()


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))




def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch unet training")
    parser.add_argument("--device", default = "cuda", help = "training device")

    #数据根目录
    parser.add_argument("--data-path", default="./data")
    #加上背景
    parser.add_argument("--num-classes", default=2, type=int)
    #如果显存不够，batch-size可以设置得小一些
    parser.add_argument("-b", "--batch-size", default=2, type=int)
    parser.add_argument("--epochs", default=100, type=int, metavar="N",help="epochs")


    #学习率
    parser.add_argument('--lr', default=5e-5, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-8, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    #从那一个epoch开始训练
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    main(args)
