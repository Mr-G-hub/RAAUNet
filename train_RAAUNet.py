import argparse
import os
import copy
import time
import scipy.io
import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from RAAUNet import RAAUNet
from datasets import TrainDataset, EvalDataset
from utils import AverageMeter, calc_psnr, EarlyStopping, TVLoss


if __name__ == '__main__':
    model_name = 'RAAUNet'
    dataset_name = 'HRSID-image-512x512-all'
    loss_name = 'L1+TV'
    parser = argparse.ArgumentParser()
    # dataset parameters
    parser.add_argument('--train-file', type=str, default='E:/guoaca1/PytorchCodes/1bitSAR-SRCNN-pytorch-master/input/TrainSet_HRSID_512_all.h5')
    parser.add_argument('--eval-file', type=str, default='E:/guoaca1/PytorchCodes/1bitSAR-SRCNN-pytorch-master/input/EvalSet_HRSID_512_all.h5')
    parser.add_argument('--num-workers', type=int, default=8)
    # output path
    parser.add_argument('--outputs-dir', type=str, default='./output/')
    # train parameters
    parser.add_argument('--lamda', type=float, default=1e-3)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-epochs', type=int, default=3)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--scale', type=int, default=3)
    args = parser.parse_args()

    file_name = 'Model({:s})-Dataset({:s})-Loss({:s}+lamda{:f})-Epochs{:d}-Batch_size{:d}-lr{:f}-Try{}'\
        .format(model_name, dataset_name, loss_name, args.lamda, args.num_epochs, args.batch_size, args.lr, args.scale)

    args.outputs_dir = os.path.join(args.outputs_dir, file_name)
    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = RAAUNet().to(device)
    
    criterion = nn.L1Loss()
    TVloss = TVLoss(args.lamda)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)

    early_stopping = EarlyStopping(args.patience, verbose=True)
    model_save_path = os.path.join(args.outputs_dir, 'best_early_stopping.pth')
    early_stopping.path = model_save_path
    trig = 0

    train_dataset = TrainDataset(args.train_file)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True)

    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset,
                                 batch_size=1,
                                 shuffle=False)

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0
    loss_avg = []
    psnr_avg = []

    for epoch in range(0, args.num_epochs):
        tc = time.time()

        # train
        model.train()
        epoch_losses = AverageMeter()
        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size)) as t:
            t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))

            for data in train_dataloader:
                obits, invs = data
                obits = obits.type(torch.FloatTensor).to(device)
                invs = invs.type(torch.FloatTensor).to(device)

                recon_image = model(obits)

                if args.lamda == 0:
                    loss = criterion(recon_image, invs)
                else:
                    tv = TVloss(recon_image)
                    loss = criterion(recon_image, invs) + tv

                epoch_losses.update(loss.item(), len(obits))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(obits))

        loss_avg.append(epoch_losses.avg)
        scipy.io.savemat((args.outputs_dir + '/loss_avg.mat'), mdict={'loss_avg': loss_avg})

        # eval
        model.eval()
        epoch_psnr = AverageMeter()
        for data in eval_dataloader:
            obits, invs = data
            obits = obits.type(torch.FloatTensor)
            invs = invs.type(torch.FloatTensor)
            obits = obits.to(device)
            invs = invs.to(device)

            with torch.no_grad():
                recon_image = model(obits)

            recon_image = recon_image.clamp(0.0, 1.0)
            epoch_psnr.update(calc_psnr(recon_image, invs), len(obits))

        psnr_avg.append(epoch_psnr.avg.cpu())
        scipy.io.savemat((args.outputs_dir + '/psnr_avg.mat'), mdict={'psnr_avg': psnr_avg})

        # update weights of model
        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, os.path.join(args.outputs_dir, 'best.pth'))

        # early stopping
        early_stopping(epoch_psnr.avg*(-1), model)
        if early_stopping.early_stop:
            trig = trig + 1
            model = torch.load(model_save_path)
            args.lr = args.lr / 10
            print('%s: lr = %s' % (str(trig), str(args.lr)))
            optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
            if trig > 1:
                print("Early stopping")
                break
            early_stopping.early_stop = False
            early_stopping.counter = 0

        te = time.time()
        print('eval psnr: {:.2f} \n Total time: {:.2f}'.format(epoch_psnr.avg, (te - tc)))

    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))


