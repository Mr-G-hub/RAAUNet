import os
import argparse
import time
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader

from RAAUNet import RAAUNet
from datasets import EvalDataset
from utils import *

if __name__ == '__main__':
    model_name = 'RAAUNet'
    dataset_name = 'HRSID-image-512x512-all'
    loss_name = 'L1+TV'
    parser = argparse.ArgumentParser()
    # load model path
    parser.add_argument('--weights-file', type=str, default='./output/')
    # dataset parameters
    parser.add_argument('--eval-file', type=str, default='E:/guoaca1/PytorchCodes/1bitSAR-SRCNN-pytorch-master/input/EvalSet_HRSID_512_all.h5')
    # output path
    parser.add_argument('--test-output-dir', type=str, default='./output/')
    # train parameters
    parser.add_argument('--lamda', type=float, default=1e-3)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-epochs', type=int, default=3)
    parser.add_argument('--scale', type=int, default=3)
    args = parser.parse_args()

    file_name = 'Model({:s})-Dataset({:s})-Loss({:s}+lamda{:f})-Epochs{:d}-Batch_size{:d}-lr{:f}-Try{}' \
        .format(model_name, dataset_name, loss_name, args.lamda, args.num_epochs, args.batch_size, args.lr, args.scale)

    args.weights_file = os.path.join(args.weights_file, '{:s}/best.pth'.format(file_name))

    args.test_output_dir = os.path.join(args.test_output_dir, '{:s}/{:s}'.format(model_name, file_name))
    if not os.path.exists(args.test_output_dir):
        os.makedirs(args.test_output_dir)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = RAAUNet().to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset,
                                 batch_size=1,
                                 shuffle=False)

    model.eval()
    psnr_vec = []
    tc = time.time()

    for i, data in enumerate(eval_dataloader):
        obits, invs = data
        obits = obits.type(torch.FloatTensor).to(device)
        invs = invs.type(torch.FloatTensor).to(device)

        with torch.no_grad():
            preds = model(obits)
            preds = preds.clamp(0.0, 1.0)

        psnr = calc_psnr(invs, preds)
        psnr_vec.append(psnr)
        print('Num: {:d} \t PSNR: {:.2f}'.format(i, psnr))

    te = time.time()
    print('Test time: {:.2f} \n PSNR_avg: {:.2f}'.format(te - tc, sum(psnr_vec)/len(psnr_vec)))

