import os
import copy
import torch
import numpy as np
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils.learning_rate_scheduler import CosineAnnealingWarmUpRestarts
from utils.dataset import TimeSeriesDataset
from utils.dataloader import load_dataset
from utils.opt import Option
from models.tdf_extractor import TDF_Extractor
from models.raw_signal_feature_extractor import Raw_Signal_Convolution_Block
from torchvision.models import resnet34
from models.raw_signal_feature_extractor import Raw_Signal_Feature_Extractor
from models.model import Method


def main(opt):

    device = opt.cuda_idx if torch.cuda.is_available() else 'cpu'

    ##### Load Dataset #####
    print(f'>>> Load Dataset!')
    X_train, X_test, y_train, y_test = load_dataset(random_state=opt.random_seed, path=opt.data_dir, noise=opt.snr)
    
    test_dataset = TimeSeriesDataset(X_test, y_test, opt.snr)
    
    test_batch_size = opt.batch_size # 256
    
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    print(f'>>> Successfully load dataset!')

    ##### Load Model #####
    print(f">>> Load Model!")

    TDFE = TDF_Extractor()

    RSCB = Raw_Signal_Convolution_Block()
    resnet = resnet34()
    resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    resnet.fc = nn.Linear(512, 6)
    RSFE = Raw_Signal_Feature_Extractor(RSCB, resnet)

    model = Method(TDFE, RSFE).to(device)

    # model_path = './checkpoints/96_269.pt'  # Insert model path
    model.load_state_dict(torch.load(os.path.join(opt.ckpt, opt.model_name)))
    model = model.to(device)

    model.eval()
    test_acc = 0
    test_loss = 0
    for batch, label in test_dataloader:
        tdf_batch, raw_batch, = Variable(batch[:, :12].unsqueeze(dim=1).float().to(device)), Variable(batch[:, 12:].unsqueeze(dim=1).float().to(device))
        label = Variable(label.view(-1).type(torch.LongTensor).to(device))

        output = model(tdf_batch, raw_batch) 

        _, preds = torch.max(output, 1)
        test_acc += torch.sum(preds == label).item() / len(label)

    test_acc = test_acc / len(test_dataloader) * 100
    test_loss = test_loss / len(test_dataloader)

    print(f'Evaluation Accuracy: {(test_acc)}')


if __name__ == '__main__':
    
    option = Option().parse()

    main(opt=option)
          