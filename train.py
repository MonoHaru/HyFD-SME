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
from models.raw_signal_feature_extractor import Raw_Signal_Feature_Extractor
from models.model import Method
from torchvision.models import resnet34
from torch.autograd import Variable


def main(opt):

    device = opt.cuda_idx if torch.cuda.is_available() else 'cpu'

    ##### Load Dataset #####
    print(f'>>> Load Dataset!')
    X_train, X_test, y_train, y_test = load_dataset(random_state=opt.random_seed, path=opt.data_dir, noise=opt.snr)

    train_dataset = TimeSeriesDataset(X_train, y_train, opt.snr)
    test_dataset = TimeSeriesDataset(X_test, y_test, opt.snr)

    train_batch_size = opt.batch_size

    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=train_batch_size, shuffle=False)

    print(f"Train {train_dataset.call()}")
    print(f"Test {test_dataset.call()}")
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

    model_name = 'Ours.pt'
    best_model_wts = copy.deepcopy(model.state_dict())
    optimizer = optim.AdamW(model.parameters(), lr=0)
    criterion = nn.CrossEntropyLoss()
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=opt.T_0, T_mult=opt.T_mult, eta_max=opt.eta_max, T_up=opt.T_up, gamma=opt.gamma)
    print(f'>>> Successfully load model!')

    num_epochs = opt.epoch
    best_acc = 0
    itera = 0
    for epoch in range(num_epochs):
        itera += 1

        # Train Phase
        model.train()
        train_acc = 0
        train_loss = 0
        for batch, label in train_dataloader:
            tdf_batch, raw_batch, = Variable(batch[:, :12].unsqueeze(dim=1).float().to(device)), Variable(batch[:, 12:].unsqueeze(dim=1).float().to(device))
            label = Variable(label.view(-1).type(torch.LongTensor).to(device))

            optimizer.zero_grad()
            output = model(tdf_batch, raw_batch)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(output, 1)
            train_acc += torch.sum(preds == label).item() / len(label)

            train_loss += loss.item()

        train_acc = train_acc / len(train_dataloader) * 100
        train_loss = train_loss / len(train_dataloader)

        print('Epoch : {0:>4}  |  Train Acc : {1:>7.3f}  |  Train Loss : {2:>7.5f}'.format(epoch, train_acc, train_loss))

        # Test Phase
        model.eval()
        val_acc = 0
        val_loss = 0
        for batch, label in test_dataloader:
            tdf_batch, raw_batch, = Variable(batch[:, :12].unsqueeze(dim=1).float().to(device)), Variable(batch[:, 12:].unsqueeze(dim=1).float().to(device))
            label = Variable(label.view(-1).type(torch.LongTensor).to(device))

            output = model(tdf_batch, raw_batch) 
            loss = criterion(output, label)

            _, preds = torch.max(output, 1)
            val_acc += torch.sum(preds == label).item() / len(label)

            val_loss += loss.item()
        
        val_acc = val_acc / len(test_dataloader) * 100
        val_loss = val_loss / len(test_dataloader)

        print('Epoch : {0:>4}  |  Val   Acc : {1:>7.3f}  |  Val   Loss : {2:>7.5f}'.format(epoch, val_acc, val_loss))

        if epoch == 0 or val_acc > best_acc:
            itera = 0
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            acc_str = str(np.round(best_acc, 3)).replace('.', '_')
            torch.save(best_model_wts, os.path.join(opt.ckpt, 'epoch_{}_{}.pt'.format(epoch, acc_str)))
            print(f'>>> Best Epoch : {epoch:>4}  |  Acc : {(best_acc):>7.3f}')
            
        scheduler.step()

        if itera > 100:
            break

    return 0


if __name__ == '__main__':

    option = Option().parse()

    main(opt=option)