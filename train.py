import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import time
import platform
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from network.tinykan import *
from network.data import *
from network.transform import Data_Transforms
from datetime import datetime
import shutil

from network.log_record import *
from network.utils import setup_seed, cal_metrics, plot_ROC

print('-'*20)
print("PyTorch version:{}".format(torch.__version__))
print("Python version:{}".format(platform.python_version()))
print("cudnn version:{}".format(torch.backends.cudnn.version()))
print("GPU name:{}".format(torch.cuda.get_device_name(0)))
print("GPU number:{}".format(torch.cuda.device_count()))
print('-'*20)

def main():
    args = parse.parse_args()

    name = args.name
    train_txt_path = args.train_txt_path
    valid_txt_path = args.valid_txt_path
    continue_train = args.continue_train
    epoches = args.epoches
    batch_size = args.batch_size
    model_name = args.model_name
    model_path = args.model_path
    num_classes = args.num_classes
    data_path = args.data_path
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")
    output_path = os.path.join('./output', name, dt_string)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    log_path = os.path.join(output_path)
    shutil.copy('./run.sh', log_path)
    shutil.copy('./network/tinykan.py', log_path)
    
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
    print('Training datetime: ', time_str)
    
    torch.backends.cudnn.benchmark = True

    train_data = SingleInputDataset(data_path=data_path, txt_path=train_txt_path, train_transform=Data_Transforms['train'])
    valid_data = SingleInputDataset(data_path=data_path, txt_path=valid_txt_path, valid_transform=Data_Transforms['val'])
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    model = Net()
    
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.0003, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # -----multiple gpus for training----- #
    model = nn.DataParallel(model.cuda())
    
    # -----------------------------------------define the train & val----------------------------------------- #
    best_acc = 0.0
    best_auc = 0.0
    time_open = time.time()

    for epoch in range(epoches):
        total_train_samples = 0.0
        correct_tra = 0.0
        sum_loss_tra = 0.0

        print('\nEpoch {}/{}'.format(epoch+1, epoches))
        print('-'*10)

        # -----Training----- #
        for i, data in enumerate(train_loader):
            img_train, labels_train = data
            
            # input images & labels
            img_train = img_train.cuda()
            labels_train = labels_train.cuda()  

            optimizer.zero_grad()
            model=model.train()
            
            # feed data to model
            pre_tra = model(img_train)

            # the average loss of a batch
            loss_tra = criterion(pre_tra, labels_train)
            sum_loss_tra += loss_tra.item() * labels_train.size(0)
            
            # prediction
            _, pred = torch.max(pre_tra.data, 1)

            loss_tra.backward()
            optimizer.step()

            # the correct number of prediction
            correct_tra += (pred == labels_train).squeeze().sum().cpu().numpy()

            # the number of all training samples
            total_train_samples += labels_train.size(0)

            torch.save(model.module.state_dict(), os.path.join(output_path, str(i+1) + '_iter_' + model_name))
          
            if i % 100 == 99:
                print("Training: Epoch[{:0>1}/{:0>1}] Iteration[{:0>1}/{:0>1}] Loss:{:.2f} Acc:{:.2%}".format(epoch + 1, epoches, i + 1, len(train_loader), sum_loss_tra/total_train_samples, correct_tra/total_train_samples))

        # -----save the pretrained model----- #
        if epoch+1 == epoches or (epoch + 1) % 1 ==0 :
            if multiple_gpus:
                torch.save(model.module.state_dict(), os.path.join(output_path, str(epoch+1) + '_epoches_' + model_name))
            else:
                torch.save(model.state_dict(), os.path.join(output_path, str(epoch+1) + '_epoches_' + model_name))
    
    # ----------------------------------------------------end----------------------------------------------------#
    
    # -----print the results----- #
    print('-'*20)        
    print('Best_accuracy:', best_acc)
    print('Best_AUC:', best_auc)
    # print time
    time_end = time.time() - time_open
    print('All time: ', time_end)



if __name__ == '__main__':
    parse = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument('--name', '-n', type=str, default='v4-no')
    parse.add_argument('--data_path', type=str, default = '/home/ghy/shujuji/FaceForensics++/c23')
    parse.add_argument('--train_txt_path', '-tp', type=str, default = 'train.txt')
    parse.add_argument('--valid_txt_path', '-vp', type=str, default = 'test.txt')
    parse.add_argument('--batch_size', '-bz', type=int, default=64)
    parse.add_argument('--epoches', '-e', type=int, default=20)
    parse.add_argument('--model_name', '-mn', type=str, default='sfic-resnet.pkl')
    parse.add_argument('--continue_train', type=bool, default=False)
    parse.add_argument('--model_path', '-mp', type=str, default="")
    parse.add_argument('--num_classes', '-nc', type=int, default=2)
    parse.add_argument('--seed', default=7, type=int)
    
    multiple_gpus = True
    gpus = [0]

    label_val_list = []
    predict_val_list = []
    
    main()
