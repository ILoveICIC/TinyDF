import os
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from network.data import TestDataset
from network.transform import Data_Transforms
from network.tinykan import Net
from network.plot_roc import plot_ROC
import os



def main():
    args = parse.parse_args()
    test_txt_path = args.test_txt_path
    batch_size = args.batch_size
    model_path = args.model_path
    num_classes = args.num_classes
    data_path = args.data_path
	
    torch.backends.cudnn.benchmark=True
	
    # -----create train&val data----- #
    test_data = TestDataset(data_path=data_path, txt_path=test_txt_path, test_transform=Data_Transforms['test'])
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    
    # -----create model----- #
    # model = MainNet(num_classes)
    model = Net()
    model.load_state_dict(torch.load(model_path, weights_only=True))
	
    if isinstance(model, nn.DataParallel):
        model = model.module
	
    model = model.cuda()
    model.eval()

    correct_test = 0.0
    total_test_samples = 0.0
    with torch.no_grad():
        from tqdm import tqdm
        tbar = tqdm(test_loader)
        for i, data in enumerate(tbar):
            img_rgb, labels_test = data

            img_rgb = img_rgb.cuda()
            labels_test = labels_test.cuda()
            
            # feed data
            pre_test = model(img_rgb)
            
            # prediction
            _, pred = torch.max(pre_test.data, 1)
            
            # the number of all testing sample
            total_test_samples += labels_test.size(0)
            
            # the correct number of prediction
            correct_test += (pred == labels_test).squeeze().sum().cpu().numpy()
            
            # compute ROC
            pre_test_abs = torch.nn.functional.softmax(pre_test, dim=1)
            pred_abs_temp = torch.zeros(pre_test_abs.size()[0])
            for m in range(pre_test_abs.size()[0]):
                pred_abs_temp[m] = pre_test_abs[m][1]

            label_test_list.extend(labels_test.detach().cpu().numpy())
            predict_test_list.extend(pred_abs_temp.detach().cpu().numpy())
            
        print("Testing Acc: {:.2%}".format(correct_test/total_test_samples))

    # ROC curve
    plot_ROC(label_test_list, predict_test_list)

if __name__ == '__main__':
    parse = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument('--batch_size', '-bz', type=int, default=64)
    parse.add_argument('--data_path', type=str, default = '/data/FaceForensics++/c23')
    parse.add_argument('--test_txt_path', '-tp', type=str, default = 'test.txt')
    parse.add_argument('--model_path', '-mp', type=str, default='TinyDF.pkl')
    parse.add_argument('--num_classes', '-nc', type=int, default=2)
    
    label_test_list = []
    predict_test_list = []
    
    main()
