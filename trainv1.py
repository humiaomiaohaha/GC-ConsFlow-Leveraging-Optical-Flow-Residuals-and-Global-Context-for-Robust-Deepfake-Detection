import sys, os
import argparse
import torch
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
import pickle
from torch.utils.data import Dataset, DataLoader
from time import perf_counter
from datetime import datetime
# from dataset.dataset import VideoFrameDataset
from dataset.datasetv2 import VideoFrameFlowDataset
import optparse
from torchsummary import summary
import random
import numpy as np
#from model.flow import flowNet
import torchvision
from torchvision import transforms
#import tensorflow as tf
#from model.flownetv3 import flowNet

from model.flowv1 import flowNet
#from model.Chen import ModelBlock

from sklearn.metrics import roc_auc_score


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.cuda.manual_seed(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
   # tf.random.set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


aug_train = torchvision.transforms.Compose([
    transforms.Resize((224,224)),
    torchvision.transforms.RandomCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.25, 0.25, 0.25])
])

aug_test = torchvision.transforms.Compose([
    transforms.Resize((224,224)),
    torchvision.transforms.RandomCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.25, 0.25, 0.25])
])


set_random_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = flowNet()
model.to(device)
print(f'using {device}')

def train(num_epochs, test_model, batch_size, lr, weight_decay):
    set_random_seed(42)
    DatasetTrain=VideoFrameFlowDataset('/your_frame_dataroot/train',
                                       '/your_flow_dataroot/train',
                                       num_frames=6,
                                       transform=aug_train)
    # DatasetTrain = VideoFrameDataset(root_dir='/data/DeepfakeDatasets/c40_video/C40df/train', num_frames=5, transform=aug_train)
    dataloaderTrain = DataLoader(DatasetTrain, batch_size=batch_size, shuffle=True,num_workers=4)

    DatasetVal= VideoFrameFlowDataset('/your_frame_dataroo/validation',
                                      '/your_flow_dataroot/validation',
                                      num_frames=6,
                                      transform=aug_test)
    # DatasetVal= VideoFrameDataset(root_dir='/data/DeepfakeDatasets/c40_video/C40df/validation', num_frames=5, transform=aug_test)
    dataloaderVal = DataLoader(DatasetVal, batch_size=batch_size, shuffle=True, num_workers=4)

    DatasetTest = VideoFrameFlowDataset('/your_frame_dataroo/test',
                                       '/your_flow_dataroot/test',
                                        num_frames=6,
                                       transform=aug_test)
    # DatasetTest = VideoFrameDataset(root_dir='/data/DeepfakeDatasets/c40_video/C40df/test', num_frames=5, transform=aug_test)
    dataloaderTest = DataLoader(DatasetTest, batch_size=batch_size, shuffle=True, num_workers=4)

    set_random_seed(42)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)
    min_val_loss = 10000
    scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_train_acc = 0.0
    best_val_acc = 0.0
    best_test_acc = 0.0
    best_test_auc=0.0
 
    min_loss = min_val_loss

    train_loss = []
    train_accu = []
    val_loss = []
    val_accu = []
    # test_accu=[]

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        model.train()

        running_loss = 0.0
        running_corrects = 0
        train_all_preds = []
        train_all_labels = []
        y_pred=[]
        # frames, flows, label, frame_paths, flow_paths
        for frames, flows, label, frame_paths, flow_paths in dataloaderTrain:

            # print(label)
            frames = frames.to(device)
            flows = flows.to(device)
            label = label.to(device)
            # print(frames.shape)
            # print(flows.shape)

            outputs = model(frames, flows)
            # print(outputs)

            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * frames.size(0)
            running_corrects += torch.sum(preds == label.data)
            train_all_preds.extend(outputs[:, 1].detach().cpu().numpy())  # 假设是二分类问题，取第1列的概率值
            train_all_labels.extend(label.cpu().numpy())

        epoch_loss=running_loss / len(DatasetTrain)
        epoch_acc = running_corrects.float() / len(DatasetTrain)
        try:
            train_auc = roc_auc_score(train_all_labels, train_all_preds)
        except ValueError as e:
            print(f"Error calculating AUC: {e}")
            train_auc = 0.0
        if epoch_acc>best_train_acc:
            best_train_acc = epoch_acc

        train_loss.append(epoch_loss)
        train_accu.append(epoch_acc)
        print('{} Loss: {:.4f} Acc: {:.4f} AUC: {:.4f}'.format('train', epoch_loss, epoch_acc, train_auc))
        # print('{} Loss: {:.4f} Acc: {:.4f}'.format('train', epoch_loss, epoch_acc))

        model.eval()

        running_loss = 0.0
        running_corrects = 0

        for frames, flows, label, frame_paths, flow_paths in dataloaderVal:
            frames = frames.to(device)
            flows = flows.to(device)
            label = label.to(device)

            with torch.no_grad():
                outputs = model(frames, flows)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, label)

            running_loss += loss.item() * frames.size(0)
            running_corrects += torch.sum(preds == label.data)
        epoch_loss = running_loss / len(DatasetVal)
        epoch_acc = running_corrects.float() / len(DatasetVal)
        if epoch_acc > best_val_acc:
            best_val_acc = epoch_acc
        val_loss.append(epoch_loss)
        val_accu.append(epoch_acc)
        print('{} Loss: {:.4f} Acc: {:.4f}'.format('val', epoch_loss, epoch_acc))

        if epoch_loss < min_loss:
            print('\nValidation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(epoch_loss, min_loss))
            min_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            curr_time = datetime.now().strftime("%B_%d_%Y_%H_%M_%S")
            torch.save(best_model_wts, f'/your_dataroot/GC-ConsFlow_{curr_time}.pth')

        scheduler.step()
        #test(model, dataloaderTest, len(DatasetTest))





    #time_elapsed = time.time() - since
 #   print(f'\nBest Train Accuracy over all epochs: {best_train_acc:.4f}')
 #   print(f'Best Validation Accuracy over all epochs: {best_val_acc:.4f}')
 #   print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)

    # with open('weight/cvit_deepfake_detection_v2.pkl', 'wb') as f:
    # pickle.dump([train_loss, train_accu, val_loss, val_accu], f)
    if test_model:
        test(model, dataloaderTest, len(DatasetTest))




def test(model, dataloaders, dataset_sizes):
    model.eval()
    correct_predictions = 0
    test_all_preds = []
    test_all_labels = []
    test_accu=[]
    for frames, flows, label, frame_paths, flow_paths in dataloaders:
        frames = frames.to(device)
        flows = flows.to(device)
        label = label.to(device)
        outputs = model(frames, flows)
        _, preds = torch.max(outputs, 1)
        correct_predictions += (preds == label).sum().item()
        test_all_preds.extend(outputs[:, 1].detach().cpu().numpy())
        test_all_labels.extend(label.cpu().numpy())

    test_acc = correct_predictions / dataset_sizes
    test_accu.append(test_acc)
    try:
        test_auc = roc_auc_score(test_all_labels, test_all_preds)
    except ValueError as e:
        print(f"Error calculating AUC: {e}")
        test_auc = 0.0
    print('Test Set AUC: {:.4f}'.format(test_auc))
    print('Test Set Prediction: ', (correct_predictions / dataset_sizes) * 100, '%')




def gen_parser():
    parser = optparse.OptionParser("Train CViT model.")

    parser.add_option("-e", "--epoch", type=int, dest='epoch',
                      help='Number of epochs used for training the CViT model.')

    parser.add_option("-b", "--batch", type=int, dest='batch', help='Batch size.')
    parser.add_option("-l", "--rate", type=float, dest='rate', help='Learning rate.')
    parser.add_option("-w", "--wdecay", type=float, dest='wdecay', help='Weight decay.')
    parser.add_option("-t", "--test", type=str, dest='test', help='Test on test set.')

    (options, _) = parser.parse_args()


    num_epochs = options.epoch if options.epoch else 1
    test_model = "y" if options.test else None
    batch_size = options.batch if options.batch else 32
    lr = float(options.rate) if options.rate else 0.0001
    weight_decay = float(options.wdecay) if options.wdecay else 0.0000001

    return num_epochs, test_model, int(batch_size), lr, weight_decay


def main():
    start_time = perf_counter()
    set_random_seed(42)
    num_epochs, test_model, batch_size, lr, weight_decay = gen_parser()
    print('Training Configuration:')

    print(f'\nepoch: {num_epochs}')
    print(f'\ntest_model: {test_model}')
    print(f'\nbatch_size: {batch_size}')
    set_random_seed(42)
    train(num_epochs, test_model, batch_size, lr, weight_decay)
    end_time = perf_counter()

    print("\n\n--- %s seconds ---" % (end_time - start_time))


if __name__ == "__main__":
    main()

