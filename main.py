# -*- coding: utf-8 -*

import os
import time
import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from bcnn import BCNN
from PIL import ImageFile
from Imagefolder_modified import Imagefolder_modified
import pickle
import math
import self_correcter
import warnings
warnings.filterwarnings('ignore')
ImageFile.LOAD_TRUNCATED_IMAGES = True

torch.manual_seed(0)
torch.cuda.manual_seed(0)

os.popen('mkdir -p model')


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--base_lr', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--drop_rate', type=float, default=0.25)
parser.add_argument('--weight_decay', type=float, default=1e-8)
parser.add_argument('--n_classes', type=int, default=200)
parser.add_argument('--step', type=int, default=None)
parser.add_argument('--resume', type=str,default=False)
parser.add_argument('--queue_size',type=int, default=5)
parser.add_argument('--warm_up',type=int, default=5)
args = parser.parse_args()

data_dir = args.dataset
learning_rate = args.base_lr
batch_size = args.batch_size
num_epochs = args.epoch
weight_decay = args.weight_decay
step = args.step
drop_rate= args.drop_rate
num_classes=args.n_classes
queue_size=args.queue_size
resume = args.resume
warm_up = args.warm_up


logfile ='training_log.txt'

# Load data
train_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=448),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomCrop(size=448),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])
test_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=448),
    torchvision.transforms.CenterCrop(size=448),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

root = '/home/dmy/data/fg-web-data'
print(os.path.join(root,data_dir, 'train'))
train_data = Imagefolder_modified(os.path.join(root,data_dir, 'train'), transform=train_transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_data = torchvision.datasets.ImageFolder(os.path.join(root,data_dir, 'val'), transform=test_transform)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
num_train_images = len(train_data)
print("num_train_images:",num_train_images)
print("test_data_images:",len(test_data))

mom = 0.9

def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)

# Train the Model
def train(train_loader, epoch, model, optimizer,warm,correcter=None):
    train_total = 0
    train_correct = 0

    # training in each mini-batch
    for it, (images, labels, ids, path) in enumerate(train_loader):
        torch.cuda.empty_cache()
        iter_start_time = time.time()
        images = images.cuda()
        labels = labels.cuda()

        logits = model(images)

        outputs = F.softmax(logits, dim=1)
        _, prec = torch.max(outputs.data, 1)
        softmax_matrix = outputs.cpu().detach().numpy()

        if warm:
            loss_array = F.cross_entropy(logits, labels,
                                         reduce=False)

            loss = torch.sum(loss_array) / ids.shape[0]
        else:
            log_P = torch.log(outputs)
            one_hot_t = torch.zeros(outputs.shape).cuda().scatter_(1, torch.unsqueeze(labels, dim=1), 1)
            one_hot_z = torch.zeros(outputs.shape).cuda().scatter_(1, torch.unsqueeze(prec, dim=1), 1)

            for i in range(len(ids)):
                id = ids[i].item()
                f_x = correcter.certainty_array[id]
                loss1 = -(f_x * one_hot_t[i] + (1.0 - f_x) * one_hot_z[i]) * log_P[i]
                loss1 = torch.unsqueeze(loss1, 0)
                if i == 0:
                    loss_array = loss1
                else:
                    loss_array = torch.cat((loss_array, loss1), 0).cuda()

            loss_array = loss_array.sum(1)
            loss = torch.sum(loss_array) / ids.shape[0]

        correcter.async_update_prediction_matrix(ids, softmax_matrix,loss_array.data)

        if not warm:
            new_ids, new_images, new_labels = correcter.patch_clean_with_corrected_sample_batch(ids.cpu().numpy(),images.cpu().numpy(),
                                                                                                labels.cpu().numpy())
            if len(new_ids) ==0:
                continue
            ids = torch.from_numpy(np.array(new_ids)).cuda()
            new_images = torch.from_numpy(np.array(new_images)).cuda()
            new_labels = torch.from_numpy(np.array(new_labels)).cuda()

            logits = model(new_images)
            labels = new_labels

            outputs = F.softmax(logits, dim=1)
            _, prec = torch.max(outputs.data, 1)

            log_P = torch.log(outputs)
            one_hot_t = torch.zeros(outputs.shape).cuda().scatter_(1, torch.unsqueeze(labels, dim=1), 1)
            one_hot_z = torch.zeros(outputs.shape).cuda().scatter_(1, torch.unsqueeze(prec, dim=1), 1)

            for i in range(len(ids)):
                id = ids[i].item()
                f_x = correcter.certainty_array[id]
                loss1 = -(f_x * one_hot_t[i] + (1.0 - f_x) * one_hot_z[i]) * log_P[i]
                loss1 = torch.unsqueeze(loss1, 0)

                if i == 0:
                    loss = loss1
                else:
                    loss = torch.cat((loss, loss1), 0).cuda()
            loss = loss.sum()

        train_total += len(ids)
        train_correct_batch = (prec == labels).sum()
        train_correct += train_correct_batch

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iter_end_time = time.time()

        print('Epoch:[{0:03d}/{1:03d}]  Iter:[{2:04d}/{3:04d}]  '
              'Train Accuracy :[{4:6.2f}]  Loss :[{6:4.4f}] '
              'Iter Runtime:[{6:6.2f}]'
              'training number:[{7:03d}]'.format(
            epoch + 1, num_epochs, it + 1, len(train_data)// batch_size,
            (1.0*train_correct_batch.item())/len(ids), loss.item(),
            iter_end_time - iter_start_time,len(ids)))

    train_acc = float(train_correct) / float(train_total)
    return train_acc,train_total


def evaluate(test_loader, model):
    model.eval()  # Change model to 'eval' mode.
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.cuda()
        logits = model(images)
        outputs = F.softmax(logits, dim=1)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred.cpu() == labels).sum().item()

    acc = 100 * float(correct) / float(total)
    return acc

def main():
    # step = args.step
    print('===> About training in a two-step process! ===')
    print('------\n'
          'drop rate: [{}]\t'    
          '\n------'.format(drop_rate))

    # step 1: only train the fc layer
    if step == 1:
        print('===> Step 1 ...')
        bnn = BCNN(pretrained=True, n_classes=num_classes)
        bnn = nn.DataParallel(bnn).cuda()
        optimizer = optim.Adam(bnn.module.fc.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # step 1: train the whole network
    elif step == 2:
        print('===> Step 2 ...')
        bnn = BCNN(pretrained=False, n_classes=num_classes)
        bnn = nn.DataParallel(bnn).cuda()
        optimizer = optim.Adam(bnn.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise AssertionError('Wrong step argument')

    correcter = self_correcter.Correcter(num_train_images, num_classes, queue_size)

    loadmodel = 'checkpoint.pth'

    # check if it is resume mode
    print('-----------------------------------------------------------------------------')
    if resume:
        assert os.path.isfile(loadmodel), 'please make sure checkpoint.pth exists'
        print('---> loading checkpoint.pth <---')
        checkpoint = torch.load(loadmodel)
        assert step == checkpoint['step'], 'step in checkpoint does not match step in argument'
        start_epoch = checkpoint['epoch']
        best_accuracy = checkpoint['best_accuracy']
        best_epoch = checkpoint['best_epoch']
        bnn.load_state_dict(checkpoint['bnn_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        correcter.all_predictions = (checkpoint['all_predictions'])
        correcter.softmax_record = (checkpoint['softmax_record'])
        correcter.update_counters = (checkpoint['update_counters'])

    else:
        if step == 2:
            print('--->        step2 checkpoint loaded         <---')
            bnn.load_state_dict(torch.load('model/bnn_step1_vgg16_best_epoch.pth'))
        else:
            print('--->        no checkpoint loaded         <---')

        start_epoch = 0
        best_accuracy = 0.0
        best_epoch = None

    print('-----------------------------------------------------------------------------')

    with open(logfile, "a") as f:
        f.write('------ Step: {} ...\n'.format(step))
        f.write('------\n'
              'drop rate: [{}]\tqueue_size: [{}]\t'
              'warm_up: [{}]\tinit_lr: [{}]\t'
              '\n'.format(drop_rate,queue_size,warm_up,learning_rate))

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5,patience=4, verbose=True, threshold=learning_rate*1e-3)

    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()

        bnn.train()
        
        if epoch<warm_up:
            warm = True
        else:
            warm = False

        if not warm:
            correcter.separate_clean_and_unclean_keys(drop_rate)
            print("干净的样本数：",len(correcter.clean_key))

        train_acc,train_total = train(train_loader, epoch, bnn, optimizer,warm,correcter=correcter)

        test_acc = evaluate(test_loader, bnn)
        if not warm:
            scheduler.step(test_acc)

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_epoch = epoch + 1
            torch.save(bnn.state_dict(), 'model/bnn_step{}_vgg16_best_epoch.pth'.format(step))

        epoch_end_time = time.time()
        print("all_predictions", len(correcter.all_predictions[0]))
        print("update_counters", correcter.update_counters[0])
        save_checkpoint({
            'epoch': epoch + 1,
            'bnn_state_dict': bnn.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_epoch': best_epoch,
            'best_accuracy': best_accuracy,
            'step': step,
            'all_predictions':correcter.all_predictions,
            'softmax_record':correcter.softmax_record,
            'update_counters': correcter.update_counters
        },filename=loadmodel)

        print('------\n'
              'Epoch: [{:03d}/{:03d}]\tTrain Accuracy: [{:6.2f}]\t'
              'Test Accuracy: [{:6.2f}]\t'
              'Epoch Runtime: [{:6.2f}]\t'\
              '\n------'.format(
            epoch + 1, num_epochs, train_acc, test_acc,
            epoch_end_time - epoch_start_time))
        with open(logfile, "a") as f:
            output = 'Epoch: [{:03d}/{:03d}]\tTrain Accuracy: [{:6.2f}]\t' \
                     'Test Accuracy: [{:6.2f}]\t' \
                     'Epoch Runtime: [{:7.2f}]\tTrain_total[{:06d}]\tclean_key[{:06d}]'.format(
                epoch + 1, num_epochs, train_acc, test_acc,
                epoch_end_time - epoch_start_time,train_total,len(correcter.clean_key))
            f.write(output + "\n")

    print('******\n'
          'Best Accuracy 1: [{0:6.2f}], at Epoch [{1:03d}] '
          '\n******'.format(best_accuracy, best_epoch))
    with open(logfile, "a") as f:
        output = '******\n' \
                 'Best Accuracy 1: [{0:6.2f}], at Epoch [{1:03d}]; ' \
                 '\n******'.format(best_accuracy, best_epoch)
        f.write(output + "\n")


if __name__ == '__main__':
    main()
