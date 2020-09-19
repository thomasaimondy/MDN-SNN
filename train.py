import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
from SNN import SNN
import time
import os
import shutil
import argparse
from dataset import loaddata
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='train.py')

parser.add_argument('-gpu', type = int, default = 0)
parser.add_argument('-seed', type = int, default = 3154)
parser.add_argument('-nps', type = str, default = 'C1')
parser.add_argument('-dts', type = str, default = 'M')

opt = parser.parse_args()

torch.cuda.set_device(opt.gpu)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
torch.backends.cudnn.deterministic = True
num_epoch = 80

find = False

test_scores = []
train_scores = []
ccost_scores = []
if_tb = False

writer, hyperparams, train_dataset, test_dataset, train_loader, test_loader = loaddata(opt.dts, if_tb)
hyperparams.append(opt.nps)
task = hyperparams[5]

path = '../dynamic_data/10_' + opt.dts + str(opt.seed) + '_' + str(hyperparams[3]) + '_' + str(hyperparams[6]) # 12 - 8
place = path + '/'

if find:
    if not os.path.exists(path):
        os.mkdir(path)

print('Dataset: ' + task)
print('Random Seed: {}'.format(opt.seed))
print('Length of Training Dataset: {}'.format(len(train_dataset)))
print('Length of Test Dataset: {}'.format(len(test_dataset)))
print('Build Model')
print('Params come from ' + hyperparams[-1])

model = SNN(hyperparams)
model.cuda()
loss_function = nn.MSELoss()

for i in range(model.len):
    if find:
        layer = model.layers[i]
        np.save(place + 'layer' + str(i) + '_a0', layer.a.detach().cpu().numpy())
        np.save(place + 'layer' + str(i) + '_b0', layer.b.detach().cpu().numpy())
        np.save(place + 'layer' + str(i) + '_c0', layer.c.detach().cpu().numpy())
        np.save(place + 'layer' + str(i) + '_d0', layer.d.detach().cpu().numpy())

paras = dict(model.named_parameters())
paras_new = []
for k, v in paras.items():
    if k[9] == 'f':
        paras_new.append({'params': [v], 'lr': hyperparams[3]})
    else:
        paras_new.append({'params': [v], 'lr': hyperparams[6]})

optimizer = torch.optim.Adam(paras_new)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 50, gamma = 0.1)
cossim = torch.nn.CosineSimilarity(dim = 1, eps = 1e-6)
sigmoid = torch.nn.Sigmoid()
norm = torch.nn.BatchNorm2d(1).cuda()

def train(epoch):
    model.train()
    scheduler.step()
    print('Train Epoch ' + str(epoch + 1))
    start_time = time.time()
    total_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        if True:
        # if i < 60:
            if images.size()[0] == hyperparams[0]:
                optimizer.zero_grad()
                if images.type() == 'torch.DoubleTensor':
                    images = images.to(torch.float32)
                images = Variable(images.cuda())
                if task == 'MNIST':
                    one_hot = torch.zeros(hyperparams[0], hyperparams[2]).scatter(1, labels.unsqueeze(1), 1)
                    labels = Variable(one_hot.cuda())
                elif task == 'FashionMNIST':
                    one_hot = torch.zeros(hyperparams[0], hyperparams[2]).scatter(1, labels.unsqueeze(1), 1)
                    labels = Variable(one_hot.cuda())
                elif task == 'NETtalk':
                    labels = labels.float()
                    labels = Variable(labels.cuda())
                elif task == 'Cifar10':
                    one_hot = torch.zeros(hyperparams[0], hyperparams[2]).scatter(1, labels.unsqueeze(1), 1)
                    labels = Variable(one_hot.cuda())
                elif task == 'NMNIST':
                    one_hot = torch.zeros(hyperparams[0], hyperparams[2]).scatter(1, labels.unsqueeze(1), 1)
                    labels = Variable(one_hot.cuda())
                elif task == 'TiDigits':
                    labels = labels.long()
                    one_hot = torch.zeros(hyperparams[0], hyperparams[2]).scatter(1, labels.unsqueeze(1), 1)
                    labels = Variable(one_hot.cuda())
                elif task == 'Timit':
                    images = norm(images.unsqueeze(1))
                    images = images.squeeze(1)
                    one_hot = torch.zeros(hyperparams[0], hyperparams[2]).scatter(1, labels.unsqueeze(1), 1)
                    labels = Variable(one_hot.cuda())

                outputs, new_potential = model(images)
                loss = loss_function(outputs, labels)
                total_loss += float(loss)
                loss.backward(retain_graph = True)
                optimizer.step()

                if task != 'Timit':
                    divide = 40
                else:
                    divide = 1
                if (i + 1) % (len(train_dataset) // (hyperparams[0] * divide)) == 0:
                    print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.6f, Time: %.2f'
                                        % (epoch + 1, num_epoch, i + 1,
                                        len(train_dataset) // hyperparams[0],
                                        total_loss / (hyperparams[0] * 40),
                                        time.time() - start_time))
                    xs = epoch * 40 + ((i + 1) // (len(train_dataset) // (hyperparams[0] * 40)))
                    if if_tb:
                        writer.add_scalar('loss_train', total_loss / (hyperparams[0] * 40), xs)
                        writer.add_scalar('time_train', time.time() - start_time, xs)
                    start_time = time.time()
                    total_loss = 0

def eval(epoch, if_test):
    model.eval()
    correct = 0
    total = 0
    if if_test:
        print('Test Epoch ' + str(epoch + 1))
        loader = test_loader
        test_or_train = 'test'
    else:
        loader = train_loader
        test_or_train = 'train'

    if task == 'MNIST':
        for i, (images, labels) in enumerate(loader):
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
            outputs, _ = model(images)
            total += labels.size(0)
            pred = outputs.max(1)[1]
            correct += (pred == labels).sum()
        correct = correct.item()
    elif task == 'FashionMNIST':
        for i, (images, labels) in enumerate(loader):
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
            outputs, _ = model(images)
            total += labels.size(0)
            pred = outputs.max(1)[1]
            correct += (pred == labels).sum()
        correct = correct.item()
    elif task == 'NETtalk':
        for i, (images, labels) in enumerate(loader):
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
            outputs, _ = model(images)
            total += 1
            if outputs.max() >= 0.05:
                pos = []
                for label in range(26):
                    if (labels[0, label] != 0) or (outputs[0, label] != 0):
                        pos.append(label)
                tem_out = torch.zeros((1, len(pos)))
                tem_lab = torch.zeros((1, len(pos)))
                for label in range(len(pos)):
                    tem_out[0, label] = outputs[0, pos[label]]
                    tem_lab[0, label] = labels[0, pos[label]]
                correct += cossim(tem_out, tem_lab)
            else:
                correct += 0
    elif task == 'Cifar10':
        for i, (images, labels) in enumerate(loader):
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
            outputs, _ = model(images)
            total += labels.size(0)
            pred = outputs.max(1)[1]
            correct += (pred == labels).sum()
        correct = correct.item()
    elif task == 'NMNIST':
        for i, (images, labels) in enumerate(loader):
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
            outputs, _ = model(images)
            total += labels.size(0)
            pred = outputs.max(1)[1]
            correct += (pred == labels).sum()
        correct = correct.item()
    elif task == 'TiDigits':
        for i, (images, labels) in enumerate(loader):
            images = Variable(images.cuda())
            labels = labels.long()
            labels = Variable(labels.cuda())
            outputs, _ = model(images)
            total += labels.size(0)
            pred = outputs.max(1)[1]
            correct += (pred == labels).sum()
        correct = correct.item()
    elif task == 'Timit':
        for i, (images, labels) in enumerate(loader):
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
            outputs, _ = model(images)
            total += labels.size(0)
            pred = outputs.max(1)[1]
            correct += (pred == labels).sum()
        correct = correct.item()

    ccost = 0
    for i in range(model.len):
        ccost += model.layers[i].sumspike.sum()
    ccost_scores.append(ccost)
    acc = 100.0 * correct / total
    print(test_or_train + ' correct: %d accuracy: %.2f%% ccost: %d' % (correct, acc, ccost.data))
    if if_tb:
        writer.add_scalar('acc_' + test_or_train, acc, epoch + 1)
    if if_test:
        test_scores.append(acc)
        # if acc >= max(test_scores):
            # torch.save(model, './' + para[3])
    else:
        train_scores.append(acc)

    for i in range(model.len):
        if find:
            layer = model.layers[i]
            np.save(place + task + 'layer' + str(i) + '_a' + str(epoch + 1), layer.a.detach().cpu().numpy())
            np.save(place + task + 'layer' + str(i) + '_b' + str(epoch + 1), layer.b.detach().cpu().numpy())
            np.save(place + task + 'layer' + str(i) + '_c' + str(epoch + 1), layer.c.detach().cpu().numpy())
            np.save(place + task + 'layer' + str(i) + '_d' + str(epoch + 1), layer.d.detach().cpu().numpy())

def main():
    for epoch in range(num_epoch):
        train(epoch)
        if (epoch + 1) % 1 == 0:
            eval(epoch, if_test = True)
        if (epoch + 1) % 20 == 0:
            eval(epoch, if_test = False)
        if (epoch + 1) % 20 == 0:
            print('Best Test Accuracy in %d: %.2f%%' % (epoch + 1, max(test_scores)))
            avg = (test_scores[-1] + test_scores[-2] + test_scores[-3] + test_scores[-4] + test_scores[-5] + test_scores[-6] + test_scores[-7] + test_scores[-8] + test_scores[-9] + test_scores[-10]) / 10
            print('Average of Last Ten Test Accuracy : %.2f%%' % (avg))
    cc_scores = np.array(ccost_scores)
    np.save('./ccs/' + opt.nps + '_' + opt.dts + '_' + str(opt.seed) + '_cc', cc_scores)
    tr_scores = np.array(train_scores)
    np.save('./trs/' + opt.nps + '_' + opt.dts + '_' + str(opt.seed) + '_tr', tr_scores)
    te_scores = np.array(test_scores)
    np.save('./tes/' + opt.nps + '_' + opt.dts + '_' + str(opt.seed) + '_te', te_scores)
    if if_tb:
        writer.close()

if __name__ == '__main__':
    main()