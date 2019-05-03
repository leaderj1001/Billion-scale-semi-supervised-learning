import torch
import torch.optim as optim
import torch.nn as nn
import cv2
import numpy as np
import os
import glob
import argparse
import json
import random
import codecs
import time

from model import resnet50

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


def get_args():
    parser = argparse.ArgumentParser('parameters')

    parser.add_argument('--epochs', type=int, default=100, help='number of epochs, (default: 100)')
    parser.add_argument('--learning-rate', type=float, default=1e-1, help='learning rate, (default: 1e-1)')
    parser.add_argument('--batch-size', type=int, default=100, help='batch size, (default: 100)')
    parser.add_argument('--dataset-mode', type=str, default="CIFAR100", help='dataset, (default: CIFAR100)')
    parser.add_argument('--input-size', type=tuple, default=(32, 32), help='input data size, (default: (32, 32))')

    args = parser.parse_args()

    return args


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def unlabeled_batch_iterator(batch_size=100, shape=(32, 32)):
    with codecs.open("./selected_image.json", "r", encoding="utf-8", errors="ignore") as f:
        json_data = json.load(f)

    image_list = json_data["all"]
    random.shuffle(image_list)
    while len(image_list) != 0:
        batch_keys = image_list[:batch_size]

        images = []
        labels = []

        for key in batch_keys:
            image = cv2.imread(key[0])
            image = cv2.resize(image, dsize=shape)

            images.append(image)
            labels.append(key[1])

        images = np.array(images)
        images = np.reshape(images, newshape=[-1, 3, 32, 32])
        labels = np.array(labels)
        yield images, labels

        del image_list[:batch_size]


def train(model, optimizer, criterion, epoch, args):
    model.train()
    step = 0
    train_loss = 0
    train_acc = 0
    for batch_image, batch_label in unlabeled_batch_iterator(args.batch_size, args.input_size):
        adjust_learning_rate(optimizer, epoch, args)
        data, target = torch.cuda.FloatTensor(batch_image), torch.cuda.LongTensor(batch_label)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.data
        y_pred = output.data.max(1)[1]

        acc = float(y_pred.eq(target.data).sum()) / len(data) * 100.
        train_acc += acc
        step += 1
        if step % 100 == 0:
            print("[Epoch {0:4d}] Loss: {1:2.3f} Acc: {2:.3f}%".format(epoch, loss.data, acc), end='')
            for param_group in optimizer.param_groups:
                print(",  Current learning rate is: {}".format(param_group['lr']))

    return train_loss / 24, train_acc / 24


def main(args):
    model = resnet50().to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=5e-4, momentum=0.9)
    criterion = nn.CrossEntropyLoss().to(device)

    start_time = time.time()
    max_train_acc = 0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train(model, optimizer, criterion, epoch, args)
        print('train set accuracy: {0:.2f}%, Best train accuracy: {1:.2f}%'.format(train_acc, max_train_acc))
        if max_train_acc < train_acc:
            print('Saving..')
            state = {
                'model': model.state_dict(),
                'acc': train_acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            filename = "student_network_Best_model_"
            torch.save(state, './checkpoint/' + filename + 'ckpt.t7')
            max_train_acc = train_acc

        time_interval = time.time() - start_time
        time_split = time.gmtime(time_interval)
        print("Training time: ", time_interval, "Hour: ", time_split.tm_hour, "Minute: ", time_split.tm_min, "Second: ",
              time_split.tm_sec)


if __name__ == "__main__":
    args = get_args()
    main(args)
