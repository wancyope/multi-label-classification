#!/usr/bin/env python
from __future__ import print_function
import argparse
import random

import numpy as np

import chainer
from chainer import training
from chainer.training import extensions
import _pickle as pickle

from models import resnet_finetune_custom_multilabel
from models import pspnet_resnet_multilabel
from fashion144kstylenet_dataset import StyleNetDataset
from fashion550k_dataset import Fashion550kDataset
class PreprocessedDataset(chainer.dataset.DatasetMixin):

    def __init__(self, split, mean, crop_size_x, crop_size_y ,random=True):
        #Dataset:StyleNet or Fashion550k
        #self.base = StyleNetDataset(split)
        self.base = Fashion550kDataset(split)
        
        self.mean = mean
        self.crop_size_x = crop_size_x
        self.crop_size_y = crop_size_y
        self.random = random
        
    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        crop_size_x = self.crop_size_x
        crop_size_y = self.crop_size_y
        
        
        image, label = self.base[i]

        _, h, w = image.shape
        
        
        if self.random:
            # Randomly crop a region and flip the image
            top = random.randint(0, h - crop_size_y - 1)
            left = random.randint(0, w - crop_size_x - 1)
            if random.randint(0, 1):
                image = image[:, :, ::-1]
        else:
            # Crop the center
            top = (h - crop_size_y) // 2
            left = (w - crop_size_x) // 2
        
        
        bottom = top + crop_size_y
        right = left + crop_size_x

        image = image[:, top:bottom, left:right]
        image -= self.mean[:, top:bottom, left:right]
        image /= 255
        return image, label


class TestModeEvaluator(extensions.Evaluator):

    def evaluate(self):
        model = self.get_target('main')
        model.train = False
        ret = super(TestModeEvaluator, self).evaluate()
        model.train = True
        return ret


def main():
    archs = {
        'resnet_multilabel': resnet_finetune_custom_multilabel.Encoder,
        'pspnet_resnet_multilabel': pspnet_resnet_multilabel.Encoder
    }

    parser = argparse.ArgumentParser(
        description='Learning convnet from ILSVRC2012 dataset')
    parser.add_argument('--arch', '-a', choices=archs.keys(), default='resnet_finetune',
                        help='Convnet architecture')
    parser.add_argument('--class_num', '-c', type=int, default=None,
                        help='class')
    parser.add_argument('--batchsize', '-B', type=int, default=32,
                        help='Learning minibatch size')
    parser.add_argument('--epoch', '-E', type=int, default=100,
                        help='Number of epochs to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU')
    parser.add_argument('--initmodel',
                        help='Initialize the model from given file')
    parser.add_argument('--loaderjob', '-j', type=int,
                        help='Number of parallel data loading processes')
    parser.add_argument('--mean', '-m', default='mean.npy',
                        help='Mean file (computed by compute_mean.py)')
    parser.add_argument('--resume', '-r', default='',
                        help='Initialize the trainer from given file')
    parser.add_argument('--out', '-o', default='result',
                        help='Output directory')
    parser.add_argument('--optimizer', '-opt', default='msgd',
                        help='Output directory')
    parser.add_argument('--root', '-R', default='.',
                        help='Root directory path of image files')
    parser.add_argument('--val_batchsize', '-b', type=int, default=100,
                        help='Validation minibatch size')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--width', '-wid', type=int, default=256,
                        help='Learning minibatch size')
    parser.add_argument('--height', '-hei', type=int, default=384,
                        help='Learning minibatch size')
    parser.add_argument('--val_iter', '-val_iter', type=int, default=1000)
    parser.set_defaults(test=False)
    args = parser.parse_args()

    # Initialize the model to train
    model = archs[args.arch]()
    print("model:"+str(archs[args.arch]))
    if args.initmodel:
        print('Load model from', args.initmodel)
        chainer.serializers.load_npz(args.initmodel, model)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make the GPU current
        model.to_gpu()

    # Load the datasets and mean file
    mean = np.load(args.mean)
    train = PreprocessedDataset('train_288416',mean, crop_size_x = args.width, crop_size_y =args.height)
    val = PreprocessedDataset('test_288416',mean, crop_size_x = args.width, crop_size_y=args.height, random = False)
    
    train_iter = chainer.iterators.MultiprocessIterator(
        train, args.batchsize, n_processes=args.loaderjob)
    val_iter = chainer.iterators.MultiprocessIterator(
        val, args.val_batchsize, repeat=False, n_processes=args.loaderjob)
    # Set up an optimizer
    if args.optimizer == 'adagrad':
            optimizer = chainer.optimizers.AdaGrad()
            print(optimizer)
            print("lr:" + str(optimizer.lr))
    elif args.optimizer == 'sgd':
            optimizer = chainer.optimizers.SGD()
            print(optimizer)
            print("lr:" + str(optimizer.lr))
    elif args.optimizer == 'adam':
            optimizer = chainer.optimizers.Adam()
            print(optimizer)
            print("alpha:" + str(optimizer.alpha))
    elif args.optimizer == 'rmsprop':
            optimizer = chainer.optimizers.RMSprop()
            print(optimizer)
            print("lr:" + str(optimizer.lr))
    elif args.optimizer == 'adadelta':
            optimizer = chainer.optimizers.AdaDelta()
            print(optimizer)
    else:
            optimizer = chainer.optimizers.MomentumSGD(lr=0.01, momentum=0.9)
            print(optimizer)
            print("lr:" + str(optimizer.lr))
    
    optimizer.setup(model)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), args.out)

    log_interval = (10 if args.test else 100), 'iteration'
    val_interval = (100 if args.test else args.val_iter), 'iteration'

    trainer.extend(extensions.dump_graph('main/loss'))
    if not args.optimizer == "adam":
        trainer.extend(extensions.LinearShift("lr", (0.01,0.001), (10000,20000)))
    trainer.extend(TestModeEvaluator(val_iter, model, device=args.gpu),
                    trigger=val_interval)
    trainer.extend(extensions.snapshot_object(
                    model, 'model_iteration_{.updater.iteration}'), trigger=val_interval)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    #trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'main/loss', 'validation/main/loss',
        'main/accuracy', 'validation/main/accuracy','validation/main/recall','\
        validation/main/precision','validation/main/f_value'
        ,'lr'
    ]), trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))
    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()

if __name__ == '__main__':
    main()
