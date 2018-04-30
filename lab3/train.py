#!/usr/bin/env python2.7
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import time
import os
from six.moves import cPickle
from model import *
from dataloader import *
import argparse

try:
    import tensorflow as tf
except ImportError:
    print("Tensorflow not installed; No tensorboard logging.")
    tf = None

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()

class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        target = target[:, :input.size(1)]
        mask =  mask[:, :input.size(1)]
        input = to_contiguous(input).view(-1, input.size(2))
        target = to_contiguous(target).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1)
        output = - input.gather(1, target) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def clip_gradient(optimizer):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-0.1, 0.1)

def add_summary_value(writer, key, value, iteration):
    summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
    writer.add_summary(summary, iteration)

def train(opt):
    opt.use_att = True
    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length

    tf_summary_writer = tf and tf.summary.FileWriter(opt.checkpoint_path)

    infos = {}
    histories = {}

    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)

    val_result_history = histories.get('val_result_history', {})
    loss_history = histories.get('loss_history', {})
    lr_history = histories.get('lr_history', {})
    ss_prob_history = histories.get('ss_prob_history', {})

    loader.iterators = infos.get('iterators', loader.iterators)
    loader.split_ix = infos.get('split_ix', loader.split_ix)
    best_val_score = infos.get('best_val_score', None)

    model = TopDownModel(opt)
    model.cuda()

    update_lr_flag = True
    # Assure in training mode
    model.train()

    crit = LanguageModelCriterion()

    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=0)


    while True:
        if update_lr_flag:
            if epoch :
                frac = epoch // 3
                decay_factor = 0.8  ** frac
                opt.current_lr = opt.learning_rate * decay_factor
                set_lr(optimizer, opt.current_lr)
            else:
                opt.current_lr = opt.learning_rate
            if epoch :
                frac = epoch // 5
                opt.ss_prob = min(0.25, 0.05 * frac)
                model.ss_prob = opt.ss_prob
            update_lr_flag = False
                
        start = time.time()
        data = loader.get_batch('train')
        print('Read data:', time.time() - start)

        torch.cuda.synchronize()
        start = time.time()

        tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks']]
        tmp = [Variable(torch.from_numpy(_), requires_grad=False).cuda() for _ in tmp]
        fc_feats, att_feats, labels, masks = tmp
        
        optimizer.zero_grad()
        loss = crit(model(fc_feats, att_feats, labels), labels[:,1:], masks[:,1:])
        loss.backward()
        clip_gradient(optimizer)
        optimizer.step()
        train_loss = loss.data[0]
        torch.cuda.synchronize()
        end = time.time()
        print("iter {} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
            .format(iteration, epoch, train_loss, end - start))

        iteration += 1
        if data['bounds']['wrapped']:
            epoch += 1
            update_lr_flag = True

        if (iteration % 25 == 0):
            if tf is not None:
                add_summary_value(tf_summary_writer, 'train_loss', train_loss, iteration)
                add_summary_value(tf_summary_writer, 'learning_rate', opt.current_lr, iteration)
                add_summary_value(tf_summary_writer, 'scheduled_sampling_prob', model.ss_prob, iteration)
                tf_summary_writer.flush()

            loss_history[iteration] = train_loss
            lr_history[iteration] = opt.current_lr
            ss_prob_history[iteration] = model.ss_prob

        if (iteration % 6000 == 0):
            best_flag = False
            if True: # if true

                checkpoint_path = os.path.join(opt.checkpoint_path, 'model.pth')
                torch.save(model.state_dict(), checkpoint_path)
                print("model saved to {}".format(checkpoint_path))
                optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer.pth')
                torch.save(optimizer.state_dict(), optimizer_path)

                infos['iter'] = iteration
                infos['epoch'] = epoch
                infos['iterators'] = loader.iterators
                infos['split_ix'] = loader.split_ix
                infos['best_val_score'] = best_val_score
                infos['opt'] = opt
                infos['vocab'] = loader.get_vocab()

                histories['val_result_history'] = val_result_history
                histories['loss_history'] = loss_history
                histories['lr_history'] = lr_history
                histories['ss_prob_history'] = ss_prob_history
                with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'.pkl'), 'wb') as f:
                    cPickle.dump(infos, f)
                with open(os.path.join(opt.checkpoint_path, 'histories_'+opt.id+'.pkl'), 'wb') as f:
                    cPickle.dump(histories, f)

                if best_flag:
                    checkpoint_path = os.path.join(opt.checkpoint_path, 'model-best.pth')
                    torch.save(model.state_dict(), checkpoint_path)
                    print("model saved to {}".format(checkpoint_path))
                    with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'-best.pkl'), 'wb') as f:
                        cPickle.dump(infos, f)

        if epoch == 5:
            break

opts = argparse.ArgumentParser()
opts.add_argument('--id', type=str, default='st')
opts.add_argument('--input_json', type=str, default='data/cocotalk.json')
opts.add_argument('--input_fc_dir', type=str, default='data/cocotalk_fc')
opts.add_argument('--input_att_dir', type=str, default='data/cocotalk_att')
opts.add_argument('--input_label_h5', type=str, default='data/cocotalk_label.h5')
opts.add_argument('--batch_size', type=int, default=10)
opts.add_argument('--learning_rate', type=int, default=0.0001)
opts.add_argument('--checkpoint_path', type=str, default='log_st')
opts.add_argument('--val_images_use', type=int, default=5000)
opts.add_argument('--seq_per_img', type=int, default=5)
opt = opts.parse_args()

train(opt)
