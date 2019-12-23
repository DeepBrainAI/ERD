import sys
import random
import torch
import importlib
from tensorboardX import SummaryWriter
import torch.nn.utils.rnn as rnn_utils
import pickle
import tqdm
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import deque
import json
import pdb
from dataUtils_CN import *
import numpy as np

def TrainCMModel_V0(sent_pooler, rdm_model, rdm_classifier, cm_model, stage, t_rw, t_steps, log_dir, logger, FLAGS, cuda=True):
    batch_size = FLAGS.batch_size
    t_acc = 0.9
    ids = np.array(range(batch_size), dtype=np.int32)
    seq_states = np.zeros([batch_size], dtype=np.int32)
    isStop = torch.zeros([batch_size], dtype=torch.int32)
    max_id = batch_size
    df_init_states = torch.zeros([1, batch_size, rdm_model.hidden_dim], dtype=torch.float32).cuda()
    writer = SummaryWriter(log_dir, filename_suffix="_ERD_CM_stage_%3d"%stage)
    D = deque()
    ssq = []
    print("in RL the begining")
    rl_optim = torch.optim.Adam([{'params': sent_pooler.parameters(), 'lr': 2e-5},
                                 {'params': rdm_model.parameters(), 'lr': 2e-5},
                                 {'params':cm_model.parameters(), 'lr':1e-3}])
    data_ID = get_data_ID()
    valid_data_len = get_valid_data_len()
    data_len = get_data_len()
    
    if len(data_ID) % batch_size == 0: # the total number of events
        flags = int(len(data_ID) / FLAGS.batch_size)
    else:
        flags = int(len(data_ID) / FLAGS.batch_size) + 1

    for i in range(flags):
        with torch.no_grad():
            x, x_len, y = get_df_batch(i, batch_size)
            seq = sent_pooler(x)
            rdm_hiddens = rdm_model(seq)
            batchsize, _, _ = rdm_hiddens.shape
            print("batch %d"%i)
            if len(ssq) > 0:
                ssq.extend([rdm_classifier(h) for h in rdm_hiddens])
            else:
                ssq = [rdm_classifier(h) for h in rdm_hiddens]
            torch.cuda.empty_cache()

    print(get_curtime() + " Now Start RL training ...")
    counter = 0
    sum_rw = 0.0 # sum of rewards
    
    while True:
    #         if counter > FLAGS.OBSERVE:
        if counter > FLAGS.OBSERVE:
            sum_rw += rw.mean()
            if counter % 200 == 0:
                sum_rw = sum_rw / 2000
                print(get_curtime() + " Step: " + str(counter-FLAGS.OBSERVE) + " REWARD IS " + str(sum_rw))
                if counter > t_steps:
                    print("Retch The Target Steps")
                    break
                sum_rw = 0.0
            s_state, s_x, s_isStop, s_rw = get_RL_Train_batch(D)
            word_tensors = torch.tensor(s_x)
            batchsize, max_sent_len, emb_dim = word_tensors.shape
            sent_tensor = sent_pooler.linear(word_tensors.reshape([-1, emb_dim]).cuda()).reshape([batchsize, max_sent_len, emb_dim]).max(axis=1)[0].unsqueeze(1)
            df_outs, df_last_state = rdm_model.gru_model(sent_tensor, s_state.unsqueeze(0).cuda())
            batchsize, _, hidden_dim = df_outs.shape
            stopScore, isStop = cm_model(df_outs.reshape([-1, hidden_dim]))
            out_action = (stopScore*s_isStop.cuda()).sum(axis=1)
            rl_cost = torch.pow(s_rw.cuda() - out_action, 2).mean()
            rl_optim.zero_grad()
            rl_cost.backward()
            torch.cuda.empty_cache()
            rl_optim.step()
            # print("RL Cost:", rl_cost)
            writer.add_scalar('RL Cost', rl_cost, counter - FLAGS.OBSERVE)
            if (counter - FLAGS.OBSERVE)%100 == 0:
                print("*** %6d|%6d *** RL Cost:%8.6f"%(counter, t_steps, rl_cost))
                valid_new_len = get_new_len_on_valid_data(sent_pooler, rdm_model, cm_model, FLAGS, cuda=True)
                print("diff len:", np.array(valid_data_len)-np.array(valid_new_len))

        x, y, ids, seq_states, max_id = get_rl_batch_0(ids, seq_states, isStop, max_id, 0)
        for j in range(FLAGS.batch_size):
            if seq_states[j] == 1:
                df_init_states[0][j].fill_(0.0)
                
        with torch.no_grad():
            word_tensors = torch.tensor(x)
            batchsize, max_sent_len, emb_dim = word_tensors.shape
            sent_tensor = sent_pooler.linear(word_tensors.reshape([-1, emb_dim]).cuda()).reshape([batchsize, max_sent_len, emb_dim]).max(axis=1)[0].unsqueeze(1)
            df_outs, df_last_state = rdm_model.gru_model(sent_tensor, df_init_states)
            batchsize, _, hidden_dim = df_outs.shape
            stopScore, isStop = cm_model(df_outs.reshape([-1, hidden_dim]))
            
        for j in range(batch_size):
            if random.random() < FLAGS.random_rate:
                isStop[j] = torch.randn(2).argmax()
            if seq_states[j] == data_len[ids[j]]:
                isStop[j] = 1
        rw, Q_val = get_reward_0(isStop, stopScore, ssq, ids, seq_states)
        for j in range(FLAGS.batch_size):
            D.append((df_init_states[0][j], x[j], isStop[j], rw[j]))
            if len(D) > FLAGS.max_memory:
                D.popleft()
        df_init_states = df_last_state
        counter += 1