#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import os
import time
import datetime
import numpy as np
import gensim
import random
import math
import re
import pickle
import torch


# In[26]:


import pkuseg


# In[45]:


from sklearn.metrics import accuracy_score


# In[2]:


files = []
data = {}
data_ID = []
data_len = []
data_y = []

valid_data_ID = []
valid_data_y = []
valid_data_len = []


# In[27]:


with open("./word2vec_CN.pkl", "rb") as handle:
    word2vec = pickle.load(handle)
print("load word2vec finished")

# In[35]:


reward_counter = 0
eval_flag = 0

seg = pkuseg.pkuseg()


# In[37]:


def get_data_ID():
    global data_ID
    return data_ID

def get_data_len():
    global data_len
    return data_len

def get_curtime():
    return time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))


def list_files(data_path):
    global data, files
    fs = os.listdir(data_path)
    for f1 in fs:
        tmp_path = os.path.join(data_path, f1)
        if not os.path.isdir(tmp_path):
            if tmp_path.split('.')[-1] == 'json':
                files.append(tmp_path)
        else:
            list_files(tmp_path)


def str2timestamp(str_time):
    month = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
             'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
             'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'}
    ss = str_time.split(' ')
    m_time = ss[5] + "-" + month[ss[1]] + '-' + ss[2] + ' ' + ss[3]
    d = datetime.datetime.strptime(m_time, "%Y-%m-%d %H:%M:%S")
    t = d.timetuple()
    timeStamp = int(time.mktime(t))
    return timeStamp


def data_process(file_path):
    ret = {}
    ss = file_path.split("/")
    data = json.load(open(file_path, mode="r", encoding="utf-8"))
    # 'Wed Jan 07 11:14:08 +0000 2015'
    # print("SS:", ss)
    ret[ss[6]] = {'label': ss[5], 'text': [data['text'].lower()], 'created_at': [str2timestamp(data['created_at'])]}
    return ret

def transIrregularWord(line):
    if not line:
        return ''
    line.lower()
    line = re.sub("@[^ ]*", "{ 提到某人 }", line)
    line = re.sub("#[^ ]*", "{ 某个话题 }", line)
    line = re.sub("http(.?)://[^ ]*", "{ 网页链接 }", line)
    return seg.cut(line) 


def load_test_data_fast():
    global data, data_ID, data_len, data_y, eval_flag
    with open("data/weibo_dict.txt", "rb") as handle:
        data = pickle.load(handle)
    data_ID = np.load("data/test_weibo_ID.npy").tolist()
    data_len = np.load("data/test_weibo_len.npy").tolist()
    data_y = np.load("data/test_weibo_y.npy").tolist()
    max_sent = max( map(lambda value: max(map(lambda txt_list: len(txt_list), value['text']) ), list(data.values()) ) )
    print("max_sent:", max_sent, ",  max_seq_len:", max(data_len))
    eval_flag = int(len(data_ID) / 4) * 3
    print("{} data loaded".format(len(data))) 
    
def load_data_fast():
    global data, data_ID, data_len, data_y, valid_data_ID, valid_data_y, valid_data_len
    with open("data/weibo_dict.txt", "rb") as handle:
        data = pickle.load(handle)
    data_ID = np.load("data/weibo_ID.npy").tolist()
    data_len = np.load("data/weibo_len.npy").tolist()
    data_y = np.load("data/weibo_y.npy").tolist()
    valid_data_ID = np.load("data/test_weibo_ID.npy").tolist()
    valid_data_len = np.load("data/test_weibo_len.npy").tolist()
    valid_data_y = np.load("data/test_weibo_y.npy").tolist()
    max_sent = max( map(lambda value: max(map(lambda txt_list: len(txt_list), value['text']) ), list(data.values()) ) )
    print("max_sent:", max_sent, ",  max_seq_len:", max(data_len))
    eval_flag = int(len(data_ID) / 4) * 3
    print("{} data loaded".format(len(data)))    


def sortTempList(temp_list):
    time = np.array([item[0] for item in temp_list])
    posts = np.array([item[1] for item in temp_list])
    idxs = time.argsort().tolist()
    rst = [[t, p] for (t, p) in zip(time[idxs], posts[idxs])]
    del time, posts
    return rst


# ### 重写load_data

# In[38]:


def load_data(weibo_file, weibo_dir):
    global data, files, data_ID, data_len, eval_flag, data_y
    data = {}
    files = []
    data_ID = []
    data_len = []
    
    with open(weibo_file) as fr:
        for line in fr:
            s = line.split('\t')
            eid = s[0].strip("eid:")
            label = int(s[1].strip('label:'))
            data_ID.append(eid)
            if label == 1:
                data_y.append([0, 1])
            elif label == 0:
                data_y.append([1, 0])

    for eid in data_ID:
        data_file = os.path.join(weibo_dir, "%s.json"%eid)
        data[eid] = {}
        with open(data_file) as fr:
            event_data = json.load(fr)
            texts = [tweet['original_text'] for tweet in event_data]
            created_at = [tweet['t'] for tweet in event_data]
            idxs = np.array(created_at).argsort().tolist()
            data[eid]['text'] = [transIrregularWord(texts[idx]) for idx in idxs]
            data[eid]['created_at'] = [created_at[idx] for idx in idxs]
            data_len.append(len(texts))


# In[6]:


def get_df_batch(start, batch_size, new_data_len=[], cuda=True):
    data_x = []
    m_data_y = np.zeros([batch_size, 2], dtype=np.int32)
    m_data_len = np.zeros([batch_size], dtype=np.int32)
    miss_vec = 0
    hit_vec = 0
    if len(new_data_len) > 0:
        t_data_len = new_data_len
    else:
        t_data_len = data_len
    mts = start * batch_size
    if mts >= len(data_ID):
        mts = mts % len(data_ID)

    for i in range(batch_size):
        m_data_y[i] = data_y[mts]
        m_data_len[i] = t_data_len[mts]
        seq = []
        for j in range(t_data_len[mts]):
            sent = []
            t_words = data[data_ID[mts]]['text'][j]
            if len(t_words) == 0:
                print("ID:%s   j:%3d    empty sentence:"%(valid_data_ID[mts], j), t_words)
                continue

            for k in range(len(t_words)):
                m_word = t_words[k]
                try:
                    sent.append( torch.tensor([word2vec[m_word]], dtype=torch.float32) )
                except KeyError:
                    miss_vec += 1
                    sent.append( torch.tensor([word2vec['{'] +word2vec['未知'] +  word2vec['词'] + word2vec['}'] ], dtype=torch.float32) )
                except IndexError:
                    raise
                else:
                    hit_vec += 1
            sent_tensor = torch.cat(sent)
            seq.append(sent_tensor)
        data_x.append(seq)
        mts += 1
        if mts >= len(data_ID): # read data looply
            mts = mts % len(data_ID)
            
    return data_x, m_data_len, m_data_y


# ``` python
# with open('./data/weibo_dict.txt', 'wb') as handle:
#     pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
# 
# idxs = np.random.randn(len(data_ID)).argsort()
# 
# data_ID = [data_ID[idx] for idx in idxs]
# data_y = [data_y[idx] for idx in idxs]
# data_len = [data_len[idx] for idx in idxs]
# 
# np.save("./data/weibo_ID.npy", np.array(data_ID)[:4000])
# np.save("./data/weibo_y.npy", np.array(data_y)[:4000])
# np.save("./data/weibo_len.npy", np.array(data_len)[:4000])
# 
# np.save("./data/test_weibo_ID.npy", np.array(data_ID)[4000:])
# np.save("./data/test_weibo_y.npy", np.array(data_y)[4000:])
# np.save("./data/test_weibo_len.npy", np.array(data_len)[4000:])
# ```

# In[44]:


def get_rl_batch(ids, seq_states, stop_states, counter_id, start_id, total_data):
    input_x = np.zeros([FLAGS.batch_size, FLAGS.max_sent_len, FLAGS.embedding_dim], dtype=np.float32)
    input_y = np.zeros([FLAGS.batch_size, FLAGS.class_num], dtype=np.float32)
    miss_vec = 0
    total_data = len(data_len)
    for i in range(FLAGS.batch_size):
        # seq_states:records the id of a sentence in a sequence
        # stop_states: records whether the sentence is judged by the program
        if stop_states[i] == 1 or seq_states[i] >= data_len[ids[i]]: 
            ids[i] = counter_id + start_id
            seq_states[i] = 0
            try:
                t_words = data[ data_ID[ids[i]] ]['text'][seq_states[i]]
            except:
                print(ids[i], seq_states[i])
            for j in range(len(t_words)):
                m_word = t_words[j]
                try:
                    input_x[i][j] = word2vec[m_word]
                except:
                    miss_vec = 1
            input_y[i] = data_y[ids[i]]
            counter_id += 1
            counter_id = counter_id % total_data
        else:
            try:
                t_words = data[ data_ID[ids[i]] ]['text'][seq_states[i]]
            except:
                print("ids and seq_states:", ids[i], seq_states[i])
                t_words = []
            for j in range(len(t_words)):
                m_word = t_words[j]
                try:
                    input_x[i][j] = word2vec[m_word]
                except:
                    miss_vec += 1
            input_y[i] = data_y[ids[i]]
        # point to the next sequence
        seq_states[i] += 1

    return input_x, input_y, ids, seq_states, counter_id


# In[46]:


def accuracy_on_valid_data(rdm_model = None, sent_pooler = None, rdm_classifier=None, new_data_len=[], cuda=True):
    batch_size = 20
    t_steps = int(len(valid_data_ID)/batch_size)
    sum_acc = 0.0
    miss_vec = 0
    mts = 0
    hit_vec = 0
    if len(new_data_len) > 0:
        t_data_len = new_data_len
    else:
        t_data_len = valid_data_len
    
    for step in range(t_steps):
        data_x = []
        m_data_y = np.zeros([batch_size, 2], dtype=np.int32)
        m_data_len = np.zeros([batch_size], dtype=np.int32)
        for i in range(batch_size):
            m_data_y[i] = valid_data_y[mts]
            m_data_len[i] = t_data_len[mts]
            seq = []
            for j in range(t_data_len[mts]):
                sent = []
                t_words = data[valid_data_ID[mts]]['text'][j]
                if len(t_words) == 0:
                    print("ID:%s   j:%3d    empty sentence:"%(valid_data_ID[mts], j), t_words)
                    continue        

                for k in range(len(t_words)):
                    m_word = t_words[k]
                    try:
                        sent.append( torch.tensor([word2vec[m_word]], dtype=torch.float32))
                    except KeyError:
                        miss_vec += 1
                        sent.append( torch.tensor([word2vec['{'] +word2vec['an'] +  word2vec['unknown'] + word2vec['word'] + word2vec['}'] ], dtype=torch.float32) )
                    except IndexError:
                        raise
                    else:
                        hit_vec += 1
                sent_tensor = torch.cat(sent)                        
                seq.append(sent_tensor)

            data_x.append(seq)
            mts += 1
            if mts >= len(valid_data_ID): # read data looply
                mts = mts % len(valid_data_ID)
        
        
        if rdm_model is not None and sent_pooler is not None and rdm_classifier is not None:
            with torch.no_grad():
                seq = sent_pooler(data_x)
                rdm_hiddens = rdm_model(seq)
                batchsize, _, _ = rdm_hiddens.shape
                rdm_outs = torch.cat(
                    [ rdm_hiddens[i][m_data_len[i]-1].unsqueeze(0) for i in range(batchsize)] 
                    # a list of tensor, where the ndim of tensor is 1 and the shape of tensor is [hidden_size]
                )
                rdm_scores = rdm_classifier(
                    rdm_outs
                )
                rdm_preds = rdm_scores.argmax(axis=1)
                y_label = torch.tensor(m_data_y).argmax(axis=1).cuda() if cuda else torch.tensor(m_data_y).argmax(axis=1)
                acc = accuracy_score(y_label.cpu().numpy(), rdm_preds.cpu().numpy())
            torch.cuda.empty_cache()

        sum_acc += acc
    mean_acc = sum_acc / (1.0*t_steps)
    return mean_acc


# In[47]:


def get_reward_0(isStop, ss, pys, ids, seq_ids):
    global reward_counter
    reward = torch.zeros([len(isStop)], dtype=torch.float32)
    Q_Val = torch.zeros([len(isStop)], dtype= torch.float32)
    for i in range(len(isStop)):
        if isStop[i] == 1:
            try:
                if pys[ids[i]][seq_ids[i]-1].argmax() == np.argmax(data_y[ids[i]]):
                    reward_counter += 1 # more number of correct prediction, more rewards
                    r = 1 + FLAGS.reward_rate * math.log(reward_counter)
                    reward[i] = r   
                else:
                    reward[i] = -100
            except:
                print("i:", i)
                print("ids_i:", ids[i])
                print("seq_ids:", seq_ids[i])
                print("pys:", pys[ids[i]])
                raise
            Q_Val[i] = reward[i]
        else:
            reward[i] = -0.01 
            Q_Val[i] = reward[i] + 0.99 * max(ss[i])
    return reward, Q_Val


# In[48]:


def get_reward(isStop, ss, pys, ids, seq_ids):
    global reward_counter
    reward = torch.zeros([len(isStop)], dtype=torch.float32)
    Q_Val = torch.zeros([len(isStop)], dtype= torch.float32)
    for i in range(len(isStop)):
        if isStop[i] == 1:
            if pys[ids[i]][seq_ids[i]-1].argmax() == np.argmax(data_y[ids[i]]):
                reward_counter += 1 # more number of correct prediction, more rewards
                r = 1 + min(FLAGS.reward_rate * math.log(reward_counter), 10)
                reward[i] = r   
            else:
                reward[i] = -100
            Q_Val[i] = reward[i]
        else:
            reward[i] = -0.01 
            Q_Val[i] = reward[i] + 0.99 * max(ss[i])
    return reward, Q_Val


# In[49]:


def get_reward_v1(isStop, mss, ssq, ids, seq_states, cm_model, rdm_hiddens_seq):
    global reward_counter
    reward = torch.zeros([len(isStop)], dtype=torch.float32)
    Q_Val = torch.zeros([len(isStop)], dtype= torch.float32)
    for i in range(len(isStop)):
        if isStop[i] == 1:
            if ssq[ids[i]][seq_states[i]-1].argmax() == np.argmax(data_y[ids[i]]):
                reward_counter += 1 # more number of correct prediction, more rewards
                r = 1 + min(FLAGS.reward_rate * math.log(reward_counter), 10)
                reward[i] = r   
                if data_len[ids[i]] > seq_states[i]:
                    with torch.no_grad():
                        subsequent_score = cm_model.Classifier(
                            nn.functional.relu(
                                cm_model.DenseLayer(
                                    rdm_hiddens_seq[ids[i]]
                                )
                            )
                        )               
                    torch.cuda.empty_cache()
                    for j in range(seq_states[i], data_len[ids[i]]):
                        if subsequent_score[j][0] > subsequent_score[j][1]:
                            reward[i] += -20
                            break
                        else:
                            reward[i] +=  15.0/data_len[ids[i]]
            else:
                reward[i] = -100
            Q_Val[i] = reward[i]
        else:
            reward[i] = -0.01 
            Q_Val[i] = reward[i] + 0.99 * max(mss[i])
    return reward, Q_Val


# In[51]:


def get_new_len(sent_pooler, rdm_model, cm_model, FLAGS, cuda):
    batch_size = 20
    new_len = []
    valid_new_len = []
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
            rdm_outs = torch.cat(
                [ rdm_hiddens[i][x_len[i]-1] for i in range(batchsize)] 
                # a list of tensor, where the ndim of tensor is 1 and the shape of tensor is [hidden_size]
            ).reshape(
                [-1, rdm_model.hidden_dim]
            )
            stopScores = cm_model.Classifier(
                    nn.functional.relu(
                        cm_model.DenseLayer(
                            rdm_hiddens.reshape([-1, rdm_model.hidden_dim])
                    )
                )
            ).reshape(
                [batchsize, -1, 2]
            )
            isStop = stopScores.argmax(axis=-1).cpu().numpy()

            tmp_len = [iS.argmax()+1 if (iS.max() ==1 and (iS.argmax()+1)<x_len[iS_idx]) else x_len[iS_idx] for iS_idx, iS in enumerate(isStop)]

            for t_idx in range(len(tmp_len)):
                try:
                    assert tmp_len[t_idx] <= x_len[t_idx]
                except:
                    print("i:", t_idx)
                    print("new_len:", tmp_len)
                    print("data_len:", x_len)
                    raise

            new_len.extend(tmp_len)

    batchsize = 20
    mts = 0
    hit_vec = 0
    miss_vec = 0
    t_steps = int(len(valid_data_ID)/batchsize)
    valid_new_len = []
    for step in range(t_steps):
        data_x = []
        m_data_y = np.zeros([batch_size, 2], dtype=np.int32)
        m_data_len = np.zeros([batch_size], dtype=np.int32)
        for i in range(batch_size):
            m_data_y[i] = valid_data_y[mts]
            m_data_len[i] = valid_data_len[mts]
            seq = []
            for j in range(valid_data_len[mts]):
                sent = []
                t_words = data[valid_data_ID[mts]]['text'][j]

                for k in range(len(t_words)):
                    m_word = t_words[k]
                    try:
                        sent.append( torch.tensor([word2vec[m_word]], dtype=torch.float32))
                    except KeyError:
                        miss_vec += 1
                        sent.append( torch.tensor([word2vec['{'] +word2vec['an'] +  word2vec['unknown'] + word2vec['word'] + word2vec['}'] ], dtype=torch.float32) )
                    except IndexError:
                        raise
                    else:
                        hit_vec += 1
                if len(sent) != 0 :
                    sent_tensor = torch.cat(sent)
                else:
                    print("empty sentence:", t_words)
                seq.append(sent_tensor)
            data_x.append(seq)
            mts += 1
            if mts >= len(data_ID): # read data looply
                mts = mts % len(data_ID)
        with torch.no_grad():
            seq = sent_pooler(data_x)
            rdm_hiddens = rdm_model(seq)
            batchsize, _, _ = rdm_hiddens.shape
            rdm_outs = torch.cat(
                [ rdm_hiddens[i][m_data_len[i]-1] for i in range(batchsize)] 
                # a list of tensor, where the ndim of tensor is 1 and the shape of tensor is [hidden_size]
            ).reshape(
                [-1, rdm_model.hidden_dim]
            )
            stopScores = cm_model.Classifier(
                    nn.functional.relu(
                        cm_model.DenseLayer(
                            rdm_hiddens.reshape([-1, rdm_model.hidden_dim])
                    )
                )
            ).reshape(
                [batchsize, -1, 2]
            )
            isStop = stopScores.argmax(axis=-1).cpu().numpy()

            tmp_len = [iS.argmax()+1 if (iS.max() ==1 and (iS.argmax()+1)<m_data_len[iS_idx]) else m_data_len[iS_idx] for iS_idx, iS in enumerate(isStop)]

            for t_idx in range(len(tmp_len)):
                try:
                    assert tmp_len[t_idx] <= m_data_len[t_idx]
                except:
                    print("i:", t_idx)
                    print("new_len:", tmp_len)
                    print("data_len:", x_len)
                    raise
        valid_new_len.extend(tmp_len)


    print("max_new_len:", max(new_len))
    print("mean_new_len:", sum(new_len)*1.0/len(new_len))
    return new_len[:len(data_len)], valid_new_len[:len(valid_data_len)]


# In[52]:


def get_RL_Train_batch_V1(D, FLAGS, batch_size, cuda=False):
    m_batch = random.sample(D, batch_size)
    rdm_state = torch.zeros([batch_size, FLAGS.hidden_dim], dtype=torch.float32)
    s_ids = []
    s_seqStates = []
    for i in range(batch_size):
        rdm_state[i] = m_batch[i][0]
        s_ids.append(m_batch[i][1])
        s_seqStates.append(m_batch[i][2])
    if cuda:
        return rdm_state.cuda(), s_ids, s_seqStates
    else:
        return rdm_state, s_ids, s_seqStates

