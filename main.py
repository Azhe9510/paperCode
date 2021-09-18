# -*- coding: utf-8 -*-

import os
import time
import scipy
import scipy.special
import scipy.io
import random
import numpy as np
import tensorflow as tf
from collections import deque

reuse = tf.AUTO_REUSE
dtype = np.float32
flag_fig = True

fd = 10
Ts = 20e-3
L = 2
C = 16


max_p = 38.
p_n = -114.

OBSERVE = 100
EPISODE = 3500
memory_size = 35000
INITIAL_EPSILON = 0.2
FINAL_EPSILON = 0.0001


train_interval = 10
batch_size = 256
update_rate = 0.01

sigma2_ = 1e-3 * pow(10., p_n / 10.)
maxP = 1e-3 * pow(10., max_p / 10.)
power_num = 10
power_set = np.hstack([np.zeros((1), dtype=dtype), 1e-3 * pow(10., np.linspace(5., max_p, power_num - 1) / 10.)])
gamma = 0.1
learning_rate = 0.001

meanM = minM = maxM = 2
min_dis = 5e-3

max_dis = 1000e-3

weight_file = r'G:\Try-Multi-user\forReview\weight.mat'
reward_file = r'G:\Try-Multi-user\forReview\reward'
avgrate_file = r'G:\Try-Multi-user\forReview\avgrate'
replay_memory = deque(maxlen=memory_size)


c = 3 * L * (L + 1) + 1
K = maxM * c
# state_num = 3*C + 2    #  3*K - 1  3*C + 2
state_num = 2 * C + 1




def Generate_H_set():

    pho = np.float32(scipy.special.k0(2 * np.pi * fd * Ts))
    H_set = np.zeros([M, K, int(Ns)], dtype=dtype)
    H_set[:, :, 0] = np.kron(np.sqrt(0.5 * (np.random.randn(M, c) ** 2 + np.random.randn(M, c) ** 2)),
                             np.ones((1, maxM), dtype=np.int32))
    for i in range(1, int(Ns)):
        H_set[:, :, i] = H_set[:, :, i - 1] * pho + np.sqrt(
            (1. - pho ** 2) * 0.5 * (np.random.randn(M, K) ** 2 + np.random.randn(M, K) ** 2))
    path_loss = Generate_path_loss()
    H2_set = np.square(H_set) * np.tile(np.expand_dims(path_loss, axis=2), [1, 1, int(Ns)])
    return H2_set


def Generate_environment():
    path_matrix = M * np.ones((n_y + 2 * L, n_x + 2 * L, maxM), dtype=np.int32)
    for i in range(L, n_y + L):
        for j in range(L, n_x + L):
            for l in range(maxM):
                path_matrix[i, j, l] = ((i - L) * n_x + (j - L)) * maxM + l
    p_array = np.zeros((M, K), dtype=np.int32)
    for n in range(N):
        i = n // n_x
        j = n % n_x
        Jx = np.zeros((0), dtype=np.int32)
        Jy = np.zeros((0), dtype=np.int32)
        for u in range(i - L, i + L + 1):
            v = 2 * L + 1 - np.abs(u - i)
            jx = j - (v - i % 2) // 2 + np.linspace(0, v - 1, num=v, dtype=np.int32) + L
            jy = np.ones((v), dtype=np.int32) * u + L
            Jx = np.hstack((Jx, jx))
            Jy = np.hstack((Jy, jy))
        for l in range(maxM):
            for k in range(c):
                for u in range(maxM):
                    p_array[n * maxM + l, k * maxM + u] = path_matrix[Jy[k], Jx[k], u]
    p_main = p_array[:, (c - 1) // 2 * maxM:(c + 1) // 2 * maxM]
    for n in range(N):
        for l in range(maxM):
            temp = p_main[n * maxM + l, l]
            p_main[n * maxM + l, l] = p_main[n * maxM + l, 0]
            p_main[n * maxM + l, 0] = temp
    p_inter = np.hstack([p_array[:, :(c - 1) // 2 * maxM], p_array[:, (c + 1) // 2 * maxM:]])
    p_array = np.hstack([p_main, p_inter])

    user = np.maximum(np.minimum(np.random.poisson(meanM, (N)), maxM), minM)
    user_list = np.zeros((N, maxM), dtype=np.int32)
    for i in range(N):
        user_list[i, :user[i]] = 1
    for k in range(N):
        for i in range(maxM):
            if user_list[k, i] == 0.:
                p_array = np.where(p_array == k * maxM + i, M, p_array)
    p_list = list()
    for i in range(M):
        p_list_temp = list()
        for j in range(K):
            p_list_temp.append([p_array[i, j]])
        p_list.append(p_list_temp)
    return p_array, p_list, user_list


def Generate_path_loss():
    slope = 0.
    p_tx = np.zeros((n_y, n_x))
    p_ty = np.zeros((n_y, n_x))
    p_rx = np.zeros((n_y, n_x, maxM))
    p_ry = np.zeros((n_y, n_x, maxM))
    dis_rx = np.random.uniform(min_dis, max_dis, size=(n_y, n_x, maxM))
    phi_rx = np.random.uniform(-np.pi, np.pi, size=(n_y, n_x, maxM))
    for i in range(n_y):
        for j in range(n_x):
            p_tx[i, j] = 2 * max_dis * j + (i % 2) * max_dis
            p_ty[i, j] = np.sqrt(3.) * max_dis * i
            for k in range(maxM):
                p_rx[i, j, k] = p_tx[i, j] + dis_rx[i, j, k] * np.cos(phi_rx[i, j, k])
                p_ry[i, j, k] = p_ty[i, j] + dis_rx[i, j, k] * np.sin(phi_rx[i, j, k])
    dis = 1e10 * np.ones((M, K), dtype=dtype)
    lognormal = np.zeros((M, K), dtype=dtype)
    for k in range(N):
        for l in range(maxM):
            for i in range(c):
                for j in range(maxM):
                    if p_array[k * maxM + l, i * maxM + j] < M:
                        bs = p_array[k * maxM + l, i * maxM + j] // maxM
                        dx2 = np.square((p_rx[k // n_x][k % n_x][l] - p_tx[bs // n_x][bs % n_x]))
                        dy2 = np.square((p_ry[k // n_x][k % n_x][l] - p_ty[bs // n_x][bs % n_x]))
                        distance = np.sqrt(dx2 + dy2)
                        dis[k * maxM + l, i * maxM + j] = distance
                        std = 8. + slope * (distance - min_dis)
                        lognormal[k * maxM + l, i * maxM + j] = np.random.lognormal(sigma=std)
    path_loss = lognormal * pow(10., -(120.9 + 37.6 * np.log10(dis)) / 10.)
    return path_loss


def Calculate_rate():

    maxC = 1000.
    P_extend = tf.concat([P, tf.zeros((1), dtype=dtype)], axis=0)
    P_matrix = tf.gather_nd(P_extend, p_list)
    path_main = tf.multiply(H2[:, 0], P_matrix[:, 0])
    path_inter = tf.reduce_sum(tf.multiply(H2[:, 1:], P_matrix[:, 1:]), axis=1)
    sinr = path_main / (path_inter + sigma2)
    sinr = tf.minimum(sinr, maxC)
    rate = W * tf.log(1. + sinr) / np.log(2)
    rate_extend = tf.concat([rate, tf.zeros((1), dtype=dtype)], axis=0)
    rate_matrix = tf.gather_nd(rate_extend, p_list)
    sinr_norm_inv = H2[:, 1:] / tf.tile(H2[:, 0:1], [1, K - 1])
    sinr_norm_inv = tf.log(1. + sinr_norm_inv) / np.log(2)
    avg_rate = tf.reduce_mean(rate)
    reward = tf.reduce_sum(rate)
    return rate_matrix, sinr_norm_inv, P_matrix, reward, avg_rate


def Generate_state(p_last, sinr_norm_inv):

    indices1 = np.tile(np.expand_dims(np.linspace(0, M - 1, num=M, dtype=np.int32), axis=1), [1, C])
    indices2 = np.argsort(sinr_norm_inv, axis=1)[:, -C:]
    p_last_ = np.hstack([p_last[:, 0:1], p_last[indices1, indices2 + 1]])
    sinr_norm_inv_ = sinr_norm_inv[indices1, indices2]
    s_t = np.hstack([p_last_, sinr_norm_inv_])
    return s_t


def Variable(shape):
    w = tf.get_variable('w', shape=shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    b = tf.get_variable('b', shape=[shape[-1]], initializer=tf.constant_initializer(0.01))
    return w, b


def Find_params(para_name):
    sets = []
    for var in tf.trainable_variables():
        if not var.name.find(para_name):
            sets.append(var)
    return sets


def Network(s, a, name):
    with tf.variable_scope(name + '.0', reuse=reuse):
        w, b = Variable([state_num, 128])
        l = tf.nn.relu(tf.matmul(s, w) + b)
    with tf.variable_scope(name + '.1', reuse=reuse):
        w, b = Variable([128, 64])
        l = tf.nn.relu(tf.matmul(l, w) + b)
    with tf.variable_scope(name + '.2', reuse=reuse):
        w, b = Variable([64, power_num])
        q_hat = tf.matmul(l, w) + b
    r = tf.reduce_sum(tf.multiply(q_hat, a), reduction_indices=1)
    a_hat = tf.argmax(q_hat, 1)
    list_var = Find_params(name)
    return q_hat, a_hat, r, list_var


def Loss(y, r):
    cost = tf.reduce_mean(tf.square((y - r)))
    return cost


def Optimizer(cost, var_list):
    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(learning_rate, global_step=global_step,
                                    decay_steps=EPISODE, decay_rate=0.1)


    add_global = global_step.assign_add(1)
    with tf.variable_scope('opt', reuse=reuse):
        train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost, var_list=var_list)
    return train_op, add_global


def Save_store(s_t, a_t, r_t, s_next):
    r_t = np.tile(r_t, (M))
    p_t = np.zeros((M, power_num), dtype=dtype)
    p_t[range(M), a_t] = 1.
    for i in range(M):
        replay_memory.append((s_t[i], p_t[i], r_t[i], s_next[i]))


def Sample():
    minibatch = random.sample(replay_memory, batch_size)
    batch_s = [d[0] for d in minibatch]
    batch_a = [d[1] for d in minibatch]
    batch_r = [d[2] for d in minibatch]
    batch_s_next = [d[3] for d in minibatch]
    return batch_s, batch_a, batch_r, batch_s_next


def Select_action(sess, s_t, episode):
    if episode > OBSERVE:
        epsilon = INITIAL_EPSILON - (episode - OBSERVE) * (INITIAL_EPSILON - FINAL_EPSILON) / (EPISODE - OBSERVE)
    elif episode <= OBSERVE:
        epsilon = INITIAL_EPSILON
    else:
        epsilon = 0.
    q_hat_ = sess.run(q_main, feed_dict={s: s_t})
    best_action = np.argmax(q_hat_, axis=1)
    random_index = np.array(np.random.uniform(size=(M)) < epsilon, dtype=np.int32)
    random_action = np.random.randint(0, high=power_num, size=(M))
    action_set = np.vstack([best_action, random_action])
    power_index = action_set[random_index, range(M)]
    power = power_set[power_index]  # W
    return power, power_index


def Step(p_t, H2_t):

    sinr_norm_, p_last, reward_, avg_rate_ = sess.run([sinr_norm_inv, P_matrix, reward, avg_rate],
                                                      feed_dict={P: p_t, H2: H2_t, W: W_, sigma2: sigma2_})
    s_next = Generate_state(p_last, sinr_norm_)
    return s_next, reward_, avg_rate_


def Experience_replay(sess):
    batch_s, batch_a, batch_r, batch_s_next = Sample()
    a_main_, q_main_ = sess.run([a_main, q_main], feed_dict={s: batch_s_next})
    q_double = q_main_[range(batch_size), a_main_]
    y_ = batch_r + gamma * q_double
    # a_targ_, q_targ_ = sess.run([a_targ, q_targ], feed_dict={s: batch_s_next})
    # q_target_ = q_targ_[range(batch_size), a_targ_]
    # y_ = batch_r + gamma*q_target_
    sess.run(train_main, feed_dict={s: batch_s, a: batch_a, y: y_})
    sess.run(update)


def Network_update(list_main, list_targ):
    update = []
    for i in range(len(list_targ)):
        value = list_main[i].value() * update_rate + (1. - update_rate) * list_targ[i].value()
        update.append(tf.assign(tf.get_default_graph().get_tensor_by_name(list_targ[i].name), value))
    return update


def Initial_para():
    H2_set = Generate_H_set()
    s_next, _, _ = Step(np.zeros([M], dtype=dtype), H2_set[:, :, 0])
    return H2_set, s_next


def Train_episode(sess, episode):

    reward_dqn_list = list()
    avg_rate_list = list()
    H2_set, s_t = Initial_para()
    for step_index in range(int(Ns)):
        p_t, a_t = Select_action(sess, s_t, episode)
        s_next, r_, rate_ = Step(p_t, H2_set[:, :, step_index])
        Save_store(s_t, a_t, r_, s_next)
        if episode > OBSERVE:
            if step_index % train_interval == 0:
                Experience_replay(sess)
        s_t = s_next
        reward_dqn_list.append(r_)
        avg_rate_list.append(rate_)
    if episode > OBSERVE:
        sess.run(add_global)
    reward_mean = sum(reward_dqn_list) / (Ns)
    avg_rate_mean = sum(avg_rate_list) / (Ns)

    return reward_mean, avg_rate_mean


def Test_episode(sess, episode):
    reward_dqn_list = list()
    avg_rate_list = list()
    H2_set, s_t = Initial_para()
    for step_index in range(int(Ns)):
        q_hat_ = sess.run(q_main, feed_dict={s: s_t})
        p_t = power_set[np.argmax(q_hat_, axis=1)]  # W
        s_next, r_, rate_ = Step(p_t, H2_set[:, :, step_index])
        s_t = s_next
        reward_dqn_list.append(r_)
        avg_rate_list.append(rate_)
    reward_mean = sum(reward_dqn_list) / (Ns)
    avg_rate_mean = sum(avg_rate_list) / (Ns)

    return reward_mean, avg_rate_mean


def Train(sess):

    st = time.time()
    reward_hist = list()
    avg_rate_hist = list()
    for k in range(1, EPISODE + 1):
        reward_mean, avg_rate_mean = Train_episode(sess, k)
        reward_hist.append(reward_mean)
        avg_rate_hist.append(avg_rate_mean)
        if k % 100 == 0:
            print("Episode(train):%d   reward_mean: %.3f  avg_rate_mean: %.3f  Time cost: %.2fs"
                  % (k, np.mean(reward_hist[-100:]), np.mean(avg_rate_hist[-100:]), time.time() - st))
            st = time.time()
    Save(weight_file)
    return reward_hist, avg_rate_hist



def Test(sess):

    sess.run(load)
    reward_hist = list()
    avg_rate_hist = list()
    for k in range(1, TEST_EPISODE + 1):
        reward_mean, avg_rate_mean = Test_episode(sess, k)
        reward_hist.append(np.mean(reward_mean))
        avg_rate_hist.append(np.mean(avg_rate_mean))
        print(k)
    print("Test(Rmax_1000): %.3f " %(np.mean(avg_rate_hist)))
    return reward_hist, avg_rate_hist


def Save(weight_file):
    dict_name = {}
    for var in list_main:
        dict_name[var.name] = var.eval()
    scipy.io.savemat(weight_file, dict_name)


def Network_ini(theta):
    update = []
    for var in list_main:
        update.append(tf.assign(tf.get_default_graph().get_tensor_by_name(var.name),
                                tf.constant(np.reshape(theta[var.name], var.shape))))
    return update


if __name__ == "__main__":


    n_x = n_y = 5
    num = [5]

    N = n_x * n_y  # BS number
    M = N * maxM  # maximum users
    W_ = np.ones((M), dtype=dtype)  # [M]
    Ns = 5e1
    with tf.Graph().as_default():
        H2 = tf.placeholder(shape=[None, K], dtype=dtype)
        P = tf.placeholder(shape=[None], dtype=dtype)
        W = tf.placeholder(shape=[None], dtype=dtype)
        sigma2 = tf.placeholder(dtype=dtype)
        p_array, p_list, user_list = Generate_environment()
        rate_matrix, sinr_norm_inv, P_matrix, reward, avg_rate = Calculate_rate()

        s = tf.placeholder(shape=[None, state_num], dtype=dtype)
        a = tf.placeholder(shape=[None, power_num], dtype=dtype)
        y = tf.placeholder(shape=[None], dtype=dtype)
        q_targ, a_targ, _, list_targ = Network(s, a, 'targ')
        q_main, a_main, r, list_main = Network(s, a, 'main')
        cost = Loss(y, r)
        train_main, add_global = Optimizer(cost, list_main)
        update = Network_update(list_main, list_targ)

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            reward_hist, avg_rate_hist = Train(sess)
            scipy.io.savemat(reward_file, {'reward_hist': reward_hist})
            scipy.io.savemat(avgrate_file, {'avg_rate_hist': avg_rate_hist})

        weight = [ r'G:\Try-Multi-user\forReview\weight.mat']


        reward_hist = list()
        avg_rate_hist = list()




        for n in num:
            n_x = n
            n_y = n
            N = n_x * n_y
            M = N * maxM
            W_ = np.ones((M), dtype=dtype)

            Ns = 1e3
            TEST_EPISODE = 100
            Ns = 4e1
            TEST_EPISODE = 500

            with tf.Graph().as_default():
                H2 = tf.placeholder(shape=[None, K], dtype=dtype)
                P = tf.placeholder(shape=[None], dtype=dtype)
                W = tf.placeholder(shape=[None], dtype=dtype)
                sigma2 = tf.placeholder(dtype=dtype)
                p_array, p_list, user_list = Generate_environment()
                rate_matrix, sinr_norm_inv, P_matrix, reward, avg_rate = Calculate_rate()

                s = tf.placeholder(shape=[None, state_num], dtype=dtype)
                a = tf.placeholder(shape=[None, power_num], dtype=dtype)
                q_main, a_main, r, list_main = Network(s, a, 'main')

                reward_hist_temp = list()
                avg_rate_hist_temp = list()

                for k in range(1):
                    load = Network_ini(scipy.io.loadmat(weight[k]))
                    with tf.Session() as sess:
                        tf.global_variables_initializer().run()
                        aaa, bbb = Test(sess)
                        aa = np.mean(aaa)
                        bb = np.mean(bbb)
                        reward_hist_temp.append(aa)
                        avg_rate_hist_temp.append(bb)
                reward_hist.append(reward_hist_temp)
                avg_rate_hist.append(avg_rate_hist_temp)
