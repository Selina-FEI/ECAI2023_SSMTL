import copy
import math
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import sklearn.datasets, sklearn.preprocessing, sklearn.model_selection
from sklearn.decomposition import PCA
from scipy import sparse
import random
from random import sample
import scipy.io as io
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


# load data
data = io.loadmat('01_rf1_1000.mat')
m = 8
p = 64
ns = []
X_train_list, X_test_list, Y_train_list, Y_test_list, X_val_list, Y_val_list = [], [], [], [], [], []
for i in range(m):
    X_train = data['tr_X'][i]
    Y_train = data['tr_Y'][i].reshape((-1, 1))
    X_test = data['te_X'][i]
    Y_test = data['te_Y'][i].reshape((-1, 1))
    X_train_list.append(X_train)
    X_test_list.append(X_test)
    Y_train_list.append(Y_train)
    Y_test_list.append(Y_test)
    ns.append(X_train.shape[0])


# reg para
h = 2
Gs = []
for i in range(h):
    # Gs = (item1, item2, item3), where each item = 1 or None
    # item1: dg1 trivial, dg2 non-trivial
    # item2: dg1 non-trivial, dg2 trivial
    # item3: dg1, dg2 non-trivial
    Gs.append((1, 1, 1))
gama1 = 1e-3
gama2 = 0.1
phi = 10
gama = (gama1, gama1, gama2, gama2, phi)

lr = 5e-5
theta = 1
epoches = 2000
miu = 1e-4
theta = 1



def sigmoid(a):
    y = a.copy()
    y[a >= 0] = 1.0 / (1 + np.exp(-a[a >= 0]))
    y[a < 0] = np.exp(a[a < 0]) / (1 + np.exp(a[a < 0]))
    return y


def projection(para, level):
    if sparse.linalg.norm(para) <= 1:
        res = para
    else:
        res = para / sparse.linalg.norm(para)
    return res

# generate dgs (linear operator)
def gen_Dg(type, G_g, level, gama, phi, row=False, col=False):
    ori_Dg = None
    if row and not col:
        ori_Dg = np.zeros(p)
    elif col and not row:
        ori_Dg = np.zeros(m)
    elif col and row:
        ori_Dg = [np.zeros(m), np.zeros(p)]

    if not row or not col:
        if type == 'feature_learn':
            ori_Dg[G_g] = 1
        elif type == 'task_cluster':
            i, j = G_g
            ori_Dg[i] = 1
            ori_Dg[j] = -1
    else:
        if type == 'feature_learn':
            i, j = G_g
            ori_Dg[0][i] = 1
            ori_Dg[1][j] = 1
        elif type == 'task_cluster':
            idx1, idx2 = G_g
            ori_Dg[0][idx1[0]] = 1
            ori_Dg[0][idx1[1]] = -1
            ori_Dg[1][idx2] = 1

    res = None
    if col and not row:
        res = sparse.kron(sparse.csr_matrix(ori_Dg), sparse.csr_matrix(np.identity(p))) * gama
        if level != 0:
            res *= (phi ** level)
    elif row and not col:
        res = sparse.kron(sparse.csr_matrix(np.identity(m)), sparse.csr_matrix(ori_Dg)) * gama
        if level != 0:
            res *= (phi ** level)
    elif row and col:
        res = sparse.kron(sparse.csr_matrix(ori_Dg[0]), sparse.csr_matrix(ori_Dg[1])) * gama
        if level != 0:
            res /= (phi ** level)
    return res


def loss_fuc(para_y, para_X_hat, para_b, para_w, para_w_list, para_D, gama, phi):
    '''g = para_gama * para_alpha.T @ para_D @ para_w - para_miu * 1/2 * (np.linalg.norm(para_alpha) ** 2)
    loss = 1/2 * (np.linalg.norm(para_y - para_X_hat @ para_w) ** 2) + g'''

    # classification
    # pred = sigmoid((para_X_hat @ para_w + para_b).toarray())
    # loss = - (para_y.toarray().T @ np.log(pred + 1e-10) + (1 - para_y.toarray()).T @ np.log(1 - pred + 1e-10))[0, 0]

    # regression
    loss = 1/2 * (sparse.linalg.norm(para_y - para_X_hat @ para_w - para_b) ** 2)
    reg = 0
    for i in range(h):
        D = para_D[i]
        reg += sum([sparse.linalg.norm(Dg @ para_w_list[i]) for Dg in D])
    loss = loss / m + reg
    return loss


def loss_fig(para_loss):
    plt.figure()
    plt.plot(para_loss[1:])
    plt.title('Training Loss Change')
    plt.xlabel('Epoch')
    plt.ylabel('Objective function value')
    name = 'res/loss.jpg'
    plt.savefig(name)
    plt.clf()

# calculate metrics
def test(X_test_list, Y_test_list, res_W, b):
    # regression
    mse = []
    rmse = []
    ex_var = []
    nmse = []
    mae = []
    idx_b = 0
    for i in range(m):
        bias = b.toarray()[idx_b:idx_b + Y_test_list[i].shape[0], 0]
        pred_y = (X_test_list[i] @ res_W[:, i] + bias).reshape((-1,))
        idx_b += Y_test_list[i].shape[0]
        real_y = Y_test_list[i].reshape((-1,))
        mse.append(mean_squared_error(real_y, pred_y))
        rmse.append(np.sqrt(mean_squared_error(real_y, pred_y)))
        ex_var.append(explained_variance_score(real_y, pred_y))
        nmse.append(np.sum((real_y - pred_y) ** 2) / np.sum((real_y - np.mean(real_y)) ** 2))
        mae.append(mean_absolute_error(real_y, pred_y))

    return (sum(nmse) / m, sum(rmse) / m, sum(ex_var) / m, sum(mae) / m)


if __name__ == '__main__':
    # X_hay, y, w initialization
    sub_X_hat = np.zeros((sum(ns), p*m))
    for i in range(m):
        sub_X_hat[sum(ns[:i]):sum(ns[:i])+ns[i], p * i:p * i + p] = X_train_list[i]
    X_hat = sub_X_hat
    for i in range(h-1):
        X_hat = np.concatenate((X_hat, sub_X_hat), axis=1)
    X_hat = sparse.csr_matrix(X_hat)
    y = Y_train_list[0]
    for i in range(1, m):
        y = np.concatenate((y, Y_train_list[i]), axis=0)
    y = sparse.csr_matrix(y)

    gama1, gama2, gama3, gama4, phi = gama
    # Dg generation
    t1 = time.time()
    Dg_list = []
    for i in range(h):
        Dg_list.append([])
        if Gs[i][0] is not None:
            for idx1_1 in range(m):
                for idx1_2 in range(idx1_1, m):
                    G_g = (idx1_1, idx1_2)
                    Dg = gen_Dg('task_group', G_g, i, gama1, phi, col=True)
                    Dg_list[i].append(Dg)

        if Gs[i][1] is not None:
            for G_g in range(p):
                Dg = gen_Dg('feature_learn', G_g, i, gama2, phi, row=True)
                Dg_list[i].append(Dg)

        if Gs[i][2] is not None:
            for idx2 in range(p):
                for idx1_1 in range(m):
                    for idx1_2 in range(idx1_1, m):
                        idx1 = (idx1_1, idx1_2)
                        G_g = (idx1, idx2)
                        Dg = gen_Dg('task_group', G_g, i, gama3, phi, row=True, col=True)
                        Dg_list[i].append(Dg)
            for idx1_1 in range(m):
                for idx1_2 in range(p):
                    G_g = (idx1_1, idx1_2)
                    Dg = gen_Dg('feature_learn', G_g, i, gama4, phi, row=True, col=True)
                    Dg_list[i].append(Dg)

    # initialize w, b; create loss
    w_list = []
    w = np.zeros((h * p * m, 1))
    w = sparse.csr_matrix(w)
    for i in range(h):
        tmp = np.zeros((p*m, 1))
        tmp = sparse.csr_matrix(tmp)
        w_list.append(tmp)
        w[i * p * m: p * m * i + p * m] = w_list[i]
    b = np.zeros((y.shape[0], 1))
    b = sparse.csr_matrix(b)
    loss_list = []
    loss = loss_fuc(y, X_hat, b, w, w_list, Dg_list, gama, phi)
    loss_list.append(loss)
    # print('   loss: ', loss)

    # start training
    for ct in tqdm(range(epoches)):
        t1 = time.time()
        dg_alpha_list = []
        for i in range(h):
            dg_alpha_list.append([])
            for Dg in Dg_list[i]:
                alpah_g = projection((Dg @ w_list[i]) / miu, i)
                tmp = Dg.T @ alpah_g
                dg_alpha_list[i].append(tmp)
            dg_alpha_list[i] = sum(dg_alpha_list[i])
        reg_grad = dg_alpha_list[0].toarray()
        for i in range(1, h):
            reg_grad = np.concatenate((reg_grad, dg_alpha_list[i].toarray()))

        # classification
        # tmp = sigmoid((X_hat @ w + b).toarray())
        # regression
        tmp = X_hat @ w + b

        grad_w = X_hat.T @ (tmp - y) + sparse.csr_matrix(reg_grad)
        grad_b = tmp - y

        new_w = w - lr * sparse.csr_matrix(grad_w)
        new_b = b - lr * sparse.csr_matrix(grad_b)

        new_theta = 2 / (ct + 3)
        new_w = new_w + new_theta * (1 - theta) / theta * (new_w - w)
        new_b = new_b + new_theta * (1 - theta) / theta * (new_b - b)
        theta = copy.deepcopy(new_theta)

        for i in range(h):
            w_list[i] = w[i * p * m: p * m * i + p * m]

        loss = loss_fuc(y, X_hat, new_b, new_w, w_list, Dg_list, gama, phi)
        # print('   loss:', loss)

        loss_list.append(loss)
        w = copy.deepcopy(new_w)
        b = copy.deepcopy(new_b)
        for i in range(h):
            w_list[i] = w[i * p * m: p * m * i + p * m]
        res_W = sum(w_list).toarray().reshape(m, p).T
                
        # adjust lr
        if np.mod(ct, 30) == 0:
            if ct == 0:
                lr /= 2
            else:
                lr /= 10
        
        # convergence criterion
        if abs(loss - loss_list[-2]) / loss_list[-2] < 1e-4:
            break

    # calculate metrics
    res_W = sum(w_list).toarray().reshape(m, p).T
    nmse, rmse, ex_var, mae = test(X_test_list, Y_test_list, res_W, b)


    print(gama)
    print('nMSE: ', nmse)
    print('MAE: ', mae)
    print('EV: ', ex_var)
