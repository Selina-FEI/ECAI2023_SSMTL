import copy
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
from tabulate import tabulate
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import time
from scipy import sparse
from sklearn.metrics import roc_auc_score
import scipy.io as io

seed = 2023
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# net paras
n_epochs = 200
batch_size = 128

# reg paras
gamma1 = 1e-6
gamma2 = 1e-6
phi = 2
gamma = (gamma1, gamma2, gamma1, gamma2, phi)

# output paras
loss_change = np.zeros(n_epochs)
tr_accs = 0
tst_accs = 0
aucs = 0


# load data
class my_dataset(Dataset):
    def __init__(self, data, string):
        self.x = {}
        self.y = {}
        if string == 'train':
            for i in range(7):
                self.x[str(i)] = torch.Tensor(data[str(i)][0][0])
                self.y[str(i)] = torch.Tensor(data[str(i)][0][2])
        if string == 'test':
            for i in range(7):
                self.x[str(i)] = torch.Tensor(data[str(i)][0][1])
                self.y[str(i)] = torch.Tensor(data[str(i)][0][3])

    def __len__(self):
        return self.x[str(0)].shape[0]

    def __getitem__(self, idx):
        res_x = {}
        res_y = {}
        for i in range(7):
            res_x[str(i)] = self.x[str(i)][idx, :]
            res_y[str(i)] = self.y[str(i)][idx, :]
        return res_x, res_y

data = io.loadmat('data/COVER/01_COVER_2000.mat')
train_data = my_dataset(data, 'train')
test_data = my_dataset(data, 'test')


train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=4)


# scipy sparse matrix -> torchsparse tensor
def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# stack the weight matrices of the same layer to construct tensor
def get_weight_tensor(model):
    n_layer = 0
    n_tasks = [0, 0, 0, 0]
    res = []
    for name, para in model.named_parameters():
        if "weight" in name:
            if n_tasks[n_layer] == 0:
                res.append(torch.zeros((para.shape[0], para.shape[1], 7)))
            res[n_layer][:, :, n_tasks[n_layer]] = para
            n_tasks[n_layer] += 1

            if n_layer == 3:
                n_layer = 0
            else:
                n_layer += 1
    return res


# accuracy & AUC calculation
def test(model, data_loader, string):
    correct = [0] * 7
    total = [0] * 7
    auc = [0] * 7
    all_label = [[]] * 7
    all_pred = [[]] * 7
    with torch.no_grad():
        for all_images, all_targets in data_loader:
            outputs = model(all_images)
            for i in range(7):
                images = all_images[str(i)].cuda()
                targets = all_targets[str(i)].cuda()
                all_pred[i].extend(outputs[i].cpu().numpy())
                label = (i == targets).int().cuda()
                all_label[i].extend(label.cpu().numpy())
                predicted = (outputs[i] > 0.5).long()
                tmp = (predicted == label).float()
                correct[i] += tmp.sum().item()
                total[i] += label.shape[0]
    sub_accs = []
    for i in range(7):
        sub_accs.append(correct[i] / total[i])
        auc[i] = roc_auc_score(np.array(all_label[i]), np.array(all_pred[i]).reshape((-1,)))
        # print(string, ' Accuracy for Net', str(i+1), ': ', correct[i] / total[i])
    acc = sum(sub_accs) / 7
    print(string, ' Accuracy for the Whole Net: ', acc)
    if string == 'Test':
        print('AUC: ', sum(auc) / 7)
        return acc, sub_accs, sum(auc) / 7, auc
    else:
        return acc, sub_accs

# generate dg (linear operator)
def gen_Dg(size, optype, sub_type, G_g):
    d1, d2, t = size
    dg = None
    # net-wise, task-level
    if sub_type == 'net':
        dg = np.zeros(t)
        i, j = G_g
        dg[i] = 1
        dg[j] = -1
    # neuron-wise, feature-level
    elif sub_type == 'neuron' and optype == 'feature_learn':
        dg = [np.zeros(t), np.zeros(d2)]
        i, j = G_g
        dg[0][i] = 1
        dg[1][j] = 1
    # neuron-wise, task-level
    elif sub_type == 'neuron' and optype == 'task_group':
        dg = [np.zeros(t), np.zeros(d2)]
        i, j, k = G_g
        dg[0][i] = 1
        dg[0][j] = -1
        dg[1][k] = 1
    # weight-wise, feature-level
    elif sub_type == 'weight':
        dg = [np.zeros(t), np.zeros(d2), np.zeros(d1)]
        i, j, k = G_g
        dg[0][i] = 1
        dg[1][j] = 1
        dg[2][k] = 1
    # kronecker product
    if sub_type == 'net':
        res = sparse.kron(sparse.kron(sparse.csr_matrix(dg), sparse.csr_matrix(np.identity(d2))),
                          sparse.csr_matrix(np.identity(d1)))
    if sub_type == 'nueron':
        res = sparse.kron(sparse.kron(sparse.csr_matrix(dg[0]), sparse.csr_matrix(dg[1])),
                          sparse.csr_matrix(np.identity(d1)))
    if sub_type == 'weight':
        res = sparse.kron(sparse.kron(sparse.csr_matrix(dg[0]), sparse.csr_matrix(dg[1])),
                          sparse.csr_matrix(dg[2]))
    res = scipy_sparse_mat_to_torch_sparse_tensor(res)
    res = res.cuda()
    return res
    
 
# generate the regularization term based on dgs
def genRegularization(model):
    tensor_W = get_weight_tensor(model)
    l_group_list = []
    for t in tensor_W:
        t = t.cuda()
        d1, d2, m = t.shape

        vec_w = t.T.reshape(-1, 1)
        reg_Dg_list = []
        for i in range(m):
            for j in range(i+1, m):
                G_g = (i, j)
                Dg = gen_Dg(t.shape, 'task_group', 'net', G_g)
                reg_Dg_list.append(torch.sqrt(torch.sum(torch.pow(torch.matmul(Dg, vec_w), 2))))
        for k in range(d2):
            for i in range(m):
                for j in range(i + 1, m):
                    G_g = (i, j, k)
                    Dg = gen_Dg(t.shape, 'task_group', 'neuron', G_g)
                    reg_Dg_list.append(torch.sqrt(torch.sum(torch.pow(torch.matmul(Dg, vec_w), 2))))
        for i in range(m):
            for j in range(d2):
                G_g = (i, j)
                Dg = gen_Dg(t.shape, 'feature_learn', 'neuron', G_g)
                reg_Dg_list.append(torch.sqrt(torch.sum(torch.pow(torch.matmul(Dg, vec_w), 2))))
        for i in range(m):
            for j in range(d2):
                for k in range(d1):
                    G_g = (i, j, k)
                    Dg = gen_Dg(t.shape, 'task_group', 'net', G_g)
                    reg_Dg_list.append(torch.sqrt(torch.sum(torch.pow(torch.matmul(Dg, vec_w), 2))))
        reg_Dg = sum(reg_Dg_list)
        l_group_list.append(reg_Dg)
    return sum(l_group_list)


# generate the reluarization term directly without the unified form
def regularization(model, gamma):
    g1, g2, g3, g4, p = gamma
    tensor_W = get_weight_tensor(model)
    l_feature_element = []
    l_feature_neuron = []
    l_task_neuron = []
    l_task_net = []
    i = 0
    for t in tensor_W:
        t = t.cuda()
        d1, d2, m = t.shape
        tmp = torch.sum(torch.sum(torch.sqrt(torch.sum(t ** 2, dim=1)), dim=0))
        tmp *= torch.tensor(g2 * p ** i)
        l_feature_neuron.append(tmp)
        tmp = torch.sum(torch.abs(t))
        tmp *= torch.tensor(g4 / p ** i)
        l_feature_element.append(tmp)
        tmp = 0
        for j in range(d2):
            for k1 in range(m):
                for k2 in range(k1+1, m):
                    tmp += torch.linalg.norm(t[:, j, k1] - t[:, j, k2])
        tmp *= torch.tensor(g1 * p ** i)
        l_task_neuron.append(tmp)
        tmp = 0
        for k1 in range(m):
            for k2 in range(k1+1, m):
                tmp += torch.sum((t[:, :, k1] - t[:, :, k2]) ** 2)
        tmp *= torch.tensor(g3 / p ** i)
        l_task_net.append(tmp)
        i += 1
    res = sum(l_task_net) + sum(l_task_neuron) + sum(l_feature_neuron) + sum(l_feature_element)
    return res


# net structure
class soft_sharing_net(torch.nn.Module):
    def __init__(self):
        super(soft_sharing_net, self).__init__()
        self.subset1 = torch.nn.Sequential(
            torch.nn.Linear(54, 50),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(50, 50),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(50, 25),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(25, 1),
            torch.nn.Sigmoid()
        )
        self.subset2 = torch.nn.Sequential(
            torch.nn.Linear(54, 50),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(50, 50),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(50, 25),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(25, 1),
            torch.nn.Sigmoid()
        )
        self.subset3 = torch.nn.Sequential(
            torch.nn.Linear(54, 50),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(50, 50),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(50, 25),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(25, 1),
            torch.nn.Sigmoid()
        )
        self.subset4 = torch.nn.Sequential(
            torch.nn.Linear(54, 50),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(50, 50),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(50, 25),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(25, 1),
            torch.nn.Sigmoid()
        )
        self.subset5 = torch.nn.Sequential(
            torch.nn.Linear(54, 50),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(50, 50),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(50, 25),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(25, 1),
            torch.nn.Sigmoid()
        )
        self.subset6 = torch.nn.Sequential(
            torch.nn.Linear(54, 50),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(50, 50),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(50, 25),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(25, 1),
            torch.nn.Sigmoid()
        )
        self.subset7 = torch.nn.Sequential(
            torch.nn.Linear(54, 50),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(50, 50),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(50, 25),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(25, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, input):
        for i in range(7):
            input[str(i)] = input[str(i)].cuda()
        # input = input.view(-1, 784)
        out1 = self.subset1(input['0'])
        out2 = self.subset2(input['1'])
        out3 = self.subset3(input['2'])
        out4 = self.subset4(input['3'])
        out5 = self.subset5(input['4'])
        out6 = self.subset6(input['5'])
        out7 = self.subset7(input['6'])
        return out1, out2, out3, out4, out5, out6, out7


# training
def train():
    model = soft_sharing_net()
    model.cuda()
    optimizer = torch.optim.Adam(params=model.parameters())
    lossfunc = torch.nn.BCELoss()
    lossfunc = lossfunc.cuda()
    for epoch in tqdm(range(n_epochs)):
        train_loss = 0.0
        for all_data, all_target in train_loader:
            optimizer.zero_grad()
            loss_list = []
            output = model(all_data)
            for i in range(7):
                data, target = all_data[str(i)].cuda(), all_target[str(i)].cuda()
                subtarget = (target == i).int().cuda()
                subout = output[i]
                loss_list.append(lossfunc(subout.double(), subtarget.double()))
            loss = sum(loss_list) / 7

            reg = regularization(model, gamma)
            reg = reg.cuda()
            loss += reg
            loss.backward()
            optimizer.step()
            train_loss += loss.cpu().item()

        train_loss = train_loss / len(train_loader.dataset)
        loss_change[epoch] = train_loss

    print(gamma, ': ')
    tr_accs, _ = test(model, train_loader, "Train")
    tst_accs, _, aucs, _ = test(model, test_loader, "Test")


if __name__ == '__main__':
    train()
