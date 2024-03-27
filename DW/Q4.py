import numpy as np


def xavier(num_layers, units_list):
    """参数是服从[-a, a]的均匀分布，其中a为sqrt(6)/sqrt(n_in+n_out) """
    params = {}
    np.random.seed(20220727)
    for layer in range(1, num_layers):
        # 第一层输入层无参数，之后每层都有参数
        a = np.sqrt(6) / np.sqrt(units_list[layer-1] + units_list[layer])
        params['w'+str(layer)] = np.random.uniform(-a, a, size=(units_list[layer], units_list[layer-1]))
        params['gamma' + str(layer)] = np.ones(shape=(1, units_list[layer]))
        params['beta' + str(layer)] = np.zeros(shape=(1, units_list[layer]))
    return params


class SGD(object):
    def __init__(self, lr=0.001):
        self.lr = lr

    def optimize(self, weight_num, params, grads, batch_size, bn=False):
        for i in range(1, weight_num + 1):
            params['w' + str(i)] -= self.lr * grads['dw' + str(i)] / batch_size
            if bn:
                params['gamma'+str(i)] -= grads['dgamma'+str(i)] / batch_size
                params['beta'+str(i)] -= grads['dbeta'+str(i)] / batch_size
        return params


# loss fn
def cross_entropy(p, q):
    """p表示真实分布，q表示预测分布（必须onehot编码）"""
    if p.ndim == 1 or q.ndim == 1:
        p = p.reshape(1, -1)
        q = q.reshape(1, -1)

    m = p.shape[0]
    loss = p * np.log(q+1e-5)
    return -np.sum(loss) / m


# activate fn
def tanh(x):
    return np.tanh(x)


def tanh_gradient(x):
    return 1 - tanh(x) ** 2


def softmax(x):
    if x.ndim == 1:
        x = x.reshape(1, -1)

    f1 = lambda x: np.exp(x - np.max(x))
    f2 = lambda x: x / np.sum(x)
    x = np.apply_along_axis(f1, axis=1, arr=x)
    x = np.apply_along_axis(f2, axis=1, arr=x)
    return x


def softmax_gradient(x, label):
    return softmax(x) - label


def onehot(labels, classes=None):
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)

    num_data = labels.shape[0]
    index_offset = np.arange(num_data) * classes
    labels_onehot = np.zeros(shape=(num_data, classes))
    labels_onehot.flat[index_offset + labels.ravel()] = 1
    return labels_onehot


class MLP(object):

    def __init__(self, num_layers, units_list=None):
        self.weight_num = num_layers - 1

        self.params = xavier(num_layers, units_list)
        self.optimizer = SGD()
        self.bn_param = {}

    def forward(self, x):
        """前向传播"""
        net_inputs = []  # 各层的输入
        net_outputs = []  # 各层激活后的输出
        net_d = []

        net_inputs.append(x)
        net_outputs.append(x)
        net_d.append(np.ones(x.shape[1:]))  # 输入层无丢弃概率
        for i in range(1, self.weight_num):  # 参数数量比层数少1
            x = x @ self.params['w'+str(i)].T
            net_inputs.append(x)
            x = tanh(x)
            net_outputs.append(x)
        out = x @ self.params['w'+str(self.weight_num)].T
        net_inputs.append(out)
        out = softmax(out)
        net_outputs.append(out)
        return {'net_inputs': net_inputs, 'net_outputs': net_outputs, 'd': net_d}, out

    def backward(self, nets, y, pred, dropout_prob=None):
        """
        dz[out] = out - y
        dw[out] = dz[out] @ outputs[out-1].T
        db[out] = dz[out]
        dz[i] = W[i+1]dz[i+1] * grad(z[i])
        dw[i] = dz[i] @ outputs[i-1]
        db[i] = dz[i]sa
        """
        grads = dict()
        grads['dz'+str(self.weight_num)] = (pred - y)
        grads['dw'+str(self.weight_num)] = grads['dz'+str(self.weight_num)].T @ nets['net_outputs'][self.weight_num-1]

        for i in reversed(range(1, self.weight_num)):
            temp = grads['dz' + str(i + 1)] @ self.params['w' + str(i + 1)] * tanh_gradient(nets['net_inputs'][i])
            if dropout_prob:
                temp = temp * nets['d'][i] / (1-dropout_prob)
            grads['dz'+str(i)] = temp   # [b, 128]
            grads['dw'+str(i)] = grads['dz'+str(i)].T @ nets['net_outputs'][i-1]
        return grads

    def train(self, data_loader, valid_loader, epochs, learning_rate):
        losses_train = []
        losses_valid = []
        for epoch in range(epochs):
            print("epoch", epoch)
            # training
            epoch_loss_train = 0
            for step, (x, y) in enumerate(data_loader):
                x = x.reshape(-1, 28 * 28)
                y = onehot(y, 10)
                nets, pred = self.forward(x)
                loss = cross_entropy(y, pred)
                epoch_loss_train += loss
                grads = self.backward(nets, y, pred)
                # SGD
                self.params = self.optimizer.optimize(self.weight_num, self.params, grads, y.shape[0])

                if step % 100 == 0:
                    print("epoch {} training step {} loss {:.4f}".format(epoch, step, loss))
            losses_train.append(epoch_loss_train)
            print(epoch_loss_train)
            data_loader.restart()

            epoch_loss_valid = 0
            for step, (x, y) in enumerate(valid_loader):
                x = x.reshape(-1, 28 * 28)
                y = onehot(y, 10)
                nets, pred = self.forward(x)
                loss = cross_entropy(y, pred)
                epoch_loss_valid += loss

                if step % 100 == 0:
                    print("epoch {} validation step {} loss {:.4f}".format(epoch, step, loss))
            losses_valid.append(epoch_loss_valid)
            valid_loader.restart()
        his = {'train_loss': losses_train, 'valid_loss': losses_valid}
        return his


# TODO: 1. dataloader实现；2. dropout实现; 3.batch_norm实现；
