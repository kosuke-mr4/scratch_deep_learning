# coding: utf-8
import sys, os

sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 10000  # 繰り返しの回数を適宜設定する
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 勾配の計算
    # grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)

    # パラメータの更新
    for key in ("W1", "b1", "W2", "b2"):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # １エポックごとに認識精度を計算
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))
        # train acc, test acc | 0.09863333333333334, 0.0958
        # train acc, test acc | 0.8010166666666667, 0.8075
        # train acc, test acc | 0.87685, 0.8792
        # train acc, test acc | 0.8984666666666666, 0.9017
        # train acc, test acc | 0.9086166666666666, 0.91
        # train acc, test acc | 0.9140833333333334, 0.916
        # train acc, test acc | 0.92, 0.9213
        # train acc, test acc | 0.9246333333333333, 0.9265
        # train acc, test acc | 0.9276, 0.9284
        # train acc, test acc | 0.93175, 0.9331
        # train acc, test acc | 0.9338, 0.9343
        # train acc, test acc | 0.93655, 0.9369
        # train acc, test acc | 0.9394833333333333, 0.9395
        # train acc, test acc | 0.9413, 0.9412
        # train acc, test acc | 0.9436333333333333, 0.9423
        # train acc, test acc | 0.9460166666666666, 0.9444
        # train acc, test acc | 0.9473833333333334, 0.9465

# グラフの描画
markers = {"train": "o", "test": "s"}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label="train acc")
plt.plot(x, test_acc_list, label="test acc", linestyle="--")
# plt.plot(train_loss_list)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc="lower right")
plt.show()
