import os
import sys

sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net import MultiLayerNet
from common.optimizer import SGD

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 過学習を再現するために、学習データを削減
x_train = x_train[:300]
t_train = t_train[:300]

# weight decay（荷重減衰）の設定 =======================
# weight_decay_lambda = 0 # weight decayを使用しない場合
weight_decay_lambda = 0.1
# ====================================================

network = MultiLayerNet(
    input_size=784,
    hidden_size_list=[100, 100, 100, 100, 100, 100],
    output_size=10,
    weight_decay_lambda=weight_decay_lambda,
)
optimizer = SGD(lr=0.01)

max_epochs = 201
train_size = x_train.shape[0]
batch_size = 100

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)
epoch_cnt = 0

for i in range(1000000000):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grads = network.gradient(x_batch, t_batch)
    optimizer.update(network.params, grads)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print(
            "epoch:"
            + str(epoch_cnt)
            + ", train acc:"
            + str(train_acc)
            + ", test acc:"
            + str(test_acc)
        )

        epoch_cnt += 1
        if epoch_cnt >= max_epochs:
            break


# 3.グラフの描画==========
markers = {"train": "o", "test": "s"}
x = np.arange(max_epochs)
plt.plot(x, train_acc_list, marker="o", label="train", markevery=10)
plt.plot(x, test_acc_list, marker="s", label="test", markevery=10)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc="lower right")
plt.show()


# ============== 1/16 ==============
# epoch:0 | 0.117 - 0.112
# ../common/multi_layer_net_extend.py:122: RuntimeWarning: overflow encountered in square
#   weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)
# /Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/numpy/core/fromnumeric.py:90: RuntimeWarning: overflow encountered in reduce
#   return ufunc.reduce(obj, axis, dtype, out, **passkwargs)
# ../common/multi_layer_net_extend.py:122: RuntimeWarning: invalid value encountered in double_scalars
#   weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)
# ../common/functions.py:32: RuntimeWarning: invalid value encountered in subtract
#   x = x - np.max(x, axis=-1, keepdims=True)  # オーバーフロー対策
# ../common/layers.py:12: RuntimeWarning: invalid value encountered in less_equal
#   self.mask = x <= 0
# epoch:1 | 0.097 - 0.108
# epoch:2 | 0.097 - 0.117
# epoch:3 | 0.097 - 0.161
# epoch:4 | 0.097 - 0.197
# epoch:5 | 0.097 - 0.215
# epoch:6 | 0.097 - 0.229
# epoch:7 | 0.097 - 0.256
# epoch:8 | 0.097 - 0.282
# epoch:9 | 0.097 - 0.298
# epoch:10 | 0.097 - 0.312
# epoch:11 | 0.097 - 0.323
# epoch:12 | 0.097 - 0.353
# epoch:13 | 0.097 - 0.36
# epoch:14 | 0.097 - 0.375
# epoch:15 | 0.097 - 0.398
# epoch:16 | 0.097 - 0.408
# epoch:17 | 0.097 - 0.418
# epoch:18 | 0.097 - 0.433
# epoch:19 | 0.097 - 0.448
# No handles with labels found to put in legend.
# ============== 2/16 ==============
# epoch:0 | 0.117 - 0.144
# epoch:1 | 0.097 - 0.117
# epoch:2 | 0.097 - 0.124
# epoch:3 | 0.097 - 0.153
# epoch:4 | 0.097 - 0.189
# epoch:5 | 0.097 - 0.206
# epoch:6 | 0.097 - 0.234
# epoch:7 | 0.097 - 0.259
# epoch:8 | 0.097 - 0.289
# epoch:9 | 0.097 - 0.296
# epoch:10 | 0.097 - 0.322
# epoch:11 | 0.097 - 0.343
# epoch:12 | 0.097 - 0.376
# epoch:13 | 0.097 - 0.384
# epoch:14 | 0.097 - 0.418
# epoch:15 | 0.097 - 0.431
# epoch:16 | 0.097 - 0.452
# epoch:17 | 0.097 - 0.468
# epoch:18 | 0.097 - 0.487
# epoch:19 | 0.097 - 0.503
# No handles with labels found to put in legend.
# ============== 3/16 ==============
# epoch:0 | 0.092 - 0.094
# epoch:1 | 0.358 - 0.108
# epoch:2 | 0.488 - 0.132
# epoch:3 | 0.584 - 0.159
# epoch:4 | 0.647 - 0.185
# epoch:5 | 0.714 - 0.221
# epoch:6 | 0.782 - 0.26
# epoch:7 | 0.802 - 0.301
# epoch:8 | 0.854 - 0.332
# epoch:9 | 0.883 - 0.357
# epoch:10 | 0.903 - 0.397
# epoch:11 | 0.903 - 0.438
# epoch:12 | 0.932 - 0.468
# epoch:13 | 0.943 - 0.503
# epoch:14 | 0.945 - 0.529
# epoch:15 | 0.966 - 0.551
# epoch:16 | 0.966 - 0.577
# epoch:17 | 0.975 - 0.584
# epoch:18 | 0.978 - 0.614
# epoch:19 | 0.978 - 0.627
# No handles with labels found to put in legend.
# ============== 4/16 ==============
# epoch:0 | 0.104 - 0.121
# epoch:1 | 0.208 - 0.143
# epoch:2 | 0.342 - 0.199
# epoch:3 | 0.483 - 0.271
# epoch:4 | 0.601 - 0.344
# epoch:5 | 0.648 - 0.416
# epoch:6 | 0.689 - 0.474
# epoch:7 | 0.708 - 0.529
# epoch:8 | 0.744 - 0.572
# epoch:9 | 0.764 - 0.604
# epoch:10 | 0.781 - 0.646
# epoch:11 | 0.79 - 0.684
# epoch:12 | 0.804 - 0.702
# epoch:13 | 0.815 - 0.723
# epoch:14 | 0.828 - 0.756
# epoch:15 | 0.832 - 0.761
# epoch:16 | 0.846 - 0.775
# epoch:17 | 0.841 - 0.79
# epoch:18 | 0.856 - 0.81
# epoch:19 | 0.867 - 0.816
# No handles with labels found to put in legend.
# ============== 5/16 ==============
# epoch:0 | 0.093 - 0.087
# epoch:1 | 0.095 - 0.103
# epoch:2 | 0.101 - 0.229
# epoch:3 | 0.097 - 0.376
# epoch:4 | 0.108 - 0.477
# epoch:5 | 0.121 - 0.577
# epoch:6 | 0.131 - 0.639
# epoch:7 | 0.148 - 0.671
# epoch:8 | 0.172 - 0.718
# epoch:9 | 0.194 - 0.743
# epoch:10 | 0.216 - 0.777
# epoch:11 | 0.239 - 0.789
# epoch:12 | 0.245 - 0.806
# epoch:13 | 0.263 - 0.818
# epoch:14 | 0.281 - 0.829
# epoch:15 | 0.307 - 0.847
# epoch:16 | 0.313 - 0.855
# epoch:17 | 0.319 - 0.86
# epoch:18 | 0.343 - 0.871
# epoch:19 | 0.348 - 0.874
# No handles with labels found to put in legend.
# ============== 6/16 ==============
# epoch:0 | 0.089 - 0.126
# epoch:1 | 0.118 - 0.255
# epoch:2 | 0.116 - 0.444
# epoch:3 | 0.041 - 0.585
# epoch:4 | 0.116 - 0.712
# epoch:5 | 0.116 - 0.762
# epoch:6 | 0.116 - 0.792
# epoch:7 | 0.116 - 0.803
# epoch:8 | 0.116 - 0.821
# epoch:9 | 0.116 - 0.834
# epoch:10 | 0.118 - 0.846
# epoch:11 | 0.071 - 0.858
# epoch:12 | 0.081 - 0.875
# epoch:13 | 0.109 - 0.89
# epoch:14 | 0.117 - 0.906
# epoch:15 | 0.117 - 0.911
# epoch:16 | 0.117 - 0.92
# epoch:17 | 0.117 - 0.928
# epoch:18 | 0.113 - 0.931
# epoch:19 | 0.073 - 0.937
# No handles with labels found to put in legend.
# ============== 7/16 ==============
# epoch:0 | 0.099 - 0.114
# epoch:1 | 0.116 - 0.341
# epoch:2 | 0.116 - 0.618
# epoch:3 | 0.162 - 0.745
# epoch:4 | 0.121 - 0.802
# epoch:5 | 0.116 - 0.851
# epoch:6 | 0.116 - 0.874
# epoch:7 | 0.116 - 0.898
# epoch:8 | 0.116 - 0.912
# epoch:9 | 0.116 - 0.93
# epoch:10 | 0.116 - 0.947
# epoch:11 | 0.116 - 0.957
# epoch:12 | 0.116 - 0.97
# epoch:13 | 0.117 - 0.976
# epoch:14 | 0.117 - 0.98
# epoch:15 | 0.117 - 0.985
# epoch:16 | 0.117 - 0.99
# epoch:17 | 0.117 - 0.992
# epoch:18 | 0.117 - 0.995
# epoch:19 | 0.117 - 0.996
# No handles with labels found to put in legend.
# ============== 8/16 ==============
# epoch:0 | 0.105 - 0.106
# epoch:1 | 0.116 - 0.374
# epoch:2 | 0.117 - 0.651
# epoch:3 | 0.117 - 0.73
# epoch:4 | 0.116 - 0.771
# epoch:5 | 0.117 - 0.837
# epoch:6 | 0.117 - 0.874
# epoch:7 | 0.117 - 0.933
# epoch:8 | 0.117 - 0.949
# epoch:9 | 0.117 - 0.961
# epoch:10 | 0.117 - 0.974
# epoch:11 | 0.117 - 0.98
# epoch:12 | 0.117 - 0.989
# epoch:13 | 0.117 - 0.989
# epoch:14 | 0.117 - 0.993
# epoch:15 | 0.117 - 0.995
# epoch:16 | 0.117 - 0.997
# epoch:17 | 0.117 - 0.997
# epoch:18 | 0.117 - 0.997
# epoch:19 | 0.117 - 0.999
# No handles with labels found to put in legend.
# ============== 9/16 ==============
# epoch:0 | 0.092 - 0.125
# epoch:1 | 0.117 - 0.452
# epoch:2 | 0.116 - 0.619
# epoch:3 | 0.117 - 0.801
# epoch:4 | 0.116 - 0.889
# epoch:5 | 0.116 - 0.927
# epoch:6 | 0.116 - 0.96
# epoch:7 | 0.116 - 0.966
# epoch:8 | 0.116 - 0.99
# epoch:9 | 0.116 - 0.994
# epoch:10 | 0.116 - 0.994
# epoch:11 | 0.117 - 0.996
# epoch:12 | 0.117 - 0.998
# epoch:13 | 0.117 - 0.998
# epoch:14 | 0.117 - 0.999
# epoch:15 | 0.117 - 1.0
# epoch:16 | 0.117 - 0.999
# epoch:17 | 0.117 - 1.0
# epoch:18 | 0.117 - 1.0
# epoch:19 | 0.117 - 0.999
# No handles with labels found to put in legend.
# ============== 10/16 ==============
# epoch:0 | 0.092 - 0.159
# epoch:1 | 0.117 - 0.617
# epoch:2 | 0.117 - 0.733
# epoch:3 | 0.117 - 0.814
# epoch:4 | 0.117 - 0.877
# epoch:5 | 0.117 - 0.879
# epoch:6 | 0.117 - 0.94
# epoch:7 | 0.117 - 0.922
# epoch:8 | 0.117 - 0.978
# epoch:9 | 0.117 - 0.977
# epoch:10 | 0.117 - 0.984
# epoch:11 | 0.117 - 0.994
# epoch:12 | 0.117 - 0.995
# epoch:13 | 0.117 - 0.997
# epoch:14 | 0.117 - 0.997
# epoch:15 | 0.117 - 0.89
# epoch:16 | 0.117 - 0.992
# epoch:17 | 0.117 - 0.999
# epoch:18 | 0.117 - 1.0
# epoch:19 | 0.117 - 1.0
# No handles with labels found to put in legend.
# ============== 11/16 ==============
# epoch:0 | 0.105 - 0.164
# epoch:1 | 0.116 - 0.546
# epoch:2 | 0.116 - 0.668
# epoch:3 | 0.116 - 0.705
# epoch:4 | 0.116 - 0.751
# epoch:5 | 0.116 - 0.814
# epoch:6 | 0.116 - 0.857
# epoch:7 | 0.116 - 0.869
# epoch:8 | 0.116 - 0.888
# epoch:9 | 0.116 - 0.883
# epoch:10 | 0.116 - 0.834
# epoch:11 | 0.116 - 0.897
# epoch:12 | 0.116 - 0.899
# epoch:13 | 0.116 - 0.895
# epoch:14 | 0.116 - 0.903
# epoch:15 | 0.116 - 0.902
# epoch:16 | 0.116 - 0.903
# epoch:17 | 0.116 - 0.903
# epoch:18 | 0.116 - 0.905
# epoch:19 | 0.116 - 0.993
# No handles with labels found to put in legend.
# ============== 12/16 ==============
# epoch:0 | 0.116 - 0.098
# epoch:1 | 0.116 - 0.338
# epoch:2 | 0.116 - 0.559
# epoch:3 | 0.117 - 0.586
# epoch:4 | 0.116 - 0.595
# epoch:5 | 0.116 - 0.6
# epoch:6 | 0.116 - 0.603
# epoch:7 | 0.116 - 0.609
# epoch:8 | 0.116 - 0.611
# epoch:9 | 0.116 - 0.605
# epoch:10 | 0.116 - 0.616
# epoch:11 | 0.116 - 0.616
# epoch:12 | 0.116 - 0.616
# epoch:13 | 0.116 - 0.617
# epoch:14 | 0.116 - 0.618
# epoch:15 | 0.116 - 0.616
# epoch:16 | 0.116 - 0.618
# epoch:17 | 0.116 - 0.618
# epoch:18 | 0.116 - 0.618
# epoch:19 | 0.116 - 0.618
# No handles with labels found to put in legend.
# ============== 13/16 ==============
# epoch:0 | 0.094 - 0.222
# epoch:1 | 0.117 - 0.35
# epoch:2 | 0.117 - 0.506
# epoch:3 | 0.116 - 0.553
# epoch:4 | 0.116 - 0.587
# epoch:5 | 0.117 - 0.589
# epoch:6 | 0.117 - 0.607
# epoch:7 | 0.117 - 0.612
# epoch:8 | 0.117 - 0.613
# epoch:9 | 0.117 - 0.615
# epoch:10 | 0.117 - 0.617
# epoch:11 | 0.117 - 0.617
# epoch:12 | 0.117 - 0.621
# epoch:13 | 0.117 - 0.625
# epoch:14 | 0.117 - 0.624
# epoch:15 | 0.117 - 0.628
# epoch:16 | 0.117 - 0.623
# epoch:17 | 0.117 - 0.646
# epoch:18 | 0.117 - 0.662
# epoch:19 | 0.117 - 0.633
# No handles with labels found to put in legend.
# ============== 14/16 ==============
# epoch:0 | 0.087 - 0.122
# epoch:1 | 0.116 - 0.415
# epoch:2 | 0.116 - 0.472
# epoch:3 | 0.116 - 0.491
# epoch:4 | 0.116 - 0.499
# epoch:5 | 0.116 - 0.574
# epoch:6 | 0.116 - 0.571
# epoch:7 | 0.116 - 0.593
# epoch:8 | 0.116 - 0.606
# epoch:9 | 0.116 - 0.592
# epoch:10 | 0.116 - 0.609
# epoch:11 | 0.116 - 0.601
# epoch:12 | 0.116 - 0.604
# epoch:13 | 0.116 - 0.602
# epoch:14 | 0.116 - 0.6
# epoch:15 | 0.116 - 0.601
# epoch:16 | 0.116 - 0.609
# epoch:17 | 0.116 - 0.611
# epoch:18 | 0.116 - 0.611
# epoch:19 | 0.116 - 0.614
# No handles with labels found to put in legend.
# ============== 15/16 ==============
# epoch:0 | 0.094 - 0.217
# epoch:1 | 0.117 - 0.452
# epoch:2 | 0.116 - 0.371
# epoch:3 | 0.116 - 0.544
# epoch:4 | 0.116 - 0.573
# epoch:5 | 0.116 - 0.585
# epoch:6 | 0.116 - 0.582
# epoch:7 | 0.116 - 0.554
# epoch:8 | 0.116 - 0.585
# epoch:9 | 0.117 - 0.609
# epoch:10 | 0.117 - 0.601
# epoch:11 | 0.116 - 0.607
# epoch:12 | 0.116 - 0.614
# epoch:13 | 0.116 - 0.609
# epoch:14 | 0.116 - 0.606
# epoch:15 | 0.117 - 0.615
# epoch:16 | 0.117 - 0.615
# epoch:17 | 0.117 - 0.615
# epoch:18 | 0.116 - 0.604
# epoch:19 | 0.116 - 0.616
# No handles with labels found to put in legend.
# ============== 16/16 ==============
# epoch:0 | 0.117 - 0.195
# epoch:1 | 0.116 - 0.37
# epoch:2 | 0.116 - 0.376
# epoch:3 | 0.117 - 0.462
# epoch:4 | 0.117 - 0.46
# epoch:5 | 0.116 - 0.482
# epoch:6 | 0.116 - 0.5
# epoch:7 | 0.116 - 0.51
# epoch:8 | 0.116 - 0.505
# epoch:9 | 0.117 - 0.513
# epoch:10 | 0.117 - 0.514
# epoch:11 | 0.117 - 0.51
# epoch:12 | 0.117 - 0.524
# epoch:13 | 0.117 - 0.52
# epoch:14 | 0.117 - 0.521
# epoch:15 | 0.117 - 0.527
# epoch:16 | 0.117 - 0.516
# epoch:17 | 0.117 - 0.523
# epoch:18 | 0.117 - 0.523
# epoch:19 | 0.117 - 0.522
# villagecBookPro:chap6 kosuke$ python3 batch_norm_test.py
# ============== 1/16 ==============
# epoch:0 | 0.092 - 0.131
# ../common/multi_layer_net_extend.py:122: RuntimeWarning: overflow encountered in square
#   weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)
# ../common/multi_layer_net_extend.py:122: RuntimeWarning: invalid value encountered in double_scalars
#   weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)
# ../common/functions.py:32: RuntimeWarning: invalid value encountered in subtract
#   x = x - np.max(x, axis=-1, keepdims=True)  # オーバーフロー対策
# ../common/layers.py:12: RuntimeWarning: invalid value encountered in less_equal
#   self.mask = x <= 0
# epoch:1 | 0.097 - 0.145
# epoch:2 | 0.097 - 0.115
# epoch:3 | 0.097 - 0.134
# epoch:4 | 0.097 - 0.142
# epoch:5 | 0.097 - 0.153
# epoch:6 | 0.097 - 0.194
# epoch:7 | 0.097 - 0.204
# epoch:8 | 0.097 - 0.231
# epoch:9 | 0.097 - 0.234
# epoch:10 | 0.097 - 0.259
# epoch:11 | 0.097 - 0.262
# epoch:12 | 0.097 - 0.299
# epoch:13 | 0.097 - 0.317
# epoch:14 | 0.097 - 0.324
# epoch:15 | 0.097 - 0.361
# epoch:16 | 0.097 - 0.367
# epoch:17 | 0.097 - 0.386
# epoch:18 | 0.097 - 0.392
# epoch:19 | 0.097 - 0.396
# No handles with labels found to put in legend.
# ============== 2/16 ==============
# epoch:0 | 0.099 - 0.113
# epoch:1 | 0.097 - 0.076
# epoch:2 | 0.097 - 0.084
# epoch:3 | 0.097 - 0.113
# epoch:4 | 0.097 - 0.141
# epoch:5 | 0.097 - 0.169
# epoch:6 | 0.097 - 0.188
# epoch:7 | 0.097 - 0.21
# epoch:8 | 0.097 - 0.238
# epoch:9 | 0.097 - 0.258
# epoch:10 | 0.097 - 0.28
# epoch:11 | 0.097 - 0.295
# epoch:12 | 0.097 - 0.315
# epoch:13 | 0.097 - 0.331
# epoch:14 | 0.097 - 0.351
# epoch:15 | 0.097 - 0.367
# epoch:16 | 0.097 - 0.385
# epoch:17 | 0.097 - 0.401
# epoch:18 | 0.097 - 0.418
# epoch:19 | 0.097 - 0.429
# No handles with labels found to put in legend.
# ============== 3/16 ==============
# epoch:0 | 0.126 - 0.08
# epoch:1 | 0.281 - 0.087
# epoch:2 | 0.44 - 0.131
# epoch:3 | 0.552 - 0.161
# epoch:4 | 0.6 - 0.189
# epoch:5 | 0.672 - 0.23
# epoch:6 | 0.723 - 0.267
# epoch:7 | 0.768 - 0.3
# epoch:8 | 0.796 - 0.334
# epoch:9 | 0.811 - 0.361
# epoch:10 | 0.839 - 0.389
# epoch:11 | 0.868 - 0.425
# epoch:12 | 0.881 - 0.457
# epoch:13 | 0.899 - 0.483
# epoch:14 | 0.913 - 0.504
# epoch:15 | 0.931 - 0.537
# epoch:16 | 0.934 - 0.557
# epoch:17 | 0.939 - 0.573
# epoch:18 | 0.951 - 0.595
# epoch:19 | 0.955 - 0.605
# No handles with labels found to put in legend.
# ============== 4/16 ==============
# epoch:0 | 0.125 - 0.122
# epoch:1 | 0.303 - 0.123
# epoch:2 | 0.421 - 0.195
# epoch:3 | 0.507 - 0.284
# epoch:4 | 0.587 - 0.356
# epoch:5 | 0.655 - 0.402
# epoch:6 | 0.667 - 0.456
# epoch:7 | 0.706 - 0.494
# epoch:8 | 0.734 - 0.519
# epoch:9 | 0.742 - 0.557
# epoch:10 | 0.762 - 0.583
# epoch:11 | 0.775 - 0.612
# epoch:12 | 0.8 - 0.646
# epoch:13 | 0.803 - 0.666
# epoch:14 | 0.814 - 0.698
# epoch:15 | 0.824 - 0.72
# epoch:16 | 0.844 - 0.735
# epoch:17 | 0.849 - 0.759
# epoch:18 | 0.862 - 0.772
# epoch:19 | 0.865 - 0.784
# No handles with labels found to put in legend.
# ============== 5/16 ==============
# epoch:0 | 0.096 - 0.086
# epoch:1 | 0.098 - 0.151
# epoch:2 | 0.098 - 0.312
# epoch:3 | 0.099 - 0.442
# epoch:4 | 0.102 - 0.512
# epoch:5 | 0.106 - 0.58
# epoch:6 | 0.107 - 0.608
# epoch:7 | 0.109 - 0.644
# epoch:8 | 0.117 - 0.687
# epoch:9 | 0.13 - 0.717
# epoch:10 | 0.137 - 0.743
# epoch:11 | 0.163 - 0.76
# epoch:12 | 0.18 - 0.778
# epoch:13 | 0.193 - 0.795
# epoch:14 | 0.208 - 0.818
# epoch:15 | 0.222 - 0.835
# epoch:16 | 0.219 - 0.838
# epoch:17 | 0.211 - 0.852
# epoch:18 | 0.215 - 0.862
# epoch:19 | 0.212 - 0.87
# No handles with labels found to put in legend.
# ============== 6/16 ==============
# epoch:0 | 0.126 - 0.106
# epoch:1 | 0.119 - 0.207
# epoch:2 | 0.158 - 0.427
# epoch:3 | 0.137 - 0.557
# epoch:4 | 0.121 - 0.629
# epoch:5 | 0.118 - 0.697
# epoch:6 | 0.117 - 0.754
# epoch:7 | 0.117 - 0.778
# epoch:8 | 0.117 - 0.813
# epoch:9 | 0.117 - 0.835
# epoch:10 | 0.117 - 0.857
# epoch:11 | 0.117 - 0.872
# epoch:12 | 0.117 - 0.886
# epoch:13 | 0.117 - 0.895
# epoch:14 | 0.117 - 0.905
# epoch:15 | 0.117 - 0.915
# epoch:16 | 0.117 - 0.925
# epoch:17 | 0.117 - 0.933
# epoch:18 | 0.117 - 0.944
# epoch:19 | 0.117 - 0.948
# No handles with labels found to put in legend.
# ============== 7/16 ==============
# epoch:0 | 0.116 - 0.105
# epoch:1 | 0.116 - 0.262
# epoch:2 | 0.117 - 0.57
# epoch:3 | 0.117 - 0.664
# epoch:4 | 0.117 - 0.721
# epoch:5 | 0.117 - 0.755
# epoch:6 | 0.117 - 0.793
# epoch:7 | 0.117 - 0.812
# epoch:8 | 0.116 - 0.862
# epoch:9 | 0.116 - 0.881
# epoch:10 | 0.116 - 0.903
# epoch:11 | 0.121 - 0.928
# epoch:12 | 0.187 - 0.939
# epoch:13 | 0.117 - 0.951
# epoch:14 | 0.117 - 0.963
# epoch:15 | 0.116 - 0.973
# epoch:16 | 0.116 - 0.979
# epoch:17 | 0.116 - 0.984
# epoch:18 | 0.116 - 0.986
# epoch:19 | 0.116 - 0.991
# No handles with labels found to put in legend.
# ============== 8/16 ==============
# epoch:0 | 0.092 - 0.15
# epoch:1 | 0.116 - 0.373
# epoch:2 | 0.099 - 0.612
# epoch:3 | 0.099 - 0.759
# epoch:4 | 0.099 - 0.808
# epoch:5 | 0.116 - 0.853
# epoch:6 | 0.116 - 0.894
# epoch:7 | 0.117 - 0.935
# epoch:8 | 0.117 - 0.956
# epoch:9 | 0.117 - 0.968
# epoch:10 | 0.117 - 0.979
# epoch:11 | 0.116 - 0.978
# epoch:12 | 0.117 - 0.982
# epoch:13 | 0.117 - 0.987
# epoch:14 | 0.117 - 0.994
# epoch:15 | 0.117 - 0.993
# epoch:16 | 0.117 - 0.999
# epoch:17 | 0.116 - 0.999
# epoch:18 | 0.116 - 0.998
# epoch:19 | 0.116 - 0.998
# No handles with labels found to put in legend.
# ============== 9/16 ==============
# epoch:0 | 0.1 - 0.096
# epoch:1 | 0.116 - 0.417
# epoch:2 | 0.116 - 0.749
# epoch:3 | 0.117 - 0.784
# epoch:4 | 0.116 - 0.821
# epoch:5 | 0.116 - 0.879
# epoch:6 | 0.116 - 0.942
# epoch:7 | 0.117 - 0.97
# epoch:8 | 0.117 - 0.983
# epoch:9 | 0.116 - 0.99
# epoch:10 | 0.116 - 0.995
# epoch:11 | 0.116 - 0.996
# epoch:12 | 0.116 - 0.997
# epoch:13 | 0.116 - 0.999
# epoch:14 | 0.117 - 0.999
# epoch:15 | 0.116 - 0.999
# epoch:16 | 0.116 - 0.999
# epoch:17 | 0.117 - 0.999
# epoch:18 | 0.117 - 1.0
# epoch:19 | 0.117 - 1.0
# No handles with labels found to put in legend.
# ============== 10/16 ==============
# epoch:0 | 0.093 - 0.144
# epoch:1 | 0.116 - 0.638
# epoch:2 | 0.117 - 0.788
# epoch:3 | 0.117 - 0.843
# epoch:4 | 0.116 - 0.866
# epoch:5 | 0.117 - 0.867
# epoch:6 | 0.117 - 0.867
# epoch:7 | 0.117 - 0.892
# epoch:8 | 0.116 - 0.897
# epoch:9 | 0.117 - 0.902
# epoch:10 | 0.117 - 0.987
# epoch:11 | 0.117 - 0.993
# epoch:12 | 0.117 - 0.994
# epoch:13 | 0.116 - 0.994
# epoch:14 | 0.116 - 0.997
# epoch:15 | 0.116 - 0.997
# epoch:16 | 0.116 - 0.998
# epoch:17 | 0.116 - 0.996
# epoch:18 | 0.116 - 0.998
# epoch:19 | 0.116 - 0.998
# No handles with labels found to put in legend.
# ============== 11/16 ==============
# epoch:0 | 0.117 - 0.127
# epoch:1 | 0.116 - 0.718
# epoch:2 | 0.117 - 0.776
# epoch:3 | 0.117 - 0.813
# epoch:4 | 0.117 - 0.836
# epoch:5 | 0.116 - 0.865
# epoch:6 | 0.116 - 0.813
# epoch:7 | 0.116 - 0.889
# epoch:8 | 0.116 - 0.861
# epoch:9 | 0.116 - 0.966
# epoch:10 | 0.117 - 0.9
# epoch:11 | 0.117 - 0.983
# epoch:12 | 0.117 - 0.99
# epoch:13 | 0.116 - 0.99
# epoch:14 | 0.116 - 0.99
# epoch:15 | 0.116 - 0.991
# epoch:16 | 0.116 - 0.988
# epoch:17 | 0.117 - 0.993
# epoch:18 | 0.117 - 0.995
# epoch:19 | 0.117 - 0.99
# No handles with labels found to put in legend.
# ============== 12/16 ==============
# epoch:0 | 0.117 - 0.109
# epoch:1 | 0.117 - 0.453
# epoch:2 | 0.117 - 0.569
# epoch:3 | 0.116 - 0.653
# epoch:4 | 0.116 - 0.645
# epoch:5 | 0.116 - 0.629
# epoch:6 | 0.116 - 0.689
# epoch:7 | 0.116 - 0.692
# epoch:8 | 0.116 - 0.762
# epoch:9 | 0.116 - 0.715
# epoch:10 | 0.116 - 0.868
# epoch:11 | 0.116 - 0.873
# epoch:12 | 0.116 - 0.869
# epoch:13 | 0.116 - 0.872
# epoch:14 | 0.116 - 0.968
# epoch:15 | 0.116 - 0.893
# epoch:16 | 0.116 - 0.966
# epoch:17 | 0.116 - 0.977
# epoch:18 | 0.116 - 0.986
# epoch:19 | 0.116 - 0.981
# No handles with labels found to put in legend.
# ============== 13/16 ==============
# epoch:0 | 0.087 - 0.215
# epoch:1 | 0.116 - 0.374
# epoch:2 | 0.116 - 0.559
# epoch:3 | 0.116 - 0.573
# epoch:4 | 0.117 - 0.544
# epoch:5 | 0.117 - 0.602
# epoch:6 | 0.116 - 0.552
# epoch:7 | 0.117 - 0.606
# epoch:8 | 0.117 - 0.603
# epoch:9 | 0.117 - 0.618
# epoch:10 | 0.117 - 0.617
# epoch:11 | 0.117 - 0.619
# epoch:12 | 0.117 - 0.605
# epoch:13 | 0.117 - 0.618
# epoch:14 | 0.116 - 0.626
# epoch:15 | 0.116 - 0.714
# epoch:16 | 0.116 - 0.704
# epoch:17 | 0.116 - 0.717
# epoch:18 | 0.116 - 0.709
# epoch:19 | 0.117 - 0.677
# No handles with labels found to put in legend.
# ============== 14/16 ==============
# epoch:0 | 0.093 - 0.097
# epoch:1 | 0.116 - 0.213
# epoch:2 | 0.116 - 0.468
# epoch:3 | 0.116 - 0.561
# epoch:4 | 0.116 - 0.59
# epoch:5 | 0.116 - 0.589
# epoch:6 | 0.116 - 0.495
# epoch:7 | 0.116 - 0.567
# epoch:8 | 0.116 - 0.605
# epoch:9 | 0.116 - 0.611
# epoch:10 | 0.116 - 0.611
# epoch:11 | 0.116 - 0.599
# epoch:12 | 0.116 - 0.614
# epoch:13 | 0.116 - 0.617
# epoch:14 | 0.116 - 0.614
# epoch:15 | 0.117 - 0.616
# epoch:16 | 0.117 - 0.615
# epoch:17 | 0.117 - 0.616
# epoch:18 | 0.116 - 0.615
# epoch:19 | 0.117 - 0.617
# No handles with labels found to put in legend.
# ============== 15/16 ==============
# epoch:0 | 0.116 - 0.098
# epoch:1 | 0.116 - 0.355
# epoch:2 | 0.116 - 0.362
# epoch:3 | 0.116 - 0.407
# epoch:4 | 0.117 - 0.45
# epoch:5 | 0.117 - 0.483
# epoch:6 | 0.117 - 0.503
# epoch:7 | 0.117 - 0.504
# epoch:8 | 0.117 - 0.507
# epoch:9 | 0.117 - 0.488
# epoch:10 | 0.117 - 0.512
# epoch:11 | 0.117 - 0.51
# epoch:12 | 0.117 - 0.514
# epoch:13 | 0.117 - 0.521
# epoch:14 | 0.116 - 0.501
# epoch:15 | 0.116 - 0.541
# epoch:16 | 0.116 - 0.58
# epoch:17 | 0.116 - 0.537
# epoch:18 | 0.116 - 0.602
# epoch:19 | 0.116 - 0.534
# No handles with labels found to put in legend.
# ============== 16/16 ==============
# epoch:0 | 0.116 - 0.143
# epoch:1 | 0.116 - 0.214
# epoch:2 | 0.117 - 0.383
# epoch:3 | 0.117 - 0.317
# epoch:4 | 0.117 - 0.321
# epoch:5 | 0.117 - 0.321
# epoch:6 | 0.116 - 0.325
# epoch:7 | 0.117 - 0.325
# epoch:8 | 0.116 - 0.323
# epoch:9 | 0.117 - 0.417
# epoch:10 | 0.117 - 0.425
# epoch:11 | 0.117 - 0.429
# epoch:12 | 0.117 - 0.43
# epoch:13 | 0.117 - 0.421
# epoch:14 | 0.117 - 0.416
# epoch:15 | 0.117 - 0.425
# epoch:16 | 0.117 - 0.421
# epoch:17 | 0.117 - 0.414
# epoch:18 | 0.117 - 0.416
# epoch:19 | 0.117 - 0.423
# villagecBookPro:chap6 kosuke$ pwd
# /Users/kosuke/dev/scratch_deep_learning/chap6
# villagecBookPro:chap6 kosuke$ ls
# batch_norm_test.png                     optimizer_compare_mnist.py
# batch_norm_test.py                      weight_init_activation_histogram.py
# chap6.md                                weight_init_compare.png
# optimizer_compare_mnist.png             weight_init_compare.py
# villagecBookPro:chap6 kosuke$ touch overfit_weight_decay.py
# villagecBookPro:chap6 kosuke$ python3 overfit_weight_decay.py
# epoch:0, train acc:0.14333333333333334, test acc:0.1394
# epoch:1, train acc:0.18666666666666668, test acc:0.1482
# epoch:2, train acc:0.19666666666666666, test acc:0.1508
# epoch:3, train acc:0.22, test acc:0.1569
# epoch:4, train acc:0.25, test acc:0.1703
# epoch:5, train acc:0.2633333333333333, test acc:0.1782
# epoch:6, train acc:0.30333333333333334, test acc:0.1993
# epoch:7, train acc:0.3333333333333333, test acc:0.2163
# epoch:8, train acc:0.3933333333333333, test acc:0.2423
# epoch:9, train acc:0.42333333333333334, test acc:0.2641
# epoch:10, train acc:0.4633333333333333, test acc:0.2886
# epoch:11, train acc:0.4866666666666667, test acc:0.2975
# epoch:12, train acc:0.51, test acc:0.3181
# epoch:13, train acc:0.5433333333333333, test acc:0.3416
# epoch:14, train acc:0.5733333333333334, test acc:0.3625
# epoch:15, train acc:0.5633333333333334, test acc:0.3718
# epoch:16, train acc:0.5566666666666666, test acc:0.3758
# epoch:17, train acc:0.5766666666666667, test acc:0.3858
# epoch:18, train acc:0.59, test acc:0.391
# epoch:19, train acc:0.59, test acc:0.3958
# epoch:20, train acc:0.59, test acc:0.404
# epoch:21, train acc:0.5833333333333334, test acc:0.4153
# epoch:22, train acc:0.5933333333333334, test acc:0.4254
# epoch:23, train acc:0.6066666666666667, test acc:0.4369
# epoch:24, train acc:0.6, test acc:0.4354
# epoch:25, train acc:0.6066666666666667, test acc:0.4384
# epoch:26, train acc:0.6066666666666667, test acc:0.4512
# epoch:27, train acc:0.62, test acc:0.4644
# epoch:28, train acc:0.6033333333333334, test acc:0.4647
# epoch:29, train acc:0.6133333333333333, test acc:0.4629
# epoch:30, train acc:0.6166666666666667, test acc:0.4729
# epoch:31, train acc:0.6233333333333333, test acc:0.475
# epoch:32, train acc:0.59, test acc:0.4802
# epoch:33, train acc:0.62, test acc:0.4832
# epoch:34, train acc:0.6133333333333333, test acc:0.4901
# epoch:35, train acc:0.6366666666666667, test acc:0.4949
# epoch:36, train acc:0.6333333333333333, test acc:0.4985
# epoch:37, train acc:0.6233333333333333, test acc:0.4934
# epoch:38, train acc:0.6533333333333333, test acc:0.5094
# epoch:39, train acc:0.6433333333333333, test acc:0.5139
# epoch:40, train acc:0.66, test acc:0.5207
# epoch:41, train acc:0.66, test acc:0.526
# epoch:42, train acc:0.68, test acc:0.5361
# epoch:43, train acc:0.69, test acc:0.538
# epoch:44, train acc:0.6533333333333333, test acc:0.5185
# epoch:45, train acc:0.67, test acc:0.5278
# epoch:46, train acc:0.7066666666666667, test acc:0.5485
# epoch:47, train acc:0.7266666666666667, test acc:0.5564
# epoch:48, train acc:0.7066666666666667, test acc:0.5453
# epoch:49, train acc:0.75, test acc:0.5637
# epoch:50, train acc:0.76, test acc:0.5725
# epoch:51, train acc:0.7766666666666666, test acc:0.5813
# epoch:52, train acc:0.7833333333333333, test acc:0.5883
# epoch:53, train acc:0.7766666666666666, test acc:0.5957
# epoch:54, train acc:0.7633333333333333, test acc:0.5844
# epoch:55, train acc:0.7766666666666666, test acc:0.5941
# epoch:56, train acc:0.7933333333333333, test acc:0.6036
# epoch:57, train acc:0.8066666666666666, test acc:0.6083
# epoch:58, train acc:0.8033333333333333, test acc:0.6029
# epoch:59, train acc:0.7933333333333333, test acc:0.5958
# epoch:60, train acc:0.8133333333333334, test acc:0.6009
# epoch:61, train acc:0.79, test acc:0.5917
# epoch:62, train acc:0.8033333333333333, test acc:0.6025
# epoch:63, train acc:0.8066666666666666, test acc:0.6095
# epoch:64, train acc:0.8066666666666666, test acc:0.6019
# epoch:65, train acc:0.8133333333333334, test acc:0.5885
# epoch:66, train acc:0.8366666666666667, test acc:0.6102
# epoch:67, train acc:0.8366666666666667, test acc:0.6146
# epoch:68, train acc:0.8366666666666667, test acc:0.6133
# epoch:69, train acc:0.85, test acc:0.6186
# epoch:70, train acc:0.8333333333333334, test acc:0.6131
# epoch:71, train acc:0.8466666666666667, test acc:0.6289
# epoch:72, train acc:0.8466666666666667, test acc:0.6225
# epoch:73, train acc:0.8533333333333334, test acc:0.6266
# epoch:74, train acc:0.8533333333333334, test acc:0.6356
# epoch:75, train acc:0.8633333333333333, test acc:0.6353
# epoch:76, train acc:0.8633333333333333, test acc:0.628
# epoch:77, train acc:0.8466666666666667, test acc:0.6272
# epoch:78, train acc:0.85, test acc:0.6311
# epoch:79, train acc:0.8733333333333333, test acc:0.6473
# epoch:80, train acc:0.87, test acc:0.645
# epoch:81, train acc:0.8666666666666667, test acc:0.6358
# epoch:82, train acc:0.8633333333333333, test acc:0.6387
# epoch:83, train acc:0.85, test acc:0.6369
# epoch:84, train acc:0.86, test acc:0.6388
# epoch:85, train acc:0.87, test acc:0.6446
# epoch:86, train acc:0.86, test acc:0.6438
# epoch:87, train acc:0.8733333333333333, test acc:0.6498
# epoch:88, train acc:0.8766666666666667, test acc:0.6422
# epoch:89, train acc:0.87, test acc:0.6437
# epoch:90, train acc:0.8666666666666667, test acc:0.6525
# epoch:91, train acc:0.88, test acc:0.6516
# epoch:92, train acc:0.8933333333333333, test acc:0.6573
# epoch:93, train acc:0.8733333333333333, test acc:0.6518
# epoch:94, train acc:0.88, test acc:0.6571
# epoch:95, train acc:0.8766666666666667, test acc:0.652
# epoch:96, train acc:0.9066666666666666, test acc:0.664
# epoch:97, train acc:0.88, test acc:0.649
# epoch:98, train acc:0.8933333333333333, test acc:0.6613
# epoch:99, train acc:0.9033333333333333, test acc:0.6675
# epoch:100, train acc:0.8966666666666666, test acc:0.6657
# epoch:101, train acc:0.9, test acc:0.6634
# epoch:102, train acc:0.89, test acc:0.6713
# epoch:103, train acc:0.9, test acc:0.667
# epoch:104, train acc:0.9066666666666666, test acc:0.6706
# epoch:105, train acc:0.8933333333333333, test acc:0.656
# epoch:106, train acc:0.8833333333333333, test acc:0.654
# epoch:107, train acc:0.8866666666666667, test acc:0.6646
# epoch:108, train acc:0.9033333333333333, test acc:0.668
# epoch:109, train acc:0.9, test acc:0.6623
# epoch:110, train acc:0.9166666666666666, test acc:0.671
# epoch:111, train acc:0.8933333333333333, test acc:0.6612
# epoch:112, train acc:0.8933333333333333, test acc:0.6639
# epoch:113, train acc:0.9, test acc:0.6645
# epoch:114, train acc:0.91, test acc:0.6695
# epoch:115, train acc:0.9033333333333333, test acc:0.6621
# epoch:116, train acc:0.9033333333333333, test acc:0.6663
# epoch:117, train acc:0.9133333333333333, test acc:0.677
# epoch:118, train acc:0.9133333333333333, test acc:0.6769
# epoch:119, train acc:0.9033333333333333, test acc:0.674
# epoch:120, train acc:0.9066666666666666, test acc:0.6728
# epoch:121, train acc:0.9033333333333333, test acc:0.6686
# epoch:122, train acc:0.88, test acc:0.6507
# epoch:123, train acc:0.9133333333333333, test acc:0.6764
# epoch:124, train acc:0.9, test acc:0.6733
# epoch:125, train acc:0.91, test acc:0.6792
# epoch:126, train acc:0.9166666666666666, test acc:0.6801
# epoch:127, train acc:0.9, test acc:0.6808
# epoch:128, train acc:0.9, test acc:0.6806
# epoch:129, train acc:0.91, test acc:0.6838
# epoch:130, train acc:0.9033333333333333, test acc:0.6743
# epoch:131, train acc:0.91, test acc:0.6799
# epoch:132, train acc:0.91, test acc:0.6797
# epoch:133, train acc:0.9033333333333333, test acc:0.6725
# epoch:134, train acc:0.9066666666666666, test acc:0.6722
# epoch:135, train acc:0.8966666666666666, test acc:0.6819
# epoch:136, train acc:0.9066666666666666, test acc:0.668
# epoch:137, train acc:0.89, test acc:0.6715
# epoch:138, train acc:0.9166666666666666, test acc:0.6816
# epoch:139, train acc:0.9133333333333333, test acc:0.6791
# epoch:140, train acc:0.9066666666666666, test acc:0.6843
# epoch:141, train acc:0.9133333333333333, test acc:0.6838
# epoch:142, train acc:0.9033333333333333, test acc:0.6676
# epoch:143, train acc:0.91, test acc:0.6766
# epoch:144, train acc:0.8966666666666666, test acc:0.676
# epoch:145, train acc:0.9, test acc:0.6843
# epoch:146, train acc:0.9066666666666666, test acc:0.69
# epoch:147, train acc:0.92, test acc:0.6822
# epoch:148, train acc:0.92, test acc:0.6847
# epoch:149, train acc:0.9133333333333333, test acc:0.6866
# epoch:150, train acc:0.8966666666666666, test acc:0.6811
# epoch:151, train acc:0.9133333333333333, test acc:0.6872
# epoch:152, train acc:0.91, test acc:0.6781
# epoch:153, train acc:0.9133333333333333, test acc:0.6766
# epoch:154, train acc:0.9133333333333333, test acc:0.6887
# epoch:155, train acc:0.9066666666666666, test acc:0.6716
# epoch:156, train acc:0.9166666666666666, test acc:0.6914
# epoch:157, train acc:0.9033333333333333, test acc:0.6882
# epoch:158, train acc:0.9066666666666666, test acc:0.6882
# epoch:159, train acc:0.92, test acc:0.6831
# epoch:160, train acc:0.9233333333333333, test acc:0.6905
# epoch:161, train acc:0.91, test acc:0.6866
# epoch:162, train acc:0.92, test acc:0.6852
# epoch:163, train acc:0.9166666666666666, test acc:0.6867
# epoch:164, train acc:0.9233333333333333, test acc:0.6845
# epoch:165, train acc:0.9133333333333333, test acc:0.683
# epoch:166, train acc:0.92, test acc:0.6872
# epoch:167, train acc:0.9166666666666666, test acc:0.6866
# epoch:168, train acc:0.9266666666666666, test acc:0.6912
# epoch:169, train acc:0.92, test acc:0.6902
# epoch:170, train acc:0.9233333333333333, test acc:0.6824
# epoch:171, train acc:0.9033333333333333, test acc:0.6856
# epoch:172, train acc:0.9066666666666666, test acc:0.6983
# epoch:173, train acc:0.92, test acc:0.6829
# epoch:174, train acc:0.9133333333333333, test acc:0.6911
# epoch:175, train acc:0.9266666666666666, test acc:0.6883
# epoch:176, train acc:0.9233333333333333, test acc:0.6859
# epoch:177, train acc:0.9233333333333333, test acc:0.6844
# epoch:178, train acc:0.92, test acc:0.6775
# epoch:179, train acc:0.92, test acc:0.6888
# epoch:180, train acc:0.92, test acc:0.6801
# epoch:181, train acc:0.9166666666666666, test acc:0.6901
# epoch:182, train acc:0.9166666666666666, test acc:0.6946
# epoch:183, train acc:0.9166666666666666, test acc:0.6989
# epoch:184, train acc:0.91, test acc:0.692
# epoch:185, train acc:0.9166666666666666, test acc:0.6905
# epoch:186, train acc:0.9166666666666666, test acc:0.6859
# epoch:187, train acc:0.91, test acc:0.6882
# epoch:188, train acc:0.91, test acc:0.6897
# epoch:189, train acc:0.91, test acc:0.6876
# epoch:190, train acc:0.9, test acc:0.701
# epoch:191, train acc:0.92, test acc:0.6865
# epoch:192, train acc:0.92, test acc:0.6918
# epoch:193, train acc:0.9166666666666666, test acc:0.6946
# epoch:194, train acc:0.92, test acc:0.6947
# epoch:195, train acc:0.91, test acc:0.6881
# epoch:196, train acc:0.9133333333333333, test acc:0.6922
# epoch:197, train acc:0.9166666666666666, test acc:0.6834
# epoch:198, train acc:0.9066666666666666, test acc:0.6872
# epoch:199, train acc:0.91, test acc:0.6825
# epoch:200, train acc:0.9166666666666666, test acc:0.6903
