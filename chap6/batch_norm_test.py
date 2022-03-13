import sys, os

sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.optimizer import SGD, Adam

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 学習データを削減
x_train = x_train[:1000]
t_train = t_train[:1000]

max_epochs = 20
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.01


def __train(weight_init_std):
    bn_network = MultiLayerNetExtend(
        input_size=784,
        hidden_size_list=[100, 100, 100, 100, 100],
        output_size=10,
        weight_init_std=weight_init_std,
        use_batchnorm=True,
    )
    network = MultiLayerNetExtend(
        input_size=784,
        hidden_size_list=[100, 100, 100, 100, 100],
        output_size=10,
        weight_init_std=weight_init_std,
    )
    optimizer = SGD(lr=learning_rate)

    train_acc_list = []
    bn_train_acc_list = []

    iter_per_epoch = max(train_size / batch_size, 1)
    epoch_cnt = 0

    for i in range(1000000000):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        for _network in (bn_network, network):
            grads = _network.gradient(x_batch, t_batch)
            optimizer.update(_network.params, grads)

        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            bn_train_acc = bn_network.accuracy(x_train, t_train)
            train_acc_list.append(train_acc)
            bn_train_acc_list.append(bn_train_acc)

            print(
                "epoch:"
                + str(epoch_cnt)
                + " | "
                + str(train_acc)
                + " - "
                + str(bn_train_acc)
            )

            epoch_cnt += 1
            if epoch_cnt >= max_epochs:
                break

    return train_acc_list, bn_train_acc_list


# 3.グラフの描画==========
weight_scale_list = np.logspace(0, -4, num=16)
x = np.arange(max_epochs)

for i, w in enumerate(weight_scale_list):
    print("============== " + str(i + 1) + "/16" + " ==============")
    train_acc_list, bn_train_acc_list = __train(w)

    plt.subplot(4, 4, i + 1)
    plt.title("W:" + str(w)[0:7])
    if i == 15:
        plt.plot(x, bn_train_acc_list, label="Batch Normalization", markevery=2)
        plt.plot(
            x,
            train_acc_list,
            linestyle="--",
            label="Normal(without BatchNorm)",
            markevery=2,
        )
    else:
        plt.plot(x, bn_train_acc_list, markevery=2)
        plt.plot(x, train_acc_list, linestyle="--", markevery=2)

    plt.ylim(0, 1.0)
    if i % 4:
        plt.yticks([])
    else:
        plt.ylabel("accuracy")
    if i < 12:
        plt.xticks([])
    else:
        plt.xlabel("epochs")
    plt.legend(loc="lower right")

plt.show()

# ============= 1/16 ==============
# epoch:0 | 0.117 - 0.117
# ../common/multi_layer_net_extend.py:122: RuntimeWarning: overflow encountered in square
#   weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)
# ../common/multi_layer_net_extend.py:122: RuntimeWarning: invalid value encountered in double_scalars
#   weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)
# ../common/functions.py:32: RuntimeWarning: invalid value encountered in subtract
#   x = x - np.max(x, axis=-1, keepdims=True)  # オーバーフロー対策
# ../common/layers.py:12: RuntimeWarning: invalid value encountered in less_equal
#   self.mask = x <= 0
# epoch:1 | 0.097 - 0.113
# epoch:2 | 0.097 - 0.147
# epoch:3 | 0.097 - 0.183
# epoch:4 | 0.097 - 0.208
# epoch:5 | 0.097 - 0.229
# epoch:6 | 0.097 - 0.243
# epoch:7 | 0.097 - 0.272
# epoch:8 | 0.097 - 0.304
# epoch:9 | 0.097 - 0.312
# epoch:10 | 0.097 - 0.323
# epoch:11 | 0.097 - 0.337
# epoch:12 | 0.097 - 0.347
# epoch:13 | 0.097 - 0.363
# epoch:14 | 0.097 - 0.379
# epoch:15 | 0.097 - 0.392
# epoch:16 | 0.097 - 0.404
# epoch:17 | 0.097 - 0.413
# epoch:18 | 0.097 - 0.435
# epoch:19 | 0.097 - 0.433
# No handles with labels found to put in legend.
# ============== 2/16 ==============
# epoch:0 | 0.097 - 0.067
# epoch:1 | 0.097 - 0.093
# epoch:2 | 0.097 - 0.155
# epoch:3 | 0.097 - 0.169
# epoch:4 | 0.097 - 0.193
# epoch:5 | 0.097 - 0.213
# epoch:6 | 0.097 - 0.239
# epoch:7 | 0.097 - 0.254
# epoch:8 | 0.097 - 0.285
# epoch:9 | 0.097 - 0.314
# epoch:10 | 0.097 - 0.324
# epoch:11 | 0.097 - 0.34
# epoch:12 | 0.097 - 0.365
# epoch:13 | 0.097 - 0.369
# epoch:14 | 0.097 - 0.385
# epoch:15 | 0.097 - 0.405
# epoch:16 | 0.097 - 0.418
# epoch:17 | 0.097 - 0.443
# epoch:18 | 0.097 - 0.458
# epoch:19 | 0.097 - 0.464
# No handles with labels found to put in legend.
# ============== 3/16 ==============
# epoch:0 | 0.093 - 0.096
# epoch:1 | 0.328 - 0.113
# epoch:2 | 0.469 - 0.128
# epoch:3 | 0.546 - 0.162
# epoch:4 | 0.618 - 0.197
# epoch:5 | 0.681 - 0.224
# epoch:6 | 0.71 - 0.269
# epoch:7 | 0.749 - 0.308
# epoch:8 | 0.77 - 0.327
# epoch:9 | 0.814 - 0.369
# epoch:10 | 0.839 - 0.401
# epoch:11 | 0.868 - 0.436
# epoch:12 | 0.866 - 0.463
# epoch:13 | 0.892 - 0.489
# epoch:14 | 0.917 - 0.509
# epoch:15 | 0.926 - 0.53
# epoch:16 | 0.931 - 0.558
# epoch:17 | 0.941 - 0.582
# epoch:18 | 0.941 - 0.604
# epoch:19 | 0.951 - 0.613
# No handles with labels found to put in legend.
# ============== 4/16 ==============
# epoch:0 | 0.064 - 0.1
# epoch:1 | 0.26 - 0.114
# epoch:2 | 0.403 - 0.149
# epoch:3 | 0.503 - 0.258
# epoch:4 | 0.569 - 0.323
# epoch:5 | 0.62 - 0.39
# epoch:6 | 0.667 - 0.449
# epoch:7 | 0.7 - 0.504
# epoch:8 | 0.712 - 0.546
# epoch:9 | 0.722 - 0.583
# epoch:10 | 0.751 - 0.613
# epoch:11 | 0.763 - 0.645
# epoch:12 | 0.794 - 0.663
# epoch:13 | 0.799 - 0.682
# epoch:14 | 0.818 - 0.701
# epoch:15 | 0.817 - 0.71
# epoch:16 | 0.832 - 0.72
# epoch:17 | 0.837 - 0.74
# epoch:18 | 0.843 - 0.746
# epoch:19 | 0.855 - 0.762
# No handles with labels found to put in legend.
# ============== 5/16 ==============
# epoch:0 | 0.06 - 0.104
# epoch:1 | 0.07 - 0.141
# epoch:2 | 0.083 - 0.295
# epoch:3 | 0.092 - 0.42
# epoch:4 | 0.117 - 0.515
# epoch:5 | 0.134 - 0.602
# epoch:6 | 0.15 - 0.642
# epoch:7 | 0.165 - 0.689
# epoch:8 | 0.168 - 0.723
# epoch:9 | 0.166 - 0.736
# epoch:10 | 0.191 - 0.764
# epoch:11 | 0.195 - 0.789
# epoch:12 | 0.197 - 0.795
# epoch:13 | 0.224 - 0.801
# epoch:14 | 0.235 - 0.823
# epoch:15 | 0.247 - 0.832
# epoch:16 | 0.257 - 0.844
# epoch:17 | 0.269 - 0.852
# epoch:18 | 0.295 - 0.856
# epoch:19 | 0.334 - 0.863
# No handles with labels found to put in legend.
# ============== 6/16 ==============
# epoch:0 | 0.079 - 0.101
# epoch:1 | 0.097 - 0.207
# epoch:2 | 0.129 - 0.452
# epoch:3 | 0.142 - 0.601
# epoch:4 | 0.117 - 0.675
# epoch:5 | 0.118 - 0.733
# epoch:6 | 0.116 - 0.763
# epoch:7 | 0.156 - 0.787
# epoch:8 | 0.13 - 0.81
# epoch:9 | 0.128 - 0.834
# epoch:10 | 0.118 - 0.842
# epoch:11 | 0.117 - 0.858
# epoch:12 | 0.117 - 0.877
# epoch:13 | 0.117 - 0.888
# epoch:14 | 0.117 - 0.897
# epoch:15 | 0.179 - 0.901
# epoch:16 | 0.116 - 0.906
# epoch:17 | 0.116 - 0.918
# epoch:18 | 0.116 - 0.925
# epoch:19 | 0.116 - 0.928
# No handles with labels found to put in legend.
# ============== 7/16 ==============
# epoch:0 | 0.099 - 0.117
# epoch:1 | 0.105 - 0.331
# epoch:2 | 0.105 - 0.541
# epoch:3 | 0.117 - 0.654
# epoch:4 | 0.117 - 0.716
# epoch:5 | 0.105 - 0.765
# epoch:6 | 0.105 - 0.81
# epoch:7 | 0.116 - 0.858
# epoch:8 | 0.116 - 0.877
# epoch:9 | 0.116 - 0.905
# epoch:10 | 0.116 - 0.933
# epoch:11 | 0.116 - 0.937
# epoch:12 | 0.116 - 0.943
# epoch:13 | 0.116 - 0.955
# epoch:14 | 0.116 - 0.965
# epoch:15 | 0.116 - 0.975
# epoch:16 | 0.116 - 0.977
# epoch:17 | 0.116 - 0.985
# epoch:18 | 0.116 - 0.988
# epoch:19 | 0.116 - 0.99
# No handles with labels found to put in legend.
# ============== 8/16 ==============
# epoch:0 | 0.1 - 0.094
# epoch:1 | 0.117 - 0.275
# epoch:2 | 0.117 - 0.554
# epoch:3 | 0.117 - 0.685
# epoch:4 | 0.117 - 0.75
# epoch:5 | 0.117 - 0.809
# epoch:6 | 0.117 - 0.859
# epoch:7 | 0.117 - 0.911
# epoch:8 | 0.117 - 0.947
# epoch:9 | 0.117 - 0.963
# epoch:10 | 0.117 - 0.977
# epoch:11 | 0.117 - 0.985
# epoch:12 | 0.117 - 0.99
# epoch:13 | 0.117 - 0.991
# epoch:14 | 0.117 - 0.991
# epoch:15 | 0.117 - 0.995
# epoch:16 | 0.117 - 0.996
# epoch:17 | 0.117 - 0.996
# epoch:18 | 0.117 - 0.997
# epoch:19 | 0.117 - 0.999
# No handles with labels found to put in legend.
# ============== 9/16 ==============
# epoch:0 | 0.105 - 0.137
# epoch:1 | 0.117 - 0.422
# epoch:2 | 0.117 - 0.779
# epoch:3 | 0.116 - 0.832
# epoch:4 | 0.116 - 0.871
# epoch:5 | 0.116 - 0.913
# epoch:6 | 0.116 - 0.949
# epoch:7 | 0.117 - 0.964
# epoch:8 | 0.117 - 0.986
# epoch:9 | 0.117 - 0.991
# epoch:10 | 0.117 - 0.991
# epoch:11 | 0.117 - 0.994
# epoch:12 | 0.116 - 0.995
# epoch:13 | 0.116 - 0.997
# epoch:14 | 0.117 - 0.997
# epoch:15 | 0.117 - 0.999
# epoch:16 | 0.116 - 0.999
# epoch:17 | 0.117 - 0.999
# epoch:18 | 0.116 - 0.999
# epoch:19 | 0.116 - 0.999
# No handles with labels found to put in legend.
# ============== 10/16 ==============
# epoch:0 | 0.116 - 0.144
# epoch:1 | 0.117 - 0.51
# epoch:2 | 0.117 - 0.771
# epoch:3 | 0.117 - 0.803
# epoch:4 | 0.117 - 0.849
# epoch:5 | 0.117 - 0.881
# epoch:6 | 0.117 - 0.887
# epoch:7 | 0.117 - 0.968
# epoch:8 | 0.117 - 0.981
# epoch:9 | 0.117 - 0.985
# epoch:10 | 0.117 - 0.935
# epoch:11 | 0.117 - 0.986
# epoch:12 | 0.117 - 0.995
# epoch:13 | 0.117 - 0.999
# epoch:14 | 0.117 - 0.999
# epoch:15 | 0.117 - 0.984
# epoch:16 | 0.117 - 0.981
# epoch:17 | 0.117 - 1.0
# epoch:18 | 0.117 - 0.968
# epoch:19 | 0.117 - 1.0
# No handles with labels found to put in legend.
# ============== 11/16 ==============
# epoch:0 | 0.094 - 0.21
# epoch:1 | 0.117 - 0.471
# epoch:2 | 0.117 - 0.709
# epoch:3 | 0.116 - 0.724
# epoch:4 | 0.116 - 0.768
# epoch:5 | 0.116 - 0.785
# epoch:6 | 0.117 - 0.779
# epoch:7 | 0.117 - 0.794
# epoch:8 | 0.117 - 0.792
# epoch:9 | 0.117 - 0.851
# epoch:10 | 0.117 - 0.888
# epoch:11 | 0.117 - 0.947
# epoch:12 | 0.117 - 0.944
# epoch:13 | 0.117 - 0.975
# epoch:14 | 0.117 - 0.944
# epoch:15 | 0.117 - 0.968
# epoch:16 | 0.117 - 0.974
# epoch:17 | 0.117 - 0.988
# epoch:18 | 0.117 - 0.989
# epoch:19 | 0.117 - 0.991
# No handles with labels found to put in legend.
# ============== 12/16 ==============
# epoch:0 | 0.105 - 0.1
# epoch:1 | 0.116 - 0.513
# epoch:2 | 0.116 - 0.4
# epoch:3 | 0.116 - 0.57
# epoch:4 | 0.116 - 0.725
# epoch:5 | 0.116 - 0.776
# epoch:6 | 0.116 - 0.78
# epoch:7 | 0.116 - 0.787
# epoch:8 | 0.116 - 0.791
# epoch:9 | 0.116 - 0.797
# epoch:10 | 0.116 - 0.757
# epoch:11 | 0.116 - 0.851
# epoch:12 | 0.116 - 0.881
# epoch:13 | 0.116 - 0.888
# epoch:14 | 0.116 - 0.897
# epoch:15 | 0.116 - 0.894
# epoch:16 | 0.116 - 0.901
# epoch:17 | 0.116 - 0.898
# epoch:18 | 0.116 - 0.903
# epoch:19 | 0.116 - 0.889
# No handles with labels found to put in legend.
# ============== 13/16 ==============
# epoch:0 | 0.116 - 0.131
# epoch:1 | 0.105 - 0.312
# epoch:2 | 0.116 - 0.518
# epoch:3 | 0.116 - 0.406
# epoch:4 | 0.117 - 0.589
# epoch:5 | 0.117 - 0.589
# epoch:6 | 0.117 - 0.604
# epoch:7 | 0.117 - 0.618
# epoch:8 | 0.117 - 0.603
# epoch:9 | 0.117 - 0.613
# epoch:10 | 0.117 - 0.623
# epoch:11 | 0.117 - 0.603
# epoch:12 | 0.117 - 0.63
# epoch:13 | 0.117 - 0.706
# epoch:14 | 0.117 - 0.711
# epoch:15 | 0.117 - 0.709
# epoch:16 | 0.117 - 0.767
# epoch:17 | 0.117 - 0.786
# epoch:18 | 0.117 - 0.782
# epoch:19 | 0.116 - 0.806
# No handles with labels found to put in legend.
# ============== 14/16 ==============
# epoch:0 | 0.105 - 0.158
# epoch:1 | 0.117 - 0.456
# epoch:2 | 0.117 - 0.521
# epoch:3 | 0.117 - 0.606
# epoch:4 | 0.117 - 0.636
# epoch:5 | 0.117 - 0.658
# epoch:6 | 0.117 - 0.678
# epoch:7 | 0.117 - 0.677
# epoch:8 | 0.117 - 0.67
# epoch:9 | 0.117 - 0.677
# epoch:10 | 0.117 - 0.7
# epoch:11 | 0.117 - 0.668
# epoch:12 | 0.117 - 0.638
# epoch:13 | 0.117 - 0.673
# epoch:14 | 0.117 - 0.68
# epoch:15 | 0.117 - 0.748
# epoch:16 | 0.117 - 0.775
# epoch:17 | 0.117 - 0.79
# epoch:18 | 0.117 - 0.783
# epoch:19 | 0.117 - 0.756
# No handles with labels found to put in legend.
# ============== 15/16 ==============
# epoch:0 | 0.1 - 0.12
# epoch:1 | 0.116 - 0.31
# epoch:2 | 0.116 - 0.382
# epoch:3 | 0.116 - 0.379
# epoch:4 | 0.117 - 0.443
# epoch:5 | 0.116 - 0.493
# epoch:6 | 0.116 - 0.493
# epoch:7 | 0.116 - 0.45
# epoch:8 | 0.116 - 0.493
# epoch:9 | 0.116 - 0.488
# epoch:10 | 0.116 - 0.485
# epoch:11 | 0.116 - 0.516
# epoch:12 | 0.116 - 0.524
# epoch:13 | 0.116 - 0.522
# epoch:14 | 0.116 - 0.521
# epoch:15 | 0.116 - 0.52
# epoch:16 | 0.116 - 0.517
# epoch:17 | 0.117 - 0.521
# epoch:18 | 0.116 - 0.516
# epoch:19 | 0.117 - 0.514
# No handles with labels found to put in legend.
# ============== 16/16 ==============
# epoch:0 | 0.105 - 0.106
# epoch:1 | 0.117 - 0.216
# epoch:2 | 0.116 - 0.415
# epoch:3 | 0.116 - 0.376
# epoch:4 | 0.116 - 0.418
# epoch:5 | 0.117 - 0.418
# epoch:6 | 0.116 - 0.433
# epoch:7 | 0.117 - 0.509
# epoch:8 | 0.116 - 0.514
# epoch:9 | 0.116 - 0.519
# epoch:10 | 0.116 - 0.524
# epoch:11 | 0.116 - 0.514
# epoch:12 | 0.116 - 0.519
# epoch:13 | 0.116 - 0.525
# epoch:14 | 0.116 - 0.526
# epoch:15 | 0.116 - 0.448
# epoch:16 | 0.116 - 0.512
# epoch:17 | 0.116 - 0.531
# epoch:18 | 0.116 - 0.528
# epoch:19 | 0.116 - 0.531
