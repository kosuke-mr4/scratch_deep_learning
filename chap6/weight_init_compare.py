import os
import sys

sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.util import smooth_curve
from common.multi_layer_net import MultiLayerNet
from common.optimizer import SGD


# 0:MNISTデータの読み込み==========
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

train_size = x_train.shape[0]
batch_size = 128
max_iterations = 2000


# 1:実験の設定==========
weight_init_types = {"std=0.01": 0.01, "Xavier": "sigmoid", "He": "relu"}
optimizer = SGD(lr=0.01)

networks = {}
train_loss = {}
for key, weight_type in weight_init_types.items():
    networks[key] = MultiLayerNet(
        input_size=784,
        hidden_size_list=[100, 100, 100, 100],
        output_size=10,
        weight_init_std=weight_type,
    )
    train_loss[key] = []


# 2:訓練の開始==========
for i in range(max_iterations):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    for key in weight_init_types.keys():
        grads = networks[key].gradient(x_batch, t_batch)
        optimizer.update(networks[key].params, grads)

        loss = networks[key].loss(x_batch, t_batch)
        train_loss[key].append(loss)

    if i % 100 == 0:
        print("===========" + "iteration:" + str(i) + "===========")
        for key in weight_init_types.keys():
            loss = networks[key].loss(x_batch, t_batch)
            print(key + ":" + str(loss))


# 3.グラフの描画==========
markers = {"std=0.01": "o", "Xavier": "s", "He": "D"}
x = np.arange(max_iterations)
for key in weight_init_types.keys():
    plt.plot(
        x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key
    )
plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0, 2.5)
plt.legend()
plt.show()


# ===========iteration:0===========
# std=0.01:2.3024590203817845
# Xavier:2.2844065817746686
# He:2.360434951153027
# ===========iteration:100===========
# std=0.01:2.3028324685162427
# Xavier:2.259042188952213
# He:1.585176696978348
# ===========iteration:200===========
# std=0.01:2.302154623927881
# Xavier:2.1279088840985114
# He:0.8409754372149485
# ===========iteration:300===========
# std=0.01:2.300780170243331
# Xavier:1.832317449966186
# He:0.4653088038083314
# ===========iteration:400===========
# std=0.01:2.3014756990918803
# Xavier:1.3957903062133847
# He:0.5228749012334101
# ===========iteration:500===========
# std=0.01:2.29914175916084
# Xavier:1.007095343531061
# He:0.49697668348880536
# ===========iteration:600===========
# std=0.01:2.2987064509421176
# Xavier:0.6823941937353981
# He:0.34601118785664264
# ===========iteration:700===========
# std=0.01:2.3019453643253494
# Xavier:0.681000052159717
# He:0.35156096444276236
# ===========iteration:800===========
# std=0.01:2.3021297858759233
# Xavier:0.5457044214862969
# He:0.39789108500778314
# ===========iteration:900===========
# std=0.01:2.3027748403390076
# Xavier:0.4473088514754418
# He:0.25834977405508924
# ===========iteration:1000===========
# std=0.01:2.3013404894573393
# Xavier:0.4027470591668288
# He:0.26101158522986945
# ===========iteration:1100===========
# std=0.01:2.2985289881263165
# Xavier:0.38930122025970715
# He:0.2723819847306507
# ===========iteration:1200===========
# std=0.01:2.29801806628259
# Xavier:0.38491669739048073
# He:0.21409365113035295
# ===========iteration:1300===========
# std=0.01:2.301845334227264
# Xavier:0.30797229001152604
# He:0.2116846288148167
# ===========iteration:1400===========
# std=0.01:2.299548681111256
# Xavier:0.24412212479257428
# He:0.1904070706239564
# ===========iteration:1500===========
# std=0.01:2.2963645109081643
# Xavier:0.3395127820564978
# He:0.201741083942515
# ===========iteration:1600===========
# std=0.01:2.3018715098086844
# Xavier:0.2533285511906944
# He:0.19072318272817362
# ===========iteration:1700===========
# std=0.01:2.300276301949439
# Xavier:0.27907632389536097
# He:0.20717300623112456
# ===========iteration:1800===========
# std=0.01:2.3059707547763915
# Xavier:0.37183432541445427
# He:0.25841645089302606
# ===========iteration:1900===========
# std=0.01:2.2944020464103145
# Xavier:0.25571103029945064
# He:0.1508691042776884
