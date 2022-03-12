import os
import sys

sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.util import smooth_curve
from common.multi_layer_net import MultiLayerNet
from common.optimizer import *


# 0:MNISTデータの読み込み==========
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

train_size = x_train.shape[0]
batch_size = 128
max_iterations = 2000


# 1:実験の設定==========
optimizers = {}
optimizers["SGD"] = SGD()
optimizers["Momentum"] = Momentum()
optimizers["AdaGrad"] = AdaGrad()
optimizers["Adam"] = Adam()
# optimizers['RMSprop'] = RMSprop()

networks = {}
train_loss = {}
for key in optimizers.keys():
    networks[key] = MultiLayerNet(
        input_size=784, hidden_size_list=[100, 100, 100, 100], output_size=10
    )
    train_loss[key] = []


# 2:訓練の開始==========
for i in range(max_iterations):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    for key in optimizers.keys():
        grads = networks[key].gradient(x_batch, t_batch)
        optimizers[key].update(networks[key].params, grads)

        loss = networks[key].loss(x_batch, t_batch)
        train_loss[key].append(loss)

    if i % 100 == 0:
        print("===========" + "iteration:" + str(i) + "===========")
        for key in optimizers.keys():
            loss = networks[key].loss(x_batch, t_batch)
            print(key + ":" + str(loss))


# 3.グラフの描画==========
markers = {"SGD": "o", "Momentum": "x", "AdaGrad": "s", "Adam": "D"}
x = np.arange(max_iterations)
for key in optimizers.keys():
    plt.plot(
        x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key
    )
plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0, 1)
plt.legend()
plt.show()

# ===========iteration:0===========
# SGD:2.4483175424871932
# Momentum:2.381339435133192
# AdaGrad:2.2763313133585363
# Adam:2.2242019925528806
# ===========iteration:100===========
# SGD:1.500460833484958
# Momentum:0.3473705913003963
# AdaGrad:0.13432086995239784
# Adam:0.28506828581044996
# ===========iteration:200===========
# SGD:0.667610580907859
# Momentum:0.15414537566289327
# AdaGrad:0.06085647111886553
# Adam:0.12872632773373865
# ===========iteration:300===========
# SGD:0.5363544771105576
# Momentum:0.2748927498026504
# AdaGrad:0.09674992773889177
# Adam:0.1787393563747794
# ===========iteration:400===========
# SGD:0.48209161490343416
# Momentum:0.2506238191605469
# AdaGrad:0.07624044896598185
# Adam:0.17958806857111478
# ===========iteration:500===========
# SGD:0.3599153089139416
# Momentum:0.13144221488968952
# AdaGrad:0.047973708990407274
# Adam:0.13483280994343327
# ===========iteration:600===========
# SGD:0.3283423895386636
# Momentum:0.15915576184455338
# AdaGrad:0.06266145148522254
# Adam:0.13486408462816565
# ===========iteration:700===========
# SGD:0.3473716774914608
# Momentum:0.15362421523955871
# AdaGrad:0.06103993367301685
# Adam:0.07453203055663792
# ===========iteration:800===========
# SGD:0.23263908824638818
# Momentum:0.07170182492885883
# AdaGrad:0.024244612923562184
# Adam:0.026793195668455137
# ===========iteration:900===========
# SGD:0.2220693616744257
# Momentum:0.0402829961863734
# AdaGrad:0.030801136906260525
# Adam:0.036553910701602334
# ===========iteration:1000===========
# SGD:0.19254805049441176
# Momentum:0.06406362783682171
# AdaGrad:0.02258250362642748
# Adam:0.029800074196796582
# ===========iteration:1100===========
# SGD:0.3340497501576948
# Momentum:0.11117516640870949
# AdaGrad:0.056980218268682825
# Adam:0.11710992603804034
# ===========iteration:1200===========
# SGD:0.26206849896172185
# Momentum:0.08182446198028805
# AdaGrad:0.024205957985127237
# Adam:0.020808299428983136
# ===========iteration:1300===========
# SGD:0.13229851246930266
# Momentum:0.0437140657306226
# AdaGrad:0.022488310750628862
# Adam:0.07125799369444293
# ===========iteration:1400===========
# SGD:0.36854027632204434
# Momentum:0.26271404681299987
# AdaGrad:0.11204057334397732
# Adam:0.22683260033392022
# ===========iteration:1500===========
# SGD:0.30193707689777516
# Momentum:0.05123251569456069
# AdaGrad:0.04257449692091829
# Adam:0.05752309276009822
# ===========iteration:1600===========
# SGD:0.18910948865633032
# Momentum:0.04565657150663849
# AdaGrad:0.042397912871651816
# Adam:0.032920986807239976
# ===========iteration:1700===========
# SGD:0.1971846402413075
# Momentum:0.08743416446588084
# AdaGrad:0.019797103653745544
# Adam:0.05108840668729184
# ===========iteration:1800===========
# SGD:0.22622096414044515
# Momentum:0.04431619665466372
# AdaGrad:0.01790822619448814
# Adam:0.013401154221139841
# ===========iteration:1900===========
# SGD:0.1593149070177167
# Momentum:0.0547744226369022
# AdaGrad:0.03074463385209847
# Adam:0.02930327380764089
