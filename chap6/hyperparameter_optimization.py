import sys, os

sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net import MultiLayerNet
from common.util import shuffle_dataset
from common.trainer import Trainer

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 高速化のため訓練データの削減
x_train = x_train[:500]
t_train = t_train[:500]

# 検証データの分離
validation_rate = 0.20
validation_num = int(x_train.shape[0] * validation_rate)
x_train, t_train = shuffle_dataset(x_train, t_train)
x_val = x_train[:validation_num]
t_val = t_train[:validation_num]
x_train = x_train[validation_num:]
t_train = t_train[validation_num:]


def __train(lr, weight_decay, epocs=50):
    network = MultiLayerNet(
        input_size=784,
        hidden_size_list=[100, 100, 100, 100, 100, 100],
        output_size=10,
        weight_decay_lambda=weight_decay,
    )
    trainer = Trainer(
        network,
        x_train,
        t_train,
        x_val,
        t_val,
        epochs=epocs,
        mini_batch_size=100,
        optimizer="sgd",
        optimizer_param={"lr": lr},
        verbose=False,
    )
    trainer.train()

    return trainer.test_acc_list, trainer.train_acc_list


# ハイパーパラメータのランダム探索======================================
optimization_trial = 100
results_val = {}
results_train = {}
for _ in range(optimization_trial):
    # 探索したハイパーパラメータの範囲を指定===============
    weight_decay = 10 ** np.random.uniform(-8, -4)
    lr = 10 ** np.random.uniform(-6, -2)
    # ================================================

    val_acc_list, train_acc_list = __train(lr, weight_decay)
    print(
        "val acc:"
        + str(val_acc_list[-1])
        + " | lr:"
        + str(lr)
        + ", weight decay:"
        + str(weight_decay)
    )
    key = "lr:" + str(lr) + ", weight decay:" + str(weight_decay)
    results_val[key] = val_acc_list
    results_train[key] = train_acc_list

# グラフの描画========================================================
print("=========== Hyper-Parameter Optimization Result ===========")
graph_draw_num = 20
col_num = 5
row_num = int(np.ceil(graph_draw_num / col_num))
i = 0

for key, val_acc_list in sorted(
    results_val.items(), key=lambda x: x[1][-1], reverse=True
):
    print("Best-" + str(i + 1) + "(val acc:" + str(val_acc_list[-1]) + ") | " + key)

    plt.subplot(row_num, col_num, i + 1)
    plt.title("Best-" + str(i + 1))
    plt.ylim(0.0, 1.0)
    if i % 5:
        plt.yticks([])
    plt.xticks([])
    x = np.arange(len(val_acc_list))
    plt.plot(x, val_acc_list)
    plt.plot(x, results_train[key], "--")
    i += 1

    if i >= graph_draw_num:
        break

plt.show()


# val acc:0.06 | lr:1.2703160733774074e-05, weight decay:2.719860544463644e-05
# val acc:0.04 | lr:1.1878750404093373e-05, weight decay:8.687657883353702e-05
# val acc:0.08 | lr:4.958040313536898e-05, weight decay:1.2660191901876862e-07
# val acc:0.05 | lr:2.414805408947654e-06, weight decay:6.581717780776597e-06
# val acc:0.13 | lr:3.1167510249846624e-06, weight decay:2.986959305071852e-06
# val acc:0.03 | lr:3.9680965247759013e-05, weight decay:7.159288562554531e-08
# val acc:0.08 | lr:0.00021183349941044158, weight decay:9.369802387390463e-05
# val acc:0.13 | lr:2.393243203044951e-06, weight decay:1.0333148643501842e-06
# val acc:0.26 | lr:0.0012550273881050736, weight decay:1.3457383185497738e-05
# val acc:0.14 | lr:1.4670782658556757e-05, weight decay:1.5859946358509303e-08
# val acc:0.09 | lr:0.00012225462938769176, weight decay:1.2677395615047744e-08
# val acc:0.06 | lr:1.161885428723261e-06, weight decay:1.0332341105833665e-08
# val acc:0.12 | lr:1.5707323485891424e-06, weight decay:7.1720947332997046e-06
# val acc:0.19 | lr:4.159945765020933e-06, weight decay:2.4982830493428933e-05
# val acc:0.47 | lr:0.003108690193174507, weight decay:1.472381807925704e-05
# val acc:0.13 | lr:0.0001145975217040866, weight decay:1.4328344536039846e-07
# val acc:0.45 | lr:0.0034309268623154937, weight decay:2.5261852470647776e-08
# val acc:0.1 | lr:3.9166016853684176e-05, weight decay:4.361494576922768e-06
# val acc:0.11 | lr:0.00010706801515090927, weight decay:5.284974901382317e-05
# val acc:0.13 | lr:8.974997996310983e-06, weight decay:2.9358037610123724e-08
# val acc:0.13 | lr:1.1673271102126016e-06, weight decay:5.99442181249917e-08
# val acc:0.09 | lr:6.45020111433972e-05, weight decay:2.1973493821443674e-08
# val acc:0.09 | lr:0.00015213687065159334, weight decay:1.720063566719492e-05
# val acc:0.02 | lr:1.8233453503010576e-05, weight decay:4.776770830057147e-07
# val acc:0.12 | lr:1.5761354533706114e-05, weight decay:2.777532199875723e-07
# val acc:0.1 | lr:4.866467118736708e-05, weight decay:4.1027391183242976e-07
# val acc:0.14 | lr:0.00017006282391273465, weight decay:1.2374504084875435e-05
# val acc:0.17 | lr:5.9862164363843584e-06, weight decay:1.5023476077498931e-05
# val acc:0.22 | lr:0.0008907144096194279, weight decay:2.2111168114259186e-07
# val acc:0.15 | lr:1.2197299858056176e-05, weight decay:3.1225344720439956e-08
# val acc:0.16 | lr:0.0008521048451165013, weight decay:7.456739930549179e-08
# val acc:0.1 | lr:1.153495237568048e-05, weight decay:7.297660809918036e-06
# val acc:0.12 | lr:0.0002703321219301884, weight decay:1.7706589807064095e-08
# val acc:0.11 | lr:1.0471298257641324e-05, weight decay:1.5390103947761543e-06
# val acc:0.05 | lr:0.00012336152438192775, weight decay:5.4831324302927514e-05
# val acc:0.18 | lr:0.00019843126697911056, weight decay:1.0556844251545753e-05
# val acc:0.14 | lr:0.00015540355427298122, weight decay:4.2165438863175425e-08
# val acc:0.14 | lr:3.602673813304217e-06, weight decay:4.449215979862019e-08
# val acc:0.15 | lr:1.0670171978755628e-06, weight decay:7.878280684276965e-06
# val acc:0.1 | lr:1.2973739123574391e-05, weight decay:9.121730997896587e-08
# val acc:0.05 | lr:1.9791653818422926e-05, weight decay:9.604016049318154e-08
# val acc:0.29 | lr:0.0015342692652872002, weight decay:4.1100976474583763e-08
# val acc:0.16 | lr:9.781613970652227e-05, weight decay:1.2906613165065485e-05
# val acc:0.71 | lr:0.0046593335222353625, weight decay:7.06531358012995e-05
# val acc:0.09 | lr:3.43768030272558e-05, weight decay:1.628326835479444e-08
# val acc:0.12 | lr:1.3847534067775887e-05, weight decay:8.992008019477472e-05
# val acc:0.24 | lr:0.002826015242454688, weight decay:2.7710072733447147e-06
# val acc:0.65 | lr:0.004238683875024437, weight decay:4.789331621675631e-07
# val acc:0.13 | lr:3.0730385458479698e-06, weight decay:4.9274304633735e-05
# val acc:0.09 | lr:0.00031841756683063726, weight decay:1.92980211540364e-07
# val acc:0.6 | lr:0.004412585867480103, weight decay:1.8694614516097234e-06
# val acc:0.08 | lr:1.4948616786132383e-06, weight decay:4.331587470518701e-07
# val acc:0.07 | lr:2.810293147282566e-06, weight decay:4.5524431394349117e-05
# val acc:0.12 | lr:4.686983430848136e-06, weight decay:9.545574460537964e-06
# val acc:0.1 | lr:1.2954771704938716e-06, weight decay:1.2372095729447572e-05
# val acc:0.11 | lr:3.234998712916762e-06, weight decay:5.316240716213556e-06
# val acc:0.09 | lr:1.6035936667906105e-06, weight decay:2.6067047414526135e-05
# val acc:0.09 | lr:0.000604358751510169, weight decay:2.186414415338223e-05
# val acc:0.09 | lr:2.3947540899951622e-05, weight decay:2.839983643267178e-06
# val acc:0.21 | lr:0.0015738527626970632, weight decay:1.02082144476431e-08
# val acc:0.09 | lr:3.0475950484227378e-05, weight decay:4.576106445144897e-06
# val acc:0.13 | lr:6.834535216160435e-06, weight decay:1.2823825596272848e-06
# val acc:0.1 | lr:1.3843068636682505e-05, weight decay:7.46546732027143e-08
# val acc:0.35 | lr:0.001889395257136538, weight decay:2.3673684530307035e-08
# val acc:0.2 | lr:0.0011844655027465857, weight decay:5.7789430687754895e-05
# val acc:0.14 | lr:0.00025174466420946204, weight decay:8.481656916508643e-06
# val acc:0.17 | lr:0.0011704797429735589, weight decay:2.3572259902707702e-07
# val acc:0.09 | lr:0.0003985507578159158, weight decay:1.9953090428251417e-07
# val acc:0.18 | lr:0.00020982283200683926, weight decay:5.570134905325748e-05
# val acc:0.78 | lr:0.00592501475408638, weight decay:2.884905840440795e-06
# val acc:0.08 | lr:6.186300846068662e-05, weight decay:2.631762690021811e-08
# val acc:0.07 | lr:0.0005866902203073075, weight decay:6.161713490933972e-06
# val acc:0.36 | lr:0.0029522196861681985, weight decay:1.8318572995694304e-06
# val acc:0.12 | lr:1.2721292042603188e-05, weight decay:5.496965625264351e-07
# val acc:0.58 | lr:0.0037568045548814007, weight decay:8.586635791523419e-07
# val acc:0.11 | lr:2.3923707189416223e-05, weight decay:1.5761227822080893e-08
# val acc:0.06 | lr:0.00010959896327381796, weight decay:4.8080178171327024e-05
# val acc:0.06 | lr:3.4085464301550606e-05, weight decay:5.439582530742481e-07
# val acc:0.33 | lr:0.001595086009717874, weight decay:3.336956730149232e-06
# val acc:0.1 | lr:2.247912161735376e-06, weight decay:1.0129224994072009e-05
# val acc:0.09 | lr:0.00022087384186536563, weight decay:4.6271819681275394e-05
# val acc:0.66 | lr:0.006024290557714003, weight decay:4.130863810219471e-06
# val acc:0.17 | lr:0.0005681417416959209, weight decay:2.462231550200661e-06
# val acc:0.1 | lr:3.138015137958785e-05, weight decay:7.239673708824999e-06
# val acc:0.08 | lr:4.5898925694380734e-05, weight decay:1.0884597222966e-05
# val acc:0.09 | lr:6.760702883046292e-06, weight decay:5.4095741760211437e-05
# val acc:0.18 | lr:0.0011459880910791405, weight decay:2.845905763950432e-07
# val acc:0.1 | lr:3.7407553982808224e-06, weight decay:1.4097827010889647e-07
# val acc:0.09 | lr:0.00039138721828309184, weight decay:1.9409927600050874e-06
# val acc:0.11 | lr:7.3323036538328745e-06, weight decay:2.0785457725314274e-05
# val acc:0.65 | lr:0.003294149174060375, weight decay:7.500473917659556e-05
# val acc:0.12 | lr:0.0002490464693322264, weight decay:2.53155763732841e-06
# val acc:0.14 | lr:7.487844035060888e-05, weight decay:2.9306742495305708e-05
# val acc:0.24 | lr:0.0016693655529397037, weight decay:1.9660530872579584e-05
# val acc:0.17 | lr:5.727133811217092e-05, weight decay:2.8652096955064116e-06
# val acc:0.48 | lr:0.002429521019985163, weight decay:4.4981712769966694e-07
# val acc:0.51 | lr:0.002257959137040223, weight decay:2.1335895666178243e-05
# val acc:0.1 | lr:0.0003129958231509853, weight decay:2.5186492254900706e-05
# val acc:0.06 | lr:2.0775597720690124e-06, weight decay:2.5305256612552017e-07
# val acc:0.08 | lr:2.0967753865100842e-05, weight decay:8.23444524940545e-07
# =========== Hyper-Parameter Optimization Result ===========
# Best-1(val acc:0.78) | lr:0.00592501475408638, weight decay:2.884905840440795e-06
# Best-2(val acc:0.71) | lr:0.0046593335222353625, weight decay:7.06531358012995e-05
# Best-3(val acc:0.66) | lr:0.006024290557714003, weight decay:4.130863810219471e-06
# Best-4(val acc:0.65) | lr:0.004238683875024437, weight decay:4.789331621675631e-07
# Best-5(val acc:0.65) | lr:0.003294149174060375, weight decay:7.500473917659556e-05
# Best-6(val acc:0.6) | lr:0.004412585867480103, weight decay:1.8694614516097234e-06
# Best-7(val acc:0.58) | lr:0.0037568045548814007, weight decay:8.586635791523419e-07
# Best-8(val acc:0.51) | lr:0.002257959137040223, weight decay:2.1335895666178243e-05
# Best-9(val acc:0.48) | lr:0.002429521019985163, weight decay:4.4981712769966694e-07
# Best-10(val acc:0.47) | lr:0.003108690193174507, weight decay:1.472381807925704e-05
# Best-11(val acc:0.45) | lr:0.0034309268623154937, weight decay:2.5261852470647776e-08
# Best-12(val acc:0.36) | lr:0.0029522196861681985, weight decay:1.8318572995694304e-06
# Best-13(val acc:0.35) | lr:0.001889395257136538, weight decay:2.3673684530307035e-08
# Best-14(val acc:0.33) | lr:0.001595086009717874, weight decay:3.336956730149232e-06
# Best-15(val acc:0.29) | lr:0.0015342692652872002, weight decay:4.1100976474583763e-08
# Best-16(val acc:0.26) | lr:0.0012550273881050736, weight decay:1.3457383185497738e-05
# Best-17(val acc:0.24) | lr:0.002826015242454688, weight decay:2.7710072733447147e-06
# Best-18(val acc:0.24) | lr:0.0016693655529397037, weight decay:1.9660530872579584e-05
# Best-19(val acc:0.22) | lr:0.0008907144096194279, weight decay:2.2111168114259186e-07
# Best-20(val acc:0.21) | lr:0.0015738527626970632, weight decay:1.02082144476431e-08
