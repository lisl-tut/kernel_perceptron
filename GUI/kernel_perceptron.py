# カーネルパーセプトロンの実装プログラム
# 更新の状況をアニメーションとして描画する

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

###########################################################

class kernel_perceptron: # カーネルパーセプトロン
    def __init__(self, data1, data2, kernel='nothing', epsilon=0.05, **kwargs):
        # データを(x, y, label)の形式でまとめる
        self.data = self.create_dataset(data1, data2)

        # カーネル関数を登録
        if kernel == 'nothing':
            self.kernel = self.id_kernel
        elif kernel == 'gauss':
            self.kernel = self.gauss_kernel
            self.sigma2 = kwargs['sigma2']
        else:
            raise Exception('kernel argument is not proper')

        # 勾配法のステップサイズ(ラーニングレート)を登録
        self.epsilon = epsilon

        # パラメータはすべて0で初期化
        self.param = np.zeros(len(self.data))

        # パーセプトロンの更新回数を初期化
        self.idx_count = 0      # 更新に使用するデータ番号のカウンタ
        self.update_count = 0   # 更新回数の確認のためのカウンタ

    def create_dataset(self, data1, data2):
        # データを(x, y, label)の形式でまとめる
        label1 = -np.ones((len(data1),1))                   # class1のラベルを生成
        label2 = +np.ones((len(data2),1))                   # class2のラベルを生成
        labeled_data1 = np.hstack((data1, label1))          # ラベルをデータに追加
        labeled_data2 = np.hstack((data2, label2))          # ラベルをデータに追加
        data = np.vstack((labeled_data1, labeled_data2))    # 2つのデータをまとめる
        np.random.shuffle(data)                             # データをシャッフル
        return data

    def update(self): # オンライン(データ1つ)で更新を行う関数
        idx = self.idx_count % len(self.data)        # 使用するデータ番号を取得
        self.idx_count += 1
        x, y, t_true = self.data[idx]                # 特徴量とラベルの答えを取り出す
        t_tilde = self.disc_func(x, y)               # 識別関数からラベルを推測
        if t_true * t_tilde < 0:                     # 答えと推測値が異なる場合
            self.param[idx] += self.epsilon * t_true # 勾配法でパラメータを更新
            self.update_count += 1
            os.system('clear')
            print('update count: %d (%d)' % (self.update_count, self.idx_count))
            print('parameter:')
            print(self.param)
            return True                              # 更新したらTrueを返却
        else:
            return False                             # 更新しなかったらFalseを返却

    def is_all_correct(self): # すべてのデータが正しく識別されたかを確認する関数
        x = self.data.T[0]
        y = self.data.T[1]
        t_true = self.data.T[2]
        t_tilde = self.disc_func(x, y)
        if all(t_true * t_tilde > 0) == True:
            return True  # 答えとすべての推定値が合っていた場合
        else:
            return False # 答えといくつかの推測値が異なる場合

    def test(self, test_data1, test_data2):
        test_data = self.create_dataset(test_data1, test_data2)
        x = test_data.T[0]
        y = test_data.T[1]
        t_true = test_data.T[2]
        t_tilde = self.disc_func(x, y)
        accuracy = len(t_true[t_true*t_tilde>0]) / len(t_true) # 正解数 / データ数
        return accuracy

    def disc_func(self, x, y): # 識別関数
        #### Σ_k α_k*K(x, x_k) ####
        val = 0
        for k in range(len(self.data)):
            x_k, y_k = self.data[k][:2]                  # x_k
            alpha_k = self.param[k]                      # α_k
            val += alpha_k * self.kernel(x, y, x_k, y_k) # Σ_k α_k*K(x, x_k)
        #### sign(Σ_k α_k*K(x, x_k)) ####
        label = np.array(val)
        label[label>=0] = 1     # if val >= 0, label is  1
        label[label<0] = -1     # if val <  0, label is -1

        return label

    def id_kernel(self, x, y, x_k, y_k):
        return 1 + x*x_k + y*y_k

    def gauss_kernel(self, x, y, x_k, y_k):
        return np.exp(-1/(2*self.sigma2)*((x-x_k)**2+(y-y_k)**2))

###########################################################

class border_plotter:
    def __init__(self, resolution=50):
        self.x_range = (-1,1)
        self.y_range = (-1,1)
        x = np.linspace(self.x_range[0], self.x_range[1], resolution)
        y = np.linspace(self.y_range[0], self.y_range[1], resolution)
        self.X, self.Y = np.meshgrid(x, y)

        self.fig = plt.figure()
        self.img_list = []
        plt.axes().set_aspect('equal')

    def fig_num(self):
        return len(self.img_list)

    def take_a_shot(self, f, update_count=None):
        Z = f(self.X, self.Y)
        img = plt.pcolor(self.X, self.Y, Z, cmap='viridis') # 画像の生成
        if update_count != None:
            txt = plt.text(
                    self.x_range[0],
                    self.y_range[1] + 0.02,
                    'update_count: ' + str(update_count))   # 番号の描画
            self.img_list.append([img, txt])                # 画像, 番号の登録
        else:
            self.img_list.append([img])                     # 画像のみ登録

    def show_figures(self, data1, data2):
        # データ点の描画
        plt.scatter(data1.T[0], data1.T[1], c='red', marker='o', s=30, label='class 1')
        plt.scatter(data2.T[0], data2.T[1], c='blue', marker='x', s=50, label='class 2')

        # 描画の設定
        plt.xlim(self.x_range[0], self.x_range[1])
        plt.ylim(self.y_range[0], self.y_range[1])
        plt.legend()
        plt.grid()

        # 最後の画面で停止するように，最後のフレームのコピーを追加
        self.img_list += [self.img_list[-1]] * 10

        # アニメーションの生成，表示
        ani = animation.ArtistAnimation(self.fig, self.img_list, interval=500)
        # ani.save('anim.gif', writer="pillow")
        plt.show()

###########################################################

def split_train_and_test(data, test_ratio):
    np.random.shuffle(data)                 # データをシャッフル
    test_num = int(len(data) * test_ratio)  # 試験データの数を計算
    test_data = data[:test_num]             # 試験データを取り出す
    train_data = data[test_num:]            # 訓練データを取り出す
    return train_data, test_data

def main(data1, data2, kernel_type, epsilon, test_ratio, resolution, **kwargs):
    # 訓練データとテストデータに分ける
    train_data1, test_data1 = split_train_and_test(data1, test_ratio)
    train_data2, test_data2 = split_train_and_test(data2, test_ratio)

    # モデルのインスタンス生成
    kp = kernel_perceptron(train_data1, train_data2,
                            kernel=kernel_type, epsilon=epsilon, **kwargs) # カーネルパーセプトロン
    plotter = border_plotter(resolution=resolution)                        # 境界面の描画器

    # 更新とグラフ画像の生成
    plotter.take_a_shot(kp.disc_func, kp.update_count)         # 初期状態の境界面の画像の生成，保存
    while plotter.fig_num() < 100 and kp.is_all_correct() != True:
        if kp.update() == True:                                # 更新したとき
            plotter.take_a_shot(kp.disc_func, kp.update_count) # 更新後の境界面の画像の生成，保存

    # 学習が完了したかを出力
    if plotter.fig_num() < 100:
        print('Complete')                   # すべてのデータの学習が終了したとき
    else:
        print('Reached the repeat limit')   # 繰り返し回数の上限に達したとき

    # アニメーションの描画
    plotter.show_figures(train_data1, train_data2)

    acc = kp.test(test_data1, test_data2)
    print('acc:', acc)