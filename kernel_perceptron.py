# このプログラムはパーセプトロンの動作を可視化するためのものです
# x, y, labelの説明くらい書いたほうがいいかな
#

import numpy as np
import matplotlib.pyplot as plt

###########################################################

class data_generator: # データ生成機
    def __init__(self):
        self.disc_func = None # データを生成したら，ここに識別関数を登録する

    def __sampling(self, bound_func, num):
        data = []
        while len(data) < num:
            x, y = np.random.rand(2)
            if bound_func(x, y) == True: # 指定範囲内にデータがあるとき
                data.append((x,y))       # データを登録
        return np.array(data)

    def get_data_type1(self, num): # 線形分離可能なデータを生成する関数
        def disc_func(x, y):   # 真の境界面(識別関数)の方程式
            return (0.20-0.80)*(x-(0.20+0.80)/2)+(0.75-0.30)*(y-(0.75+0.30)/2)
        def bound_func1(x, y): # クラス1のデータの生成領域
            return disc_func(x, y) > 0 and (x-0.20)**2+(y-0.75)**2 < 0.07
        def bound_func2(x, y): # クラス2のデータの生成領域
            return disc_func(x, y) < 0 and (x-0.80)**2+(y-0.30)**2 < 0.07

        data1 = self.__sampling(bound_func1, num)   # クラス1のサンプリング
        data2 = self.__sampling(bound_func2, num)   # クラス2のサンプリング
        self.disc_func = disc_func                  # 真の識別関数を保持
        return data1, data2

    def get_data_type2(self, num): # 線形分離不可能なデータを生成する関数
        def disc_func(x, y):   # 真の境界面(識別関数)の方程式
            return -(10*(x-0.5))**3+2*(10*(x-0.5))**2+7*(10*(x-0.5))+(10*(y-0.5))**3-(10*(y-0.5))**2+(10*(y-0.5))
        def bound_func1(x, y): # クラス1のデータの生成領域
            return disc_func(x, y) > 0 and (x-0.5)**2+3*(y-0.5)**2-0.6*x*y-0.2 < 0
        def bound_func2(x, y): # クラス2のデータの生成領域
            return disc_func(x, y) < 0 and (x-0.6)**2+(y-0.4)**2-0.16 < 0

        data1 = self.__sampling(bound_func1, num)   # クラス1のサンプリング
        data2 = self.__sampling(bound_func2, num)   # クラス2のサンプリング
        self.disc_func = disc_func                  # 真の識別関数を保持
        return data1, data2

###########################################################

class kernel_perceptron: # カーネルパーセプトロン
    def __init__(self, data1, data2, kernel=None, epsilon=0.01):
        # データを(x, y, label)の形式でまとめる
        label1 = -np.ones((len(data1),1))                     # class1のラベルを生成
        label2 = +np.ones((len(data2),1))                     # class2のラベルを生成
        labeled_data1 = np.hstack((data1, label1))            # ラベルをデータに追加
        labeled_data2 = np.hstack((data2, label2))            # ラベルをデータに追加
        self.data = np.vstack((labeled_data1, labeled_data2)) # 2つのデータをまとめる
        np.random.shuffle(self.data)                          # データをシャッフル

        # カーネル関数を登録
        self.kernel = self.normal_kernel
        # self.kernel = self.gauss_kernel

        # パラメータ更新のステップサイズ(ラーニングレート)を登録
        self.epsilon = epsilon

        # パラメータはすべて0で初期化
        self.param = np.zeros(len(self.data))

        # パーセプトロンの更新回数を初期化
        self.update_count = 0

    def update(self): # オンライン(データ1つ)で更新を行う関数
        idx = self.update_count % len(self.data)     # 今回使用するデータ番号を取得
        x, y, t_true = self.data[idx]                # 特徴量とラベルの答えを取り出す
        t_tilde = self.disc_func(x, y)               # 識別関数からラベルを推測
        print(t_true, t_tilde)
        if t_true * t_tilde < 0:                     # 答えと推測値が異なる場合
            self.param[idx] += self.epsilon * t_true # パラメータを更新
        self.update_count += 1                       # 更新をカウント

    def disc_func(self, x, y): # 識別関数
        val = 0
        for k in range(len(self.data)):
            x_k, y_k = self.data[k][:2]                  # x_k
            alpha_k = self.param[k]                      # α_k
            val += alpha_k * self.kernel(x, y, x_k, y_k) # Σ_k α_k*K(x, x_k)
        label = np.sign(val)                             # sign(Σ_k α_k*K(x, x_k))
        return label

    def normal_kernel(self, x, y, x_k, y_k):
        return x*x_k + y*y_k

    def gauss_kernel(self, x, y, x_k, y_k):
        sigma2 = 0.1
        return np.exp(-1/(2*sigma2)*((x-x_k)**2+(y-y_k)**2))

###########################################################

def plot_figure(data1, data2, f):
    # データ点の描画
    plt.scatter(data1.T[0], data1.T[1], marker='o', label='class 1')
    plt.scatter(data2.T[0], data2.T[1], marker='x', label='class 2')

    # 境界線の描画
    x_range = (0, 1)
    y_range = (0, 1)
    x = np.linspace(x_range[0], x_range[1], 500)
    y = np.linspace(y_range[0], y_range[1], 500)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    plt.pcolor(X, Y, Z)
    # plt.contour(X, Y, Z, [0]) # f(x, y) = 0 となる部分を描画する

    # 描画の設定，表示
    plt.xlim(x_range[0], x_range[1])
    plt.ylim(y_range[0], y_range[1])
    plt.legend()
    plt.axis('equal')
    plt.grid()
    plt.show()

###########################################################

if __name__ == '__main__':
    dg = data_generator()
    data1, data2 = dg.get_data_type1(30)
    kp = kernel_perceptron(data1, data2)
    for i in range(30):
        kp.update()
        print(i)
        if i % 10 == 0:
            plot_figure(data1, data2, kp.disc_func)

