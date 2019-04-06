# カーネルパーセプトロンの実装プログラム
# 更新の状況をアニメーションとして描画する

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
        x1, y1 = (0.30, 0.70)  # クラス1のデータ生成の中心
        x2, y2 = (0.75, 0.35)  # クラス2のデータ生成の中心
        def disc_func(x, y):   # 真の境界面(識別関数)の方程式
            return (x1-x2)*(x-(x1+x2)/2)+(y1-y2)*(y-(y1+y2)/2)
        def bound_func1(x, y): # クラス1のデータの生成領域
            return disc_func(x, y) > 0 and (x-x1)**2+(y-y1)**2 < 0.07
        def bound_func2(x, y): # クラス2のデータの生成領域
            return disc_func(x, y) < 0 and (x-x2)**2+(y-y2)**2 < 0.07

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

    def get_data_type3(self, num): # 線形分離不可能なデータを生成する関数
        a = 0.15
        b = 0.85
        def disc_func(x, y):   # 真の境界面(識別関数)の方程式
            return ((x-a)**2+(y-a)**2)*((x-b)**2+(y-a)**2)*((x-a)**2+(y-b)**2)*((x-b)**2+(y-b)**2) - 0.0035
        def bound_func1(x, y): # クラス1のデータの生成領域
            return disc_func(x, y) > 0
        def bound_func2(x, y): # クラス2のデータの生成領域
            return disc_func(x, y) < 0

        data1 = self.__sampling(bound_func1, num)   # クラス1のサンプリング
        data2 = self.__sampling(bound_func2, num)   # クラス2のサンプリング
        self.disc_func = disc_func                  # 真の識別関数を保持
        return data1, data2

###########################################################

class kernel_perceptron: # カーネルパーセプトロン
    def __init__(self, data1, data2, kernel='normal', epsilon=0.05):
        # データを(x, y, label)の形式でまとめる
        label1 = -np.ones((len(data1),1))                     # class1のラベルを生成
        label2 = +np.ones((len(data2),1))                     # class2のラベルを生成
        labeled_data1 = np.hstack((data1, label1))            # ラベルをデータに追加
        labeled_data2 = np.hstack((data2, label2))            # ラベルをデータに追加
        self.data = np.vstack((labeled_data1, labeled_data2)) # 2つのデータをまとめる
        np.random.shuffle(self.data)                          # データをシャッフル

        # カーネル関数を登録
        if kernel == 'normal':
            self.kernel = self.normal_kernel
        elif kernel == 'gauss':
            self.kernel = self.gauss_kernel
        else:
            raise Exception('kernel argument is not proper')

        # 勾配法のステップサイズ(ラーニングレート)を登録
        self.epsilon = epsilon

        # パラメータはすべて0で初期化
        self.param = np.zeros(len(self.data))

        # パーセプトロンの更新回数を初期化
        self.idx_count = 0      # 更新に使用するデータ番号のカウンタ
        self.update_count = 0   # 更新回数の確認のためのカウンタ

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

    def normal_kernel(self, x, y, x_k, y_k):
        return x*x_k + y*y_k # bug! 計算方法がおかしい，切片が存在しない

    def gauss_kernel(self, x, y, x_k, y_k):
        sigma2 = 0.01
        return np.exp(-1/(2*sigma2)*((x-x_k)**2+(y-y_k)**2)) # bug! normal_kernelのバグと同様

###########################################################

def plot_colormap(f, x_range=(0,1), y_range=(0,1)):
    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    img = plt.pcolor(X, Y, Z, cmap='viridis')
    return img

def plot_implicit(f, x_range=(0,1), y_range=(0,1)):
    x = np.linspace(x_range[0], x_range[1], 200)
    y = np.linspace(y_range[0], y_range[1], 200)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    # f(x, y) = 0 となる部分を描画する
    plt.contour(X, Y, Z, [0], colors=['magenta'], linestyles='dashed')

def show_figures(data1, data2, img_list, f_true=None, x_range=(0,1), y_range=(0,1)):
    # データ点の描画
    plt.scatter(data1.T[0], data1.T[1], c='red', marker='o', s=30, label='class 1')
    plt.scatter(data2.T[0], data2.T[1], c='blue', marker='x', s=50, label='class 2')

    # 描画の設定
    plt.xlim(x_range[0], x_range[1])
    plt.ylim(y_range[0], y_range[1])
    # plt.axis('equal')
    plt.legend()
    plt.grid()

    # 真の境界を描画
    if f_true != None:
        plot_implicit(f_true)

    # 最後の画面で停止するように，最後のフレームのコピーを追加
    img_list += [img_list[-1]] * 20

    # アニメーションの生成，表示
    ani = animation.ArtistAnimation(fig, img_list, interval=100)
    plt.show()

###########################################################

if __name__ == '__main__':
    dg = data_generator()
    data1, data2 = dg.get_data_type3(50)
    kp = kernel_perceptron(data1, data2, kernel='gauss')

    fig = plt.figure()
    img_list = []
    for i in range(1000):
        if kp.update() == True:
            img = plot_colormap(kp.disc_func)
            img_list.append([img])
        if kp.is_all_correct() == True:
            print('Complete')
            break
    else:
        print('Reached the repeat limit')
    show_figures(data1, data2, img_list, dg.disc_func)
