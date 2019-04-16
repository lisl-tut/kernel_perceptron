# カーネルパーセプトロンの実装プログラム
# 更新の状況をアニメーションとして描画する

import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

###########################################################

class data_generator: # データ生成機
    def __init__(self):
        self.disc_func = None # データを生成したら，ここに識別関数を登録する

    def get_data(self, type_, num):
        if type_ == 1:
            return self.get_data_type1(num)
        elif type_ == 2:
            return self.get_data_type2(num)
        elif type_ == 3:
            return self.get_data_type3(num)
        else:
            raise Exception('type augment is not proper')

    def __sampling(self, bound_func, num):
        data = []
        while len(data) < num:
            x, y = np.random.rand(2)
            if bound_func(x, y) == True: # 指定範囲内にデータがあるとき
                data.append((x,y))       # データを登録
        return np.array(data)

    def get_data_type1(self, num): # 線形分離可能なデータを生成する関数
        x1, y1 = (0.30, 0.35)  # クラス1のデータ生成の中心
        x2, y2 = (0.75, 0.70)  # クラス2のデータ生成の中心
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
        a, b = (10, 10)
        x1, y1 = (0.5, 0.5)
        def disc_func(x, y):   # 真の境界面(識別関数)の方程式
            return -(a*(x-x1))**3+2*(a*(x-x1))**2+7*(a*(x-x1))+(b*(y-y1))**3-(b*(y-y1))**2+(b*(y-y1))
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
            self.sigma2 = float(input('sigma^2: '))
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
        return 1 + x*x_k + y*y_k

    def gauss_kernel(self, x, y, x_k, y_k):
        return np.exp(-1/(2*self.sigma2)*((x-x_k)**2+(y-y_k)**2))

###########################################################

class kernel_perceptron_plotter:
    def __init__(self, resolution=50):
        self.x_range = (0,1)
        self.y_range = (0,1)
        x = np.linspace(self.x_range[0], self.x_range[1], resolution)
        y = np.linspace(self.y_range[0], self.y_range[1], resolution)
        self.X, self.Y = np.meshgrid(x, y)

        self.fig = plt.figure()
        self.img_list = []

    def fig_num(self):
        return len(self.img_list)

    def take_a_shot(self, f, update_count=None):
        Z = f(self.X, self.Y)
        img = plt.pcolor(self.X, self.Y, Z, cmap='viridis')         # 画像の生成
        if update_count != None:
            txt = plt.text(0, 1.02, 'update_count: '+str(update_count)) # 番号の描画
            self.img_list.append([img, txt])                        # 画像, 番号の登録
        else:
            self.img_list.append([img])                             # 画像のみ登録

    def show_figures(self, data1, data2, f_true=None):
        # データ点の描画
        plt.scatter(data1.T[0], data1.T[1], c='red', marker='o', s=30, label='class 1')
        plt.scatter(data2.T[0], data2.T[1], c='blue', marker='x', s=50, label='class 2')

        # 描画の設定
        plt.xlim(self.x_range[0], self.x_range[1])
        plt.ylim(self.y_range[0], self.y_range[1])
        # plt.axis('equal')
        plt.legend()
        plt.grid()

        # 真の境界線を描画
        if f_true != None:
            # f(x, y) = 0 となる部分を描画する
            resolution = 200
            x = np.linspace(self.x_range[0], self.x_range[1], resolution)
            y = np.linspace(self.y_range[0], self.y_range[1], resolution)
            X, Y = np.meshgrid(x, y)
            Z = f_true(X, Y)
            plt.contour(X, Y, Z, [0], colors=['white'], linestyles='dashed')

        # 最後の画面で停止するように，最後のフレームのコピーを追加
        self.img_list += [self.img_list[-1]] * 10

        # アニメーションの生成，表示
        ani = animation.ArtistAnimation(self.fig, self.img_list, interval=500)
        ani.save('anim.gif', writer="imagemagick")
        # plt.show()

###########################################################

def main(data_type, data_num, kernel_type, epsilon, resolution, show_f_true):
    # データ生成
    dg = data_generator()                                     # データ生成器
    data1, data2 = dg.get_data(type_=data_type, num=data_num) # データを生成

    # カーネルパーセプトロン
    kp = kernel_perceptron(data1, data2, kernel=kernel_type, epsilon=epsilon)

    # カーネルパーセプトロンの状態の描画器
    plotter = kernel_perceptron_plotter(resolution=resolution)

    # 更新とグラフ画像の生成
    while plotter.fig_num() < 100:
        if kp.update() == True: # 更新したとき
            # グラフ画像の生成，保存
            plotter.take_a_shot(kp.disc_func, kp.update_count)

            # 全データの答え合わせ
            if kp.is_all_correct() == True:
                print('Complete')
                break                       # すべて答えが合っていれば終了
    else:
        print('Reached the repeat limit')   # 繰り返し回数の上限に達したとき終了

    # アニメーションの描画
    if show_f_true == True:
        plotter.show_figures(data1, data2, f_true=dg.disc_func)
    else:
        plotter.show_figures(data1, data2)


###########################################################

if __name__ == '__main__':
    # np.random.seed(777)
    main(
        data_type=int(sys.argv[1]),
        data_num=int(sys.argv[2]),
        kernel_type=str(sys.argv[3]),
        epsilon=float(sys.argv[4]),
        resolution=int(sys.argv[5]),
        show_f_true=False if sys.argv[6] == 'False' else True
    )
