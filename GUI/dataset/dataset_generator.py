# カーネルパーセプトロンの実装プログラム, データ生成器

import sys, os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
        data1 = data1*2-1                           # データを[0,1]から[-1,1]の範囲にスケーリング
        data2 = data2*2-1                           # データを[0,1]から[-1,1]の範囲にスケーリング
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
        data1 = data1*2-1                           # データを[0,1]から[-1,1]の範囲にスケーリング
        data2 = data2*2-1                           # データを[0,1]から[-1,1]の範囲にスケーリング
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
        data1 = data1*2-1                           # データを[0,1]から[-1,1]の範囲にスケーリング
        data2 = data2*2-1                           # データを[0,1]から[-1,1]の範囲にスケーリング
        return data1, data2


if __name__ == '__main__':    
    dg = data_generator()                       
    data1, data2 = dg.get_data(type_=1, num=100)
    
    plt.scatter(data1.T[0], data1.T[1], c='red', marker='o', s=30, label='class 1')
    plt.scatter(data2.T[0], data2.T[1], c='blue', marker='x', s=50, label='class 2')
    plt.show()

    np.savez('データセット1(線形分離可能).npz', data1=data1, data2=data2)