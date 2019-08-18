import numpy as np

import tkinter as tk                            # tk本体
import tkinter.ttk as ttk                       # スタイル付きtk
from tkinter import filedialog as tkFileDialog  # ファイルダイアログボックス

import kernel_perceptron as kp

class kernel_perceptron_learner():
    def __init__(self):
        # ウィンドウの作成・設定
        self.win = tk.Tk()                # ウィンドウを作成
        win = self.win
        win.title('カーネルパーセプトロン') # ウィンドウタイトルの設定

        # 全体のフォントの設定
        win.option_add("*Font", "TkDefaultFont 12")
        ttk.Style().configure('TRadiobutton', font=('TkDefaultFont', 12), bg='white')
        ttk.Style().configure('TButton', font=('TkDefaultFont', 12), bg='white')

        # 左右のフレーム
        fr_l = tk.Frame(win)
        fr_l.pack(padx=5, pady=5, fill='y', side='left')
        fr_r = tk.Frame(win)
        fr_r.pack(padx=5, pady=5, fill='y', side='left')

        #### #### 左側 #### ####
        # キャンバス用のフレーム
        fr_canvas = tk.LabelFrame(fr_l, bd=2, relief='groove', text='データキャンバス')
        fr_canvas.pack(padx=5, pady=2, fill='x')

        # キャンバス操作ボタン
        fr_sw = tk.Frame(fr_canvas)
        fr_sw.pack(padx=5, pady=2, fill='x')
        load_btn = ttk.Button(fr_sw)
        load_btn.configure(text='読込', width=5, command=lambda:self.load_data())
        load_btn.pack(side='left')
        save_btn = ttk.Button(fr_sw)
        save_btn.configure(text='保存', width=5, command=lambda:self.save_data())
        save_btn.pack(side='left')
        clear_btn = ttk.Button(fr_sw)
        clear_btn.configure(text='クリア', width=5, command=lambda:self.clear_data())
        clear_btn.pack(side='left')
        clear_btn = ttk.Button(fr_sw)
        clear_btn.configure(text='更新', width=5, command=lambda:print('update clicked'))
        clear_btn.pack(padx=10, side='left')

        # キャンバス
        self.canvas_size = 500
        self.canvas = tk.Canvas(fr_canvas, width=self.canvas_size, height=self.canvas_size, bg='white')
        self.canvas.bind('<Button-1>', lambda event:self.click_left(event))   # 左クリックのイベントを設定
        self.canvas.bind('<Button-3>', lambda event:self.click_right(event))  # 右クリックのイベントを設定
        self.canvas.pack(padx=5, pady=5)

        #### #### 右側 #### ####
        # オプション設定
        fr_opt = tk.LabelFrame(fr_r, bd=2, relief='groove', text='学習オプション')
        fr_opt.pack(padx=5, pady=2, fill='x')

        # オプション1
        fr_opt1 = tk.LabelFrame(fr_opt, relief='flat', text='【特徴量変換】',)
        fr_opt1.pack(padx=5, pady=4, fill='x')
        self.opt1_feature = tk.StringVar()                              # 特徴量の取得用変数
        self.opt1_feature.set('2d')

        opt1_rbtn1 = ttk.Radiobutton(fr_opt1, variable=self.opt1_feature, value='2d', text='2個 (直線)')
        opt1_rbtn1.grid(row=1, column=1, sticky=tk.W)
        opt1_feature2D_label = tk.Label(fr_opt1, text=' 式 = ')         # 特徴量の式のラベル
        opt1_feature2D_label.grid(row=1, column=2, sticky=tk.E)
        self.opt1_feature2D_formula = tk.Entry(fr_opt1)                 # 特徴量の式
        self.opt1_feature2D_formula.grid(row=1, column=3)
        opt1_feature2D_label.configure(state='disabled')                #### 無効化
        self.opt1_feature2D_formula.configure(state='disabled')         #### 無効化

        opt1_rbtn2 = ttk.Radiobutton(fr_opt1, variable=self.opt1_feature, value='3d', text='3個 (平面)')
        opt1_rbtn2.grid(row=2, column=1, sticky=tk.W)
        opt1_feature3D_label = tk.Label(fr_opt1, text=' 式 = ')         # 特徴量の式のラベル
        opt1_feature3D_label.grid(row=2, column=2, sticky=tk.E)
        self.opt1_feature3D_formula = tk.Entry(fr_opt1)                 # 特徴量の式
        self.opt1_feature3D_formula.grid(row=2, column=3)
        opt1_rbtn2.configure(state='disabled')                          #### 無効化
        opt1_feature3D_label.configure(state='disabled')                #### 無効化
        self.opt1_feature3D_formula.configure(state='disabled')         #### 無効化

        opt1_rbtn3 = ttk.Radiobutton(fr_opt1, variable=self.opt1_feature, value='gauss', text='カーネル法 (超平面) ： ガウス')
        opt1_rbtn3.grid(row=3, column=1, sticky=tk.W)
        opt1_featureGauss_sigma2 = tk.Label(fr_opt1, text='   σ^2 = ')  # 特徴量の式のラベル
        opt1_featureGauss_sigma2.grid(row=3, column=2, sticky=tk.E)
        self.opt1_featureGauss_sigma2 = tk.Entry(fr_opt1)               # 特徴量の式
        self.opt1_featureGauss_sigma2.grid(row=3, column=3)

        # オプション2
        fr_opt2 = tk.LabelFrame(fr_opt, relief='flat', text='【データ数】',)
        fr_opt2.pack(padx=5, pady=4, fill='x')
        opt2_specified_data_num_label = tk.Label(fr_opt2, text=' n = ')     # 指定データ数のラベル
        opt2_specified_data_num_label.grid(row=1, column=1, sticky=tk.E)
        self.opt2_specified_data_num = tk.Entry(fr_opt2, width=5)           # 指定データ数
        self.opt2_specified_data_num.grid(row=1, column=2)
        self.opt2_total_data_num = tk.StringVar()                           # 総データ数の取得用変数
        self.opt2_total_data_num.set(' (max=0)')
        opt2_total_data_nu_label = tk.Label(fr_opt2, textvariable=self.opt2_total_data_num) # 総データ数のラベル
        opt2_total_data_nu_label.grid(row=1, column=3)

        # オプション3
        fr_opt3 = tk.LabelFrame(fr_opt, relief='flat', text='【テストデータの割合】',)
        fr_opt3.pack(padx=5, pady=4, fill='x')
        self.opt3_test_ratio = tk.StringVar()                           # テストデータの割合の取得用変数
        self.opt3_test_ratio.set('0.25')
        opt3_rbtn1 = ttk.Radiobutton(fr_opt3, variable=self.opt3_test_ratio, value='0.1', text='10％')
        opt3_rbtn1.grid(row=1, column=1)
        opt3_rbtn2 = ttk.Radiobutton(fr_opt3, variable=self.opt3_test_ratio, value='0.25', text='25％')
        opt3_rbtn2.grid(row=1, column=2)
        opt3_rbtn3 = ttk.Radiobutton(fr_opt3, variable=self.opt3_test_ratio, value='0.5', text='50％')
        opt3_rbtn3.grid(row=1, column=3)

        # オプション4
        fr_opt4 = tk.LabelFrame(fr_opt, relief='flat', text='【ラーニングレート】',)
        fr_opt4.pack(padx=5, pady=4, fill='x')
        self.opt4_lr = tk.StringVar()                                   # ラーニングレートの取得用変数
        self.opt4_lr.set('0.10')
        opt4_rbtn1 = ttk.Radiobutton(fr_opt4, variable=self.opt4_lr, value='0.50', text='0.50')
        opt4_rbtn1.grid(row=1, column=1)
        opt4_rbtn2 = ttk.Radiobutton(fr_opt4, variable=self.opt4_lr, value='0.10', text='0.10')
        opt4_rbtn2.grid(row=1, column=2)
        opt4_rbtn3 = ttk.Radiobutton(fr_opt4, variable=self.opt4_lr, value='0.01', text='0.01')
        opt4_rbtn3.grid(row=1, column=3)

        # オプション5
        fr_opt5 = tk.LabelFrame(fr_opt, relief='flat', text='【表示解像度】',)
        fr_opt5.pack(padx=5, pady=4, fill='x')
        self.opt5_resolution = tk.StringVar()                           # 表示解像度の取得用変数
        self.opt5_resolution.set('50')
        opt5_rbtn1 = ttk.Radiobutton(fr_opt5, variable=self.opt5_resolution, value='50', text='50')
        opt5_rbtn1.grid(row=1, column=1)
        opt5_rbtn2 = ttk.Radiobutton(fr_opt5, variable=self.opt5_resolution, value='100', text='100')
        opt5_rbtn2.grid(row=1, column=2)
        opt5_rbtn3 = ttk.Radiobutton(fr_opt5, variable=self.opt5_resolution, value='200', text='200')
        opt5_rbtn3.grid(row=1, column=3)

        # 学習開始ボタン
        start_btn = ttk.Button(fr_r)
        start_btn.configure(text='学習開始', width=15, command=lambda:self.start_learning())
        start_btn.pack(padx=5, pady=12)

        # ログ画面
        fr_log = tk.LabelFrame(fr_r, bd=2, relief='flat', text='ログ')
        fr_log.pack(padx=5, pady=5, fill='x')
        self.logbox = tk.Listbox(fr_log, font=('TkDefaultFont', 10))
        self.logbox.configure(height=10, width=80)
        self.logbox.pack(padx=5, pady=5)

        # すべての設定終了後の操作
        self.init_canvas()      # キャンバス初期化，データセット初期化
        win.mainloop()          # ループに入る

    ###########################################################################

    # ログを出力するメソッド
    def print_log(self, *args):
        if len(args) == 0:
            return
        string = args[0]
        for arg in args[1:]:
            string += ' ' + arg
        self.logbox.insert(0, string)

    # キャンバスをクリアしてグリッドを描画するメソッド
    def init_canvas(self):
        # 描画されているものをすべて削除
        self.canvas.delete("all")

        # データセットの初期化
        self.data1_all = np.empty((0,2), int)   # すべてのデータ
        self.data2_all = np.empty((0,2), int)   # すべてのデータ
        self.data1 = np.empty((0,2), int)       # サンプリングされたあとのデータ
        self.data2 = np.empty((0,2), int)       # サンプリングされたあとのデータ

        # グリッド線の描画
        center = self.canvas_size / 2
        for i in range(0, self.canvas_size, int(self.canvas_size/10)):
            if i != center:
                self.canvas.create_line(0, i, self.canvas_size, i, fill='grey')
                self.canvas.create_line(i, 0, i, self.canvas_size, fill='grey')
            else:
                self.canvas.create_line(0, i, self.canvas_size, i, fill='black')
                self.canvas.create_line(i, 0, i, self.canvas_size, fill='black')

        # データ点総数を更新
        self.opt2_total_data_num.set(' (max='+str(self.get_total_data_num())+')')


    # キャンバスに点（タイプA）を打つメソッド
    def dot_point_A(self, x, y):
        self.canvas.create_oval(x-5, y-5, x+5, y+5, fill='red', width=0)

    # キャンバスに点（タイプB）を打つメソッド
    def dot_point_B(self, x, y):
        self.canvas.create_oval(x-5, y-5, x+5, y+5, fill='blue', width=0)

    # キャンバスに点（タイプA:犬）を打つメソッド
    def dot_dog(self, x, y):
        pass

    # キャンバスに点（タイプB:猫）を打つメソッド
    def dot_cat(self, x, y):
        pass

    ###########################################################################

    # 左クリック用：データ点（タイプA）を描画するメソッド
    def click_left(self, event):
        self.print_log('dot A:', str([event.x, event.y]))
        self.dot_point_A(event.x, event.y)                                        # 点を描画
        self.data1 = np.append(self.data1, [[event.x,event.y]], axis=0)           # 点をデータベースに追加
        self.opt2_total_data_num.set(' (max='+str(self.get_total_data_num())+')') # データ点総数を更新

    # 右クリック用：データ点（タイプB）を描画するメソッド
    def click_right(self, event):
        self.print_log('dot B:', str([event.x, event.y]))
        self.dot_point_B(event.x, event.y)                                        # 点を描画
        self.data2 = np.append(self.data2, [[event.x,event.y]], axis=0)           # 点をデータベースに追加
        self.opt2_total_data_num.set(' (max='+str(self.get_total_data_num())+')') # データ点総数を更新

    # 読み込みボタン用：データセットをロードするメソッド
    def load_data(self):
        filename = tkFileDialog.askopenfilename(filetypes=[('npzファイル', '*.npz')], initialdir='./dataset/')
        if filename == '':
            return # キャンセル
        self.print_log('load:', filename)

        # データセットの読み込み
        dataset = np.load(filename)
        np_data1 = self.transform_coordinate_system_for_canvas(dataset['data1']) # 座標をキャンバス用に変換
        np_data2 = self.transform_coordinate_system_for_canvas(dataset['data2']) # 座標をキャンバス用に変換

        # データをすべて更新
        self.init_canvas()
        self.data1 = np_data1
        self.data2 = np_data2
        self.opt2_total_data_num.set(' (max='+str(self.get_total_data_num())+')') # データ点総数を更新

        # 描画
        for x, y in np_data1:
            self.dot_point_A(x, y)
        for x, y in np_data2:
            self.dot_point_B(x, y)

    # 保存ボタン用：描画されたデータ点をデータセットとして保存するメソッド
    def save_data(self):
        filename = tkFileDialog.asksaveasfilename(filetypes=[("npzファイル","*.npz")], initialdir="./dataset/")
        if filename == '':
            return # キャンセル
        self.print_log('save:', filename)

        # データセットを保存
        np_data1 = self.transform_coordinate_system_from_canvas(self.data1) # 座標系を通常の座標に変換
        np_data2 = self.transform_coordinate_system_from_canvas(self.data2) # 座標系を通常の座標に変換
        np.savez(filename, data1=np_data1, data2=np_data2)

    # クリアボタン用：描画されたデータ点をすべて消すメソッド
    def clear_data(self):
        self.print_log('clear all')
        self.init_canvas()

    # 学習開始ボタン用：カーネルパーセプトロンの学習を開始するメソッド
    def start_learning(self):
        # 総データ数・指定データ数が異常でないかを確認
        if self.get_total_data_num() == 0:
            return # キャンセル
        try:
            specified_data_num = int(self.opt2_specified_data_num.get())
            if specified_data_num <= 0:
                raise Exception()
        except Exception:
            self.print_log('ERROR: データ数の値が異常です')
            return # キャンセル

        # 指定データ数に総データ数を変更

        self.print_log('サンプリングしました')
        data1 = np.random.permutation(self.data1)        # データをシャッフル
        data2 = np.random.permutation(self.data2)        # データをシャッフル


        # ログを出力
        self.print_log('data num:',
                        'total =',      self.opt2_specified_data_num.get() + ',',
                        '(A, B) =',     str((len(self.data1),len(self.data2))))
        self.print_log('option:',
                        'features =',   self.opt1_feature.get() + ',',
                        'test =',       self.opt3_test_ratio.get() + ',',
                        'lr =',         self.opt4_lr.get() + ',',
                        'resolution =', self.opt5_resolution.get())
        self.print_log('Learning Start')

        # 座標系を通常の座標に変換
        np_data1 = self.transform_coordinate_system_from_canvas(self.data1)
        np_data2 = self.transform_coordinate_system_from_canvas(self.data2)

        # カーネルパーセプトロンのプログラムを実行
        if self.opt1_feature.get() == 'gauss':
            try:
                sigma2 = float(self.opt1_featureGauss_sigma2.get())
                self.print_log('option: sigma =', self.opt1_featureGauss_sigma2.get())
            except Exception:
                self.print_log('ERROR: σ^2の値が異常です')
                return # キャンセル
            kp.main(
                np_data1, np_data2,
                kernel_type='gauss',
                epsilon=float(self.opt4_lr.get()),
                test_ratio=float(self.opt3_test_ratio.get()),
                resolution=int(self.opt5_resolution.get()),
                sigma2=sigma2
            )
        elif self.opt1_feature.get() == '2d':
            kp.main(
                np_data1, np_data2,
                kernel_type='nothing',
                epsilon=float(self.opt4_lr.get()),
                test_ratio=float(self.opt3_test_ratio.get()),
                resolution=int(self.opt5_resolution.get())
            )

    ###########################################################################

    # データの総数を求めるメソッド
    def get_total_data_num(self):
        return len(self.data1) + len(self.data2)

    # 普通の座標からキャンバス用の座標に変換する
    def transform_coordinate_system_for_canvas(self, points):
        canvas_width  = self.canvas_size
        canvas_height = self.canvas_size
        data_range   = {'x':{'min':-1, 'max':1           }, 'y':{'min':-1, 'max':1            } }
        canvas_range = {'x':{'min':0,  'max':canvas_width}, 'y':{'min':0,  'max':canvas_height} }
        x = points.T[0]; y = points.T[1]

        x = (x - data_range['x']['min']) / (data_range['x']['max'] - data_range['x']['min'])   # [0,1]区間へ正規化
        x = x*(canvas_range['x']['max'] - canvas_range['x']['min']) + canvas_range['x']['min'] # 描画範囲へスケーリングとシフト

        y = (y - data_range['y']['min']) / (data_range['y']['max'] - data_range['y']['min'])   # [0,1]区間へ正規化
        y = (y * -1) + 1                                                                       # 軸反転
        y = y*(canvas_range['y']['max'] - canvas_range['y']['min']) + canvas_range['y']['min'] # 描画範囲へスケーリングとシフト

        return np.vstack((x, y)).T

    # キャンバス用の座標から普通の座標に変換する
    def transform_coordinate_system_from_canvas(self, points):
        canvas_width  = self.canvas_size
        canvas_height = self.canvas_size
        data_range   = {'x':{'min':-1, 'max':1           }, 'y':{'min':-1, 'max':1            } }
        canvas_range = {'x':{'min':0,  'max':canvas_width}, 'y':{'min':0,  'max':canvas_height} }
        x = points.T[0]; y = points.T[1]

        x = (x - canvas_range['x']['min']) / (canvas_range['x']['max'] - canvas_range['x']['min']) # [0,1]区間へ正規化
        x = x*(data_range['x']['max'] - data_range['x']['min']) + data_range['x']['min']           # 描画範囲へスケーリングとシフト

        y = (y - canvas_range['y']['min']) / (canvas_range['y']['max'] - canvas_range['y']['min']) # [0,1]区間へ正規化
        y = (y * -1) + 1                                                                           # 軸反転
        y = y*(data_range['y']['max'] - data_range['y']['min']) + data_range['y']['min']           # 描画範囲へスケーリングとシフト

        return np.vstack((x, y)).T


class entry_dialog:
    def __init__(self, root, msg, init_val=None):
        self.win = tk.Tk()
        win = self.win
        win.title('入力フォーム')                # ウィンドウタイトルの設定
        # win.geometry('200x200')                 # ウィンドウサイズを設定
        win.resizable(0,0)                      # ウィンドウサイズの変更を不可に設定

        label = tk.Label(win, text=msg)         # ラベルを生成
        label.pack(padx=10, pady=8)             # ラベルを1段目に設置、padding設定
        self.entry = tk.Entry(win)              # 入力フォームを生成
        self.entry.pack(padx=10, pady=8)        # 入力フォームを2段目に設置、padding設定
        btn = ttk.Button(win)                   # ボタンを生成
        btn.configure(text='Enter', command=lambda:self.enter()) # ボタンの各種設定
        btn.pack(padx=10, pady=8)               # ボタンを3段目に設置、padding設定

        win.mainloop()

    def enter(self):
        self.val = self.entry.get()
        self.win.quit()

    def get(self):
        return self.val

    def tmp(self):
        # データ数を選択
        ed = entry_dialog(self.win, 'OK???')
        print(ed.get())
        print('aaaaa')

if __name__ == '__main__':
    kp = kernel_perceptron_learner()
