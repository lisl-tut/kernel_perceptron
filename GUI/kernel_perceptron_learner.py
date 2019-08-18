import numpy as np

import tkinter as tk                            # tk本体
import tkinter.ttk as ttk                       # スタイル付きtk
from tkinter import filedialog as tkFileDialog  # ファイルダイアログボックス
from PIL import Image, ImageTk

import kernel_perceptron as kp

class kernel_perceptron_learner():
    def __init__(self):
        #### #### tkウィンドウの設定 #### ####
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

        # キャンバス
        self.canvas_size = 500
        self.canvas = tk.Canvas(fr_canvas, width=self.canvas_size, height=self.canvas_size, bg='white')
        self.canvas.bind('<Button-1>', lambda event:self.click_left(event))   # 左クリックのイベントを設定
        self.canvas.bind('<Button-3>', lambda event:self.click_right(event))  # 右クリックのイベントを設定
        self.canvas.pack(padx=5, pady=5)

        # 点の描画タイプ設定のラジオボタン
        fr_p_type = tk.Frame(fr_canvas)
        fr_p_type.pack(padx=5, pady=2, fill='x')
        p_type_label = tk.Label(fr_p_type, text=' 描画タイプ ：')
        p_type_label.grid(row=1, column=1)
        self.p_type = tk.StringVar()                                    # 点の描画タイプの取得用変数
        self.p_type.set('cat_and_dog')
        p_type_rbtn1 = ttk.Radiobutton(fr_p_type, variable=self.p_type, value='point', text='点のみ',
                                                    command=lambda:self.change_dot_type_to_NormalPoint())
        p_type_rbtn1.grid(row=1, column=2)
        p_type_rbtn2 = ttk.Radiobutton(fr_p_type, variable=self.p_type, value='cat_and_dog', text='犬と猫',
                                                    command=lambda:self.change_dot_type_to_CatAndDog())
        p_type_rbtn2.grid(row=1, column=3)

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

        # オプション3
        fr_opt3 = tk.LabelFrame(fr_opt, relief='flat', text='【ラーニングレート】',)
        fr_opt3.pack(padx=5, pady=4, fill='x')
        self.opt3_lr = tk.StringVar()                                   # ラーニングレートの取得用変数
        self.opt3_lr.set('0.10')
        opt3_rbtn1 = ttk.Radiobutton(fr_opt3, variable=self.opt3_lr, value='0.50', text='0.50')
        opt3_rbtn1.grid(row=1, column=1)
        opt3_rbtn2 = ttk.Radiobutton(fr_opt3, variable=self.opt3_lr, value='0.10', text='0.10')
        opt3_rbtn2.grid(row=1, column=2)
        opt3_rbtn3 = ttk.Radiobutton(fr_opt3, variable=self.opt3_lr, value='0.01', text='0.01')
        opt3_rbtn3.grid(row=1, column=3)

        # オプション4
        fr_opt4 = tk.LabelFrame(fr_opt, relief='flat', text='【テストデータの割合】',)
        fr_opt4.pack(padx=5, pady=4, fill='x')
        self.opt4_test_ratio = tk.StringVar()                           # テストデータの割合の取得用変数
        self.opt4_test_ratio.set('0.25')
        opt4_rbtn1 = ttk.Radiobutton(fr_opt4, variable=self.opt4_test_ratio, value='0.1', text='10％')
        opt4_rbtn1.grid(row=1, column=1)
        opt4_rbtn2 = ttk.Radiobutton(fr_opt4, variable=self.opt4_test_ratio, value='0.25', text='25％')
        opt4_rbtn2.grid(row=1, column=2)
        opt4_rbtn3 = ttk.Radiobutton(fr_opt4, variable=self.opt4_test_ratio, value='0.5', text='50％')
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

        # オプション6
        fr_opt6 = tk.LabelFrame(fr_opt, relief='flat', text='【乱数】',)
        fr_opt6.pack(padx=5, pady=4, fill='x')
        self.opt6_random_seed = tk.StringVar()                          # 乱数の取得用変数
        self.opt6_random_seed.set('fixed')
        opt6_rbtn1 = ttk.Radiobutton(fr_opt6, variable=self.opt6_random_seed, value='fixed', text='無効')
        opt6_rbtn1.grid(row=1, column=1)
        opt6_rbtn2 = ttk.Radiobutton(fr_opt6, variable=self.opt6_random_seed, value='auto', text='有効')
        opt6_rbtn2.grid(row=1, column=2)

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

        #### #### すべての設定終了後の操作 #### ####
        self.init_canvas()      # キャンバス初期化，データセット初期化

        # 点の描画用画像の読み込み
        img_dog = Image.open("./image/dog.png")
        img_dog = img_dog.resize((35, 35))
        self.img_dog = ImageTk.PhotoImage(img_dog)
        img_cat = Image.open("./image/cat.png")
        img_cat = img_cat.resize((35, 35))
        self.img_cat = ImageTk.PhotoImage(img_cat)

        # 点の描画タイプを設定
        if self.p_type.get() == 'cat_and_dog':
            self.dot_A = self.dot_A_dog
            self.dot_B = self.dot_B_cat
        else:
            self.dot_A = self.dot_A_point
            self.dot_B = self.dot_B_point

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
        self.data1 = np.empty((0,2), int)
        self.data2 = np.empty((0,2), int)

        # グリッド線の描画
        self.draw_grid()

    # キャンバスにグリッド線を描画するメソッド
    def draw_grid(self):
        center = self.canvas_size / 2
        for i in range(0, self.canvas_size, int(self.canvas_size/10)):
            if i != center:
                self.canvas.create_line(0, i, self.canvas_size, i, fill='grey')
                self.canvas.create_line(i, 0, i, self.canvas_size, fill='grey')
            else:
                self.canvas.create_line(0, i, self.canvas_size, i, fill='black')
                self.canvas.create_line(i, 0, i, self.canvas_size, fill='black')

    # キャンバスに点（タイプA）を打つメソッド
    def dot_A_point(self, x, y):
        self.canvas.create_oval(x-5, y-5, x+5, y+5, fill='red', width=0)

    # キャンバスに点（タイプB）を打つメソッド
    def dot_B_point(self, x, y):
        self.canvas.create_oval(x-5, y-5, x+5, y+5, fill='blue', width=0)

    # キャンバスに点（タイプA:犬）を打つメソッド
    def dot_A_dog(self, x, y):
        self.canvas.create_image(x, y, image=self.img_dog)

    # キャンバスに点（タイプB:猫）を打つメソッド
    def dot_B_cat(self, x, y):
        self.canvas.create_image(x, y, image=self.img_cat)

    # キャンバスの内容を再描画するメソッド
    def redraw(self):
        self.canvas.delete("all")   # 描画されているものをすべて削除
        self.draw_grid()            # グリッド線を描画
        for x, y in self.data1:     # data1を描画
            self.dot_A(x, y)
        for x, y in self.data2:     # data2を描画
            self.dot_B(x, y)

    ###########################################################################

    # 点の描画タイプを普通の点に変更するメソッド
    def change_dot_type_to_NormalPoint(self):
        self.dot_A = self.dot_A_point
        self.dot_B = self.dot_B_point
        self.redraw()                   # キャンバスを再描画

    # 点の描画タイプを犬と猫に変更するメソッド
    def change_dot_type_to_CatAndDog(self):
        self.dot_A = self.dot_A_dog
        self.dot_B = self.dot_B_cat
        self.redraw()                   # キャンバスを再描画

    # 左クリック用：データ点（タイプA）を描画するメソッド
    def click_left(self, event):
        self.print_log('dot A:', str([event.x, event.y]))
        self.dot_A(event.x, event.y)                                        # 点を描画
        self.data1 = np.append(self.data1, [[event.x,event.y]], axis=0)     # 点をデータベースに追加

    # 右クリック用：データ点（タイプB）を描画するメソッド
    def click_right(self, event):
        self.print_log('dot B:', str([event.x, event.y]))
        self.dot_B(event.x, event.y)                                        # 点を描画
        self.data2 = np.append(self.data2, [[event.x,event.y]], axis=0)     # 点をデータベースに追加

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

        # 描画
        for x, y in np_data1:
            self.dot_A(x, y)
        for x, y in np_data2:
            self.dot_B(x, y)

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
        # 総データ数が異常でないかを確認
        if len(self.data1) + len(self.data2) == 0:
            return # キャンセル

        # ログを出力
        self.print_log('---------- Learning Start ----------')
        self.print_log('data num:',
                        'total =',      str(len(self.data1)+len(self.data2)) + ',',
                        '(A, B) =',     str((len(self.data1),len(self.data2))))
        self.print_log('option:',
                        'features =',   self.opt1_feature.get() + ',',
                        'lr =',         self.opt3_lr.get() + ',',
                        'test =',       self.opt4_test_ratio.get() + ',',
                        'resolution =', self.opt5_resolution.get() + ',',
                        'random seed =', self.opt6_random_seed.get())

        # 座標系を通常の座標に変換
        np_data1 = self.transform_coordinate_system_from_canvas(self.data1)
        np_data2 = self.transform_coordinate_system_from_canvas(self.data2)

        # 引数を指定
        kwargs = {
            'data1':np_data1,
            'data2':np_data2,
            'epsilon':float(self.opt3_lr.get()),
            'test_ratio':float(self.opt4_test_ratio.get()),
            'resolution':int(self.opt5_resolution.get()),
            'random_seed':self.opt6_random_seed.get()
        }

        # カーネルパーセプトロンのプログラムを実行
        if self.opt1_feature.get() == 'gauss':
            try:
                sigma2 = float(self.opt1_featureGauss_sigma2.get())
                self.print_log('option: sigma =', self.opt1_featureGauss_sigma2.get())
            except Exception:
                self.print_log('ERROR: σ^2の値が異常です')
                return # キャンセル
            kwargs['kernel_type'] = 'gauss'
            kwargs['sigma2'] = sigma2
            kp.main(**kwargs)                   # 実行

        elif self.opt1_feature.get() == '2d':
            kwargs['kernel_type'] = 'nothing'
            kp.main(**kwargs)                   # 実行

        # ログを出力
        self.print_log('==== ==== Learning Finished ==== ====')

    ###########################################################################

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


if __name__ == '__main__':
    kp = kernel_perceptron_learner()