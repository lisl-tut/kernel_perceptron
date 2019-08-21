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
        win.option_add('*Font', 'TkDefaultFont 12')
        ttk.Style().configure('TRadiobutton', font=('TkDefaultFont', 12), bg='white')
        ttk.Style().configure('TButton', font=('TkDefaultFont', 12), bg='white')

        # 左・中・右のフレーム
        fr_l = tk.Frame(win)
        fr_l.pack(padx=2, pady=5, fill='y', side='left')
        fr_c = tk.Frame(win)
        fr_c.pack(padx=2, pady=5, fill='y', side='left')
        fr_r = tk.Frame(win)
        fr_r.pack(padx=2, pady=5, fill='y', side='left')

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
        self.canvas_size = 400
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

        #### #### 中央 #### ####
        # パーセプトロンの図用のフレーム
        fr_figure = tk.LabelFrame(fr_c, bd=2, relief='groove', text='パーセプトロン（手動調節）')
        fr_figure.pack(padx=5, pady=2, fill='x')

        # パーセプトロンの図
        self.figure = tk.Canvas(fr_figure, width=400, height=300, bd=2, relief='ridge', bg='white')
        self.figure.pack(padx=5, pady=2)

        # スライダー
        fr_sliders = tk.Frame(fr_figure)
        fr_sliders.pack(padx=5, pady=8, fill='x')

        # スライダー a
        self.slider_a_label_val = tk.StringVar()                                            # ラベルの変数
        self.slider_a_label_val.set('    a = +0.00 ：  ')
        self.slider_a_label = tk.Label(fr_sliders, textvariable=self.slider_a_label_val)    # ラベル
        self.slider_a_label.grid(row=1, column=1, sticky=tk.E)
        self.slider_a_val = tk.DoubleVar()                                                  # スライダーの変数
        self.slider_a = ttk.Scale(fr_sliders, variable=self.slider_a_val,
                                orient='horizontal', length=250, from_=-1, to=1,
                                command=lambda v: self.slider_a_label_val.set('    a = %+.2f ：  ' % float(v))
                            )                                                               # スライダー
        self.slider_a.grid(row=1, column=2)

        # スライダー b
        self.slider_b_label_val = tk.StringVar()                                            # ラベルの変数
        self.slider_b_label_val.set('    b = +0.00 ：  ')
        self.slider_b_label = tk.Label(fr_sliders, textvariable=self.slider_b_label_val)    # ラベル
        self.slider_b_label.grid(row=2, column=1, sticky=tk.E)
        self.slider_b_val = tk.DoubleVar()                                                  # スライダーの変数
        self.slider_b = ttk.Scale(fr_sliders, variable=self.slider_b_val,
                                orient='horizontal', length=250, from_=-1, to=1,
                                command=lambda v: self.slider_b_label_val.set('    b = %+.2f ：  ' % float(v))
                            )                                                               # スライダー
        self.slider_b.grid(row=2, column=2)

        # スライダー c
        self.slider_c_label_val = tk.StringVar()                                            # ラベルの変数
        self.slider_c_label_val.set('    c = +0.00 ：  ')
        self.slider_c_label = tk.Label(fr_sliders, textvariable=self.slider_c_label_val)    # ラベル
        self.slider_c_label.grid(row=3, column=1, sticky=tk.E)
        self.slider_c_val = tk.DoubleVar()                                                  # スライダーの変数
        self.slider_c = ttk.Scale(fr_sliders, variable=self.slider_c_val,
                                orient='horizontal', length=250, from_=-1, to=1,
                                command=lambda v: self.slider_c_label_val.set('    c = %+.2f ：  ' % float(v))
                            )                                                               # スライダー
        self.slider_c.grid(row=3, column=2)

        # スライダー d
        self.slider_d_label_val = tk.StringVar()                                            # ラベルの変数
        self.slider_d_label_val.set('    d = +0.00 ：  ')
        self.slider_d_label = tk.Label(fr_sliders, textvariable=self.slider_d_label_val)    # ラベル
        self.slider_d_label.grid(row=4, column=1, sticky=tk.E)
        self.slider_d_val = tk.DoubleVar()                                                  # スライダーの変数
        self.slider_d = ttk.Scale(fr_sliders, variable=self.slider_d_val,
                                orient='horizontal', length=250, from_=-1, to=1,
                                command=lambda v: self.slider_d_label_val.set('    d = %+.2f ：  ' % float(v))
                            )                                                               # スライダー
        self.slider_d.grid(row=4, column=2)

        # 更新
        self.update_btn = ttk.Button(fr_figure)
        self.update_btn.configure(text='更新', width=10, command=lambda:self.update_canvas_and_figure())
        self.update_btn.pack(padx=5, pady=10)

        #### #### 右側 #### ####
        # オプション設定
        fr_opt = tk.LabelFrame(fr_r, bd=2, relief='groove', text='学習オプション')
        fr_opt.pack(padx=5, pady=2, fill='x')

        # オプション1
        fr_opt1 = tk.LabelFrame(fr_opt, relief='flat', text='【特徴量変換】',)
        fr_opt1.pack(padx=5, pady=4, fill='x')
        self.opt1_feature = tk.StringVar()                              # 特徴量の取得用変数
        self.opt1_feature.set('2d')

        opt1_rbtn1 = ttk.Radiobutton(fr_opt1, variable=self.opt1_feature,
                                        value='2d', text='2個 + バイアス',
                                        command=lambda:self.enable_feature2D())
        opt1_rbtn1.grid(row=1, column=1, sticky=tk.W)
        opt1_feature2D_label = tk.Label(fr_opt1, text=' 式 = ')         # 特徴量の式のラベル
        opt1_feature2D_label.grid(row=1, column=2, sticky=tk.E)
        self.opt1_feature2D_formula = tk.Entry(fr_opt1)                 # 特徴量の式
        self.opt1_feature2D_formula.configure(width=10)
        self.opt1_feature2D_formula.grid(row=1, column=3)

        opt1_rbtn2 = ttk.Radiobutton(fr_opt1, variable=self.opt1_feature,
                                        value='3d', text='3個 + バイアス',
                                        command=lambda:self.enable_feature3D())
        opt1_rbtn2.grid(row=2, column=1, sticky=tk.W)
        opt1_feature3D_label = tk.Label(fr_opt1, text=' 式 = ')         # 特徴量の式のラベル
        opt1_feature3D_label.grid(row=2, column=2, sticky=tk.E)
        self.opt1_feature3D_formula = tk.Entry(fr_opt1)                 # 特徴量の式
        self.opt1_feature3D_formula.configure(width=10)
        self.opt1_feature3D_formula.grid(row=2, column=3)

        opt1_rbtn3 = ttk.Radiobutton(fr_opt1, variable=self.opt1_feature,
                                        value='gauss', text='カーネル法 ： ガウス',
                                        command=lambda:self.enable_featureGauss())
        opt1_rbtn3.grid(row=3, column=1, sticky=tk.W)
        opt1_featureGauss_sigma2 = tk.Label(fr_opt1, text='   σ^2 = ')  # 特徴量の式のラベル
        opt1_featureGauss_sigma2.grid(row=3, column=2, sticky=tk.E)
        self.opt1_featureGauss_sigma2 = tk.Entry(fr_opt1)               # 特徴量の式
        self.opt1_featureGauss_sigma2.configure(width=10)
        self.opt1_featureGauss_sigma2.grid(row=3, column=3)

        opt1_feature2D_label.configure(state='disabled')                #### 無効化
        self.opt1_feature2D_formula.configure(state='disabled')         #### 無効化
        opt1_rbtn2.configure(state='disabled')                          #### 無効化
        opt1_feature3D_label.configure(state='disabled')                #### 無効化
        self.opt1_feature3D_formula.configure(state='disabled')         #### 無効化

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
        start_btn.configure(text='学習開始（自動）', width=20, command=lambda:self.start_learning())
        start_btn.pack(padx=5, pady=12)

        # ログ画面
        fr_log = tk.LabelFrame(fr_r, bd=2, relief='flat', text='ログ')
        fr_log.pack(padx=5, pady=5, fill='x')
        self.logbox = tk.Listbox(fr_log, font=('TkDefaultFont', 10))
        self.logbox.configure(height=10, width=55)
        self.logbox.pack(padx=5, pady=5)

        #### #### すべての設定終了後の操作 #### ####
        self.init_canvas()  # キャンバス初期化，データセット初期化

        # 点の描画用画像の読み込み
        dot_size = int(self.canvas_size * 0.07)
        img_dog = Image.open('./image/dog.png')
        img_dog = img_dog.resize((dot_size, dot_size))
        self.img_dog = ImageTk.PhotoImage(img_dog)
        img_cat = Image.open('./image/cat.png')
        img_cat = img_cat.resize((dot_size, dot_size))
        self.img_cat = ImageTk.PhotoImage(img_cat)

        # 点の描画タイプを設定
        if self.p_type.get() == 'cat_and_dog':
            self.dot_A = self.dot_A_dog
            self.dot_B = self.dot_B_cat
        else: # self.p_type.get() == 'point'
            self.dot_A = self.dot_A_point
            self.dot_B = self.dot_B_point

        # パーセプトロンの描画
        if self.opt1_feature.get() == '2d':
            self.enable_feature2D()
        elif self.opt1_feature.get() == '3d':
            self.enable_feature3D()
        elif self.opt1_feature.get() == 'gauss':
            self.enable_featureGauss()
        else:
            raise Exception('feature is not proper')

        win.mainloop()          # ループに入る

    ###########################################################################

    # キャンバスをクリアしてグリッドを描画するメソッド
    def init_canvas(self):
        # 描画されているものをすべて削除
        self.canvas.delete('all')

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
        self.canvas.delete('all')   # 描画されているものをすべて削除
        self.draw_back()            # バックの色を描画
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
        filename = tkFileDialog.asksaveasfilename(filetypes=[('npzファイル','*.npz')], initialdir='./dataset/')
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
        self.disc_func = None           # 識別関数の設定を消去
        self.draw_back = lambda : None  # キャンバスの背面描画の関数を登録(何もしない関数)
        self.init_canvas()              # キャンバスを再度初期化

    ###########################################################################

    # 更新ボタン用：スライダーの値を読み込み、それをデータキャンバスとパーセプトロンの図に反映させるメソッド
    def update_canvas_and_figure(self):
        if self.opt1_feature.get() == '2d':
            # パーセプトロンの図の現在の入力線を消す
            self.figure.delete(self.figure_line_in1)
            self.figure.delete(self.figure_line_in2)
            self.figure.delete(self.figure_line_in3)

            # スライダーからパラメータを取得
            a = self.slider_a_val.get()
            b = self.slider_b_val.get()
            c = self.slider_c_val.get()

            # パラメータに応じてパーセプトロンの図の入力線を引く
            self.figure_line_in1 = self.draw_perceptron_input_line(70,  80, 190, 140, a)
            self.figure_line_in2 = self.draw_perceptron_input_line(70, 150, 190, 150, b)
            self.figure_line_in3 = self.draw_perceptron_input_line(70, 220, 190, 160, c)

            # 識別関数を背景に描画するための関数を登録し、再描画
            if (a, b, c) == (0, 0, 0):
                self.disc_func = None           # 識別関数の設定を消去
                self.redraw()                   # キャンバスの再描画
            else:
                def disc_func(x, y):
                    val = np.array(a*x+b*y+c)
                    val[val>=0] = 1             # if val >= 0, label is  1
                    val[val<0] = -1             # if val <  0, label is -1
                    return val
                self.disc_func = disc_func
                self.set_draw_back_function()   # 識別関数を描画するように登録
                self.redraw()                   # キャンバスの再描画

        elif self.opt1_feature.get() == '3d':
            raise Exception('feature 3d is not implemented')

        elif self.opt1_feature.get() == 'gauss':
            if self.disc_func == None:
                self.print_log('ERROR: 学習をまだ行っていません')
                return # キャンセル
            self.set_draw_back_function()       # 識別関数を描画するように登録
            self.redraw()                       # キャンバスの再描画

        else:
            raise Exception('feature is not proper')

        # 最後にテストを実行
        self.test_disc_function()

    # パーセプトロンの入力線を書くためのメソッド
    def draw_perceptron_input_line(self, x1, y1, x2, y2, param):
        if param == 0:
            return self.figure.create_line(x1, y1, x2, y2, fill='grey', width=2)
        elif param > 0:
            return self.figure.create_line(x1, y1, x2, y2, fill='red', width=int(param*10))
        else:
            return self.figure.create_line(x1, y1, x2, y2, fill='blue', width=int(abs(param)*10))

    # 識別関数を背景描画の関数として登録するメソッド
    def set_draw_back_function(self):
        if self.disc_func == None:
            raise Exception('disc_func is not registed')

        # 指定されている解像度を取り出す
        resolution = int(self.opt5_resolution.get())

        # 識別関数の値を計算
        x = np.linspace(-1, 1, resolution)
        y = np.linspace(-1, 1, resolution)
        X, Y = np.meshgrid(x, -y)           # y軸がcanvasでは反転している関係で、ここでyを反転
        f = self.disc_func(X, Y)

        # 背景描画用の関数を設計し、それを登録
        def draw_back():
            box_size = self.canvas_size / resolution
            for i in range(resolution):
                for j in range(resolution):
                    if f[j][i] >= 0:
                        color = 'yellow'
                    else:
                        color = 'purple'
                    self.canvas.create_rectangle(i*box_size, j*box_size, (i+1)*box_size, (j+1)*box_size, fill=color, width=0)
        self.draw_back = draw_back  # 作成した関数を登録

    # 識別関数の精度を計算して表示するメソッド
    def test_disc_function(self):
        if self.disc_func == None:
            self.print_log('学習をまだ行っていません')
            return # キャンセル

        # 座標系を通常の座標に変換
        np_data1 = self.transform_coordinate_system_from_canvas(self.data1)
        np_data2 = self.transform_coordinate_system_from_canvas(self.data2)

        # Accuracyを計算
        count = 0
        for x, y in np_data1:
            if self.disc_func(x, y) * -1 > 0:
                count += 1
        for x, y in np_data2:
            if self.disc_func(x, y) * +1 > 0:
                count += 1
        total = len(self.data1) + len(self.data2)
        if total == 0:
            self.print_log(r'Accuracy: 100.00%')
        else:
            self.print_log('Accuracy: %.2f%%' % (count/total*100))

    ###########################################################################

    # 特徴量のラジオボタン用：特徴量2個を選択したときに、パーセプトロンの図を書き直すメソッド
    def enable_feature2D(self):
        # 図を全て削除
        self.figure.delete('all')

        # 識別関数・描画関数をクリア
        self.disc_func = None           # 識別関数の設定を消去
        self.draw_back = lambda : None  # キャンバスの背面描画の関数を登録(何もしない関数)
        self.redraw()

        # パーセプトロンを描画
        self.figure.create_oval(190, 120, 250, 180, outline='grey', width=2)                    # 本体の丸
        self.figure_line_in1 = self.figure.create_line(70,  80, 190, 140, fill='grey', width=2) # 入力の線
        self.figure_line_in2 = self.figure.create_line(70, 150, 190, 150, fill='grey', width=2) # 入力の線
        self.figure_line_in3 = self.figure.create_line(70, 220, 190, 160, fill='grey', width=2) # 入力の線
        self.figure.create_line(250, 150, 320, 150, fill='grey', width=2)                       # 出力の線
        self.figure.create_oval(65,  75, 75,  85, fill='green', width=0)                        # 入力の端の点
        self.figure.create_oval(65, 145, 75, 155, fill='green', width=0)                        # 入力の端の点
        self.figure.create_oval(65, 215, 75, 225, fill='green', width=0)                        # 入力の端の点
        self.figure.create_oval(315, 145, 325, 155, fill='green', width=0)                      # 出力の端の点
        self.figure.create_text(50,  80, text='x', fill='green', font=('Purisa', 20))           # 入力の文字
        self.figure.create_text(50, 150, text='y', fill='green', font=('Purisa', 20))           # 入力の文字
        self.figure.create_text(50, 220, text='1', fill='green', font=('Purisa', 20))           # 入力の文字
        self.figure.create_text(320, 130, text='σ(ax+bx+c)', fill='green', font=('Purisa', 18)) # 出力の文字
        self.figure.create_text(130,  95, text='a', fill='#FFA500', font=('Purisa', 20))        # パラメータ
        self.figure.create_text(130, 135, text='b', fill='#FFA500', font=('Purisa', 20))        # パラメータ
        self.figure.create_text(130, 175, text='c', fill='#FFA500', font=('Purisa', 20))        # パラメータ

        # スライダーの文字を切り替え
        self.slider_a_val.set(0)
        self.slider_b_val.set(0)
        self.slider_c_val.set(0)
        self.slider_d_val.set(0)
        self.slider_a_label_val.set('    a = +0.00 ：  ')
        self.slider_b_label_val.set('    b = +0.00 ：  ')
        self.slider_c_label_val.set('    c = +0.00 ：  ')
        self.slider_d_label_val.set('    d = +0.00 ：  ')

        # 図に関する動作をすべて有効化
        self.slider_a_label.configure(state='active')
        self.slider_b_label.configure(state='active')
        self.slider_c_label.configure(state='active')
        self.slider_d_label.configure(state='disabled')
        self.slider_a.configure(state='active')
        self.slider_b.configure(state='active')
        self.slider_c.configure(state='active')
        self.slider_d.configure(state='disabled')
        self.update_btn.configure(state='active')

    # 特徴量のラジオボタン用：特徴量3個を選択したときに、パーセプトロンの図を書き直すメソッド
    def enable_feature3D(self):
        pass

    # 特徴量のラジオボタン用：ガウスカーネルのカーネル法を選択したときに、パーセプトロンの図を書き直すメソッド
    def enable_featureGauss(self):
        # 図を全て削除
        self.figure.delete('all')

        # 識別関数・描画関数をクリア
        self.disc_func = None           # 識別関数の設定を消去
        self.draw_back = lambda : None  # キャンバスの背面描画の関数を登録(何もしない関数)
        self.redraw()

        # パーセプトロンを描画
        self.figure.create_oval(190, 120, 250, 180, outline='grey', width=2)                    # 本体の丸
        self.figure.create_rectangle(100, 90, 160, 210, outline='grey', width=2)                # 特徴変換器の四角
        self.figure.create_line(70, 115, 100, 115, fill='grey', width=2)                        # 入力の線
        self.figure.create_line(70, 185, 100, 185, fill='grey', width=2)                        # 入力の線
        self.figure.create_line(160,  98, 200, 125, fill='grey', width=2)                       # φの線
        self.figure.create_line(160, 113, 196, 131, fill='grey', width=2)                       # φの線
        self.figure.create_line(160, 128, 192, 138, fill='grey', width=2)                       # φの線
        self.figure.create_line(160, 143, 190, 146, fill='grey', width=2)                       # φの線
        self.figure.create_line(160, 157, 190, 154, fill='grey', width=2)                       # φの線
        self.figure.create_line(160, 172, 192, 162, fill='grey', width=2)                       # φの線
        self.figure.create_text(175, 182, text='︙', fill='grey', font=('Purisa', 14))          # ︙の文字
        self.figure.create_line(160, 202, 200, 175, fill='grey', width=2)                       # φの線
        self.figure.create_line(250, 150, 320, 150, fill='grey', width=2)                       # 出力の線
        self.figure.create_oval(65, 110, 75, 120, fill='green', width=0)                        # 入力の端の点
        self.figure.create_oval(65, 180, 75, 190, fill='green', width=0)                        # 入力の端の点
        self.figure.create_oval(315, 145, 325, 155, fill='green', width=0)                      # 出力の端の点
        self.figure.create_text(50, 115, text='x', fill='green', font=('Purisa', 20))           # 入力の文字
        self.figure.create_text(50, 185, text='y', fill='green', font=('Purisa', 20))           # 入力の文字
        self.figure.create_text(320, 130, text='σ(Σw_i・φ_i)', fill='green', font=('Purisa', 17))  # 出力の文字
        self.figure.create_text(130, 140, text='特徴量\n変換器', fill='green', font=('Purisa', 14)) # 特徴量変換器の文字
        self.figure.create_text(130, 170, text='φ', fill='green', font=('Purisa', 17))          # φの文字
        self.figure.create_text(175, 150, text='w', fill='#FFA500', font=('Purisa', 20))        # パラメータ

        # スライダーの文字を切り替え
        self.slider_a_val.set(0)
        self.slider_b_val.set(0)
        self.slider_c_val.set(0)
        self.slider_d_val.set(0)
        self.slider_a_label_val.set('  w_1 = +0.00 ：  ')
        self.slider_b_label_val.set('  w_2 = +0.00 ：  ')
        self.slider_c_label_val.set('  w_3 = +0.00 ：  ')
        self.slider_d_label_val.set('  . . . = +0.00 ：  ')

        # 図に関する動作をすべて無効化
        self.slider_a_label.configure(state='disabled')
        self.slider_b_label.configure(state='disabled')
        self.slider_c_label.configure(state='disabled')
        self.slider_d_label.configure(state='disabled')
        self.slider_a.configure(state='disabled')
        self.slider_b.configure(state='disabled')
        self.slider_c.configure(state='disabled')
        self.slider_d.configure(state='disabled')
        self.update_btn.configure(state='active')   # 更新だけできるようにしておく

    ###########################################################################

    # 学習開始ボタン用：カーネルパーセプトロンの学習を開始するメソッド
    def start_learning(self):
        # 総データ数が異常でないかを確認
        if len(self.data1) + len(self.data2) == 0:
            self.print_log('ERROR: データキャンバスにデータを入力してください')
            return # キャンセル

        # ログを出力
        self.print_log('------------------- Learning Start -------------------')
        self.print_log('dot type =',    self.p_type.get())
        self.print_log('data num:',
                        'total =',      str(len(self.data1)+len(self.data2)) + ',',
                        '(A, B) =',     str((len(self.data1),len(self.data2))) )
        self.print_log('option:',
                        'features =',   self.opt1_feature.get() + ',',
                        'test =',       self.opt4_test_ratio.get() + ',',
                        'resolution =', self.opt5_resolution.get() + ',',
                        'random seed =', self.opt6_random_seed.get() )

        # 座標系を通常の座標に変換
        np_data1 = self.transform_coordinate_system_from_canvas(self.data1)
        np_data2 = self.transform_coordinate_system_from_canvas(self.data2)

        # 引数を指定
        kwargs = {
            'data1':np_data1,
            'data2':np_data2,
            'test_ratio':float(self.opt4_test_ratio.get()),
            'resolution':int(self.opt5_resolution.get()),
            'random_seed':self.opt6_random_seed.get(),
            'dot_type':self.p_type.get()
        }

        # カーネルパーセプトロンのプログラムを実行
        if self.opt1_feature.get() == 'gauss':
            try:
                sigma2 = float(self.opt1_featureGauss_sigma2.get())
                self.print_log('option: sigma =', self.opt1_featureGauss_sigma2.get())
            except Exception:
                self.print_log('ERROR: σ^2の値が異常です')
                return # キャンセル
            kwargs['feature'] = 'gauss'
            kwargs['sigma2'] = sigma2
            result = kp.main(**kwargs)          # 実行

        elif self.opt1_feature.get() == '2d':
            kwargs['feature'] = '2d'
            result = kp.main(**kwargs)          # 実行

            # パラメータを0~1の間に収めて、スライダーにセット
            params = result['pc'].param
            params = params / np.linalg.norm(params)
            self.slider_a_val.set(params[1])
            self.slider_b_val.set(params[2])
            self.slider_c_val.set(params[0])
            self.slider_a_label_val.set('    a = %+.2f ：  ' % params[1])
            self.slider_b_label_val.set('    b = %+.2f ：  ' % params[2])
            self.slider_c_label_val.set('    c = %+.2f ：  ' % params[0])

        # データキャンバスに識別関数の最終結果を描画
        self.disc_func = result['pc'].disc_func
        self.update_canvas_and_figure()

        # 結果のログを出力
        self.print_log('==== ==== ==== ==== Learning Finished ==== ==== ==== ====')
        self.print_log(result['msg'])
        self.print_log('Test Accuracy: %.2f%%' % (result['accuracy']*100))

    ###########################################################################

    # ログを出力するメソッド
    def print_log(self, *args):
        if len(args) == 0:
            return
        string = args[0]
        for arg in args[1:]:
            string += ' ' + arg
        for i, top in enumerate(range(0, len(string), 70)):
            self.logbox.insert(i, string[top:top+70])

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