#!/bin/sh

data_type=1          # 1 or 2 or 3
data_num=50          # 1 ~ 1000 (integer)
kernel_type='gauss'  # 'normal' or 'gauss'
learning_rate=0.01   # epsilon > 0 (real number)
screen_resolution=50 # 50 ~ 200 (integer)
show_f_true='False'  # 'False' or 'True'

python kernel_perceptron.py\
       $data_type\
       $data_num\
       $kernel_type\
       $learning_rate\
       $screen_resolution\
       $show_f_true\