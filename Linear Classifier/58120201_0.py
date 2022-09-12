# 2.1 闭式解训练线性分类器
# utf-8
# np.dot(x,y) #<x,y>，x,y的内积
# np.linalg.pinv(X) #X的伪逆矩阵
# np.linalg.norm(z) #向量z的L2范数
import time
import pandas as pd
import numpy as np
import csv


def sigmoid(a):
    return 1 / (1 + np.exp(-a))


# 数据读取
train_feature = pd.read_csv('./ML4_programming-all./train_features.csv')
train_target = pd.read_csv('./ML4_programming-all./train_target.csv')
train_feature['add_column'] = 1  # 增广
x = train_feature.values
y = train_target.values

# 模型训练
time_start = time.time()
beta = np.dot(np.dot(np.linalg.pinv(np.dot(x.T, x)), x.T), y)
time_end = time.time()
print('%s  %f s' % ("训练耗时为", (time_end - time_start)))
print(beta)

# 模型评估
verify_feature = pd.read_csv('./ML4_programming-all./val_features.csv')
verify_target = pd.read_csv('./ML4_programming-all./val_target.csv')
verify_feature['add_column'] = 1
x_ = verify_feature.values
y_ = verify_target.values
TP = FN = FP = TN = 0
theta = 0.7                                   # 0.7时，可达到验证集上最优精度，提交的excel文件也是由theta=0.7生成的
z_ = np.dot(x_, beta)
z_ = sigmoid(z_)
for i in range(160):
    if z_[i] > theta and y_[i] == 1:
        TP += 1
        continue
    if z_[i] > theta and y_[i] == 0:
        FP += 1
        continue
    if z_[i] < theta and y_[i] == 1:
        FN += 1
        continue
    if z_[i] < theta and y_[i] == 0:
        TN += 1
        continue
accurancy = (TP + TN) / 160
P = TP / (TP + FP)
R = TP / (TP + FN)
print('%s  %.0f' % ("真阳性TP=", TP))
print('%s  %.0f' % ("假阳性FP=", FP))
print('%s  %.0f' % ("真阴性TN=", TN))
print('%s  %.0f' % ("假阴性FN=", FN))
print('%s %f' % ("精度为", accurancy))
print('%s  %f' % ("查准率P=", P))
print('%s  %f' % ("查全率R=", R))

# 验证结束，开始测试
test_feature = pd.read_csv('./ML4_programming-all./test_feature.csv')
test_feature['add_column'] = 1
x__ = test_feature.values
z__ = sigmoid(np.dot(x__, beta))
f = open('test_target_2.csv', 'w', encoding='utf-8', newline="")
csv_writer = csv.writer(f)
csv_writer.writerow(["class"])
for i in range(len(z__)):
    if z__[i] > theta:
        csv_writer.writerow([1])
    else:
        csv_writer.writerow([0])
