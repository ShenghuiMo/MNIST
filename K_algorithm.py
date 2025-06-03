#数据集下载链接：https://pan.baidu.com/s/1TaL3dCHxAj17LgvSSd_eTA?pwd=xl8n 
# author@Shenghui Mo
import os
import cv2
import re
import numpy as np
import matplotlib.pyplot as plt

folder_path = "mnist_jpg"

# ---------- 训练集 ----------
images_train = []
labels_train = []
# 类似于特征提取核大小
feature_kernel = 2
pattern_train = re.compile(r"training_(\d+)_(\d+)\.jpg")
# 设置需要的训练数据载入量
max_images_train = 10000
count = 0
# 遍历文件夹来读取数据
for filename in os.listdir(folder_path):
    if count >= max_images_train:
        break
    match = pattern_train.match(filename)
    if match:
        label_train = int(match.group(2))
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images_train.append(img)
            labels_train.append(label_train)
            count += 1
# 对数据进行归一化，并将数据转化为np格式
images_train = np.array(images_train)//255.0
labels_train = np.array(labels_train, dtype=np.int32).reshape(-1, 1)
length = len(images_train)

# 训练集特征提取（28x28 -> 28//feature*28//feature）
feature_train = np.zeros((length, 28//feature_kernel, 28//feature_kernel), dtype=np.float32)
for ni in range(length):
    for nr in range(28):
        for nc in range(28):
            # 对核内部的白色像素计数来达到提取特征的目的
            if images_train[ni, nr, nc] == 1:
                feature_train[ni, nr // feature_kernel, nc // feature_kernel] += 1

train = feature_train.reshape(length, -1)

# ---------- 测试集 ----------
images_test = []
labels_test = []

pattern_test = re.compile(r"test_(\d+)_(\d+)\.jpg")
max_images_test = 100
count_test = 0

for filename in os.listdir(folder_path):
    if count_test >= max_images_test:
        break
    match = pattern_test.match(filename)
    if match:
        label_test = int(match.group(2))
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images_test.append(img)
            labels_test.append(label_test)
            count_test += 1

images_test = np.array(images_test)/255.0
labels_test = np.array(labels_test, dtype=np.int32).reshape(-1, 1)
length_test = len(images_test)

# 测试集特征提取（同样 7x7）
feature_test = np.zeros((length_test, 28//feature_kernel, 28//feature_kernel), dtype=np.float32)
for ni in range(length_test):
    for nr in range(28):
        for nc in range(28):
            if images_test[ni, nr, nc] == 1:
                feature_test[ni, nr // feature_kernel, nc // feature_kernel] += 1

test = feature_test.reshape(length_test, -1)

# ---------- KNN 训练与预测 ----------
knn = cv2.ml.KNearest_create()
knn.train(train, cv2.ml.ROW_SAMPLE, labels_train)

ret, results, neighbours, dist = knn.findNearest(test, k=11)

# ---------- 打印结果 ----------
correct = np.sum(results.flatten() == labels_test.flatten())
accuracy = correct / length_test * 100
print(ret)
print(f"预测准确率: {accuracy:.2f}%")
print("预测标签前10个:", results[:10].flatten())
print("真实标签前10个:", labels_test[:10].flatten())
# 对实验结果进行可视化处理
x = np.arange(1,11)
plt.xlabel("number")
plt.ylabel('result')
plt.scatter(x, results[:10], s=80, c="b", marker="o")
plt.scatter(x, labels_test[:10], s=80, c='r', marker='s')
plt.grid()
plt.show()
cv2.waitKey()
cv2.destroyAllWindows()
