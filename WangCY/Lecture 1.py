import datetime

from sklearn.datasets import fetch_openml
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np

images, targets = fetch_openml("mnist_784", return_X_y=True, as_frame=False)

# 可视化图片
images = images.reshape(-1, 28, 28)
plt.imshow(images[0], cmap='gray')
plt.show()

starttime=datetime.datetime.now()

# 取3个类各200张图完成即可
# 示例取类2,3,4
classes = ['2', '3', '4']
data = []
label = [[i]*200 for i in range(len(classes))]
for l in classes:
    data.append(images[targets == l][: 200])
data = np.concatenate(data, axis=0)
label = np.concatenate(label)
data = data.reshape(600, -1) / 255 # 压缩值0-1之间方便计算距离

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data, label,test_size=0.3, random_state=1)


def kNNClassify(newInput, dataSet, labels, k):
    numSamples = dataSet.shape[0]

    diff = np.tile(newInput, (numSamples, 1)) - dataSet  # 按元素求差值
    squaredDiff = diff ** 2  # 将差值平方
    squaredDist = np.sum(squaredDiff, axis=1)  # 按行累加
    distance = squaredDist ** 0.5  # 将差值平方和求开方，即得距离

    # # step 2: 对距离排序
    sortedDistIndices = np.argsort(distance)
    classCount = {}
    for i in range(k):
        # # step 3: 选择k个最近邻
        voteLabel = labels[sortedDistIndices[i]]

        # # step 4: 计算k个最近邻中各类别出现的次数
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

     # # step 5: 返回出现次数最多的类别标签
    maxCount = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            maxIndex = key
    return maxIndex

predictions_knn=[]
for k in range(3,11):
    predictions = []
    for i in range(X_test.shape[0]):
        predictions_labes = kNNClassify(X_test[i], X_train, y_train, k)
        predictions.append(predictions_labes)
    m = 0
    for j in range(len(predictions)):
        if predictions[j] == list(y_test)[j]:
            m = m+1
    predictions_k = m / len(predictions)
    predictions_knn.append(predictions_k)


#准备绘制数据
x = range(3,11)
y = predictions_knn
# "g" 表示红色，marksize用来设置'D'菱形的大小
plt.plot(x, y, "g", marker='D', markersize=5)
#绘制坐标轴标签
plt.xlabel("k")
plt.ylabel("accuracy")

plt.show()

# print(confusion_matrix(y_test, predictions))
# # 打印预测结果混淆矩阵
# print(classification_report(y_test, predictions))
# # 打印精度、召回率、FI结果
endtime = datetime.datetime.now()
print(endtime - starttime)

# # 建立映射方便表示。由于我们只取3类,映射为0,1,2就行
# class_map = {'2' : 0, '3': 1, '4': 2}