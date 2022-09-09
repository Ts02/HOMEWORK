from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib as plt

images, targets = fetch_openml("mnist_784", return_X_y=True, as_frame=False)

# 可视化图片
images = images.reshape(-1, 28, 28)
plt.imshow(images[0], cmap='gray')
plt.show()

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
X_train, X_test, y_train, y_test = train_test_split(data, label)

# 建立映射方便表示。由于我们只取3类,映射为0,1,2就行
class_map = {'2' : 0, '3': 1, '4': 2}

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]

    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet  # x0-x1 y0-y1
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)  # 处理成一行
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=lambda k: k[1], reverse=True)
    return sortedClassCount[0][0]



def train(TestSet,labelsTest,TrainSet,labelsTrain,k):
    length=TestSet.shape[0]+TrainSet.shape[0];
    correct = 0.0
    labelsForOutput = []
    for d in Testset:
        labelsForOutput.append(classify0(d, Trainset, labelsTrain, k))
    for i in range(len(labelsForOutput)):
        if labelsForOutput[i] == labelsTest[i]:
            correct += 1
    return correct / len(labelsForOutput)  # 正确率
if __name__=='__main__':

    train(X_test,y_test,X_train,y_train);