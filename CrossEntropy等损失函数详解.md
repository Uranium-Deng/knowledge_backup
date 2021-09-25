## CrossEntropy和MSE损失函数详解

### 一 CrossEntropy 函数

​	crossentropy == softmax + log + NLLLoss(), 这里的log以自然对数e为底就是ln, 因此分类任务中若采用交叉熵作为损失函数，则网络最后一层不需要softmax()函数。CrossEntropy函数的计算公式如下所示。
$$
-\sum_{i=1}^{N}P(i)*log(Q(i))
$$
​	其中P(i)表示真实label，Q(i)表示模型输出的label，这里的Q(i)是转换为one_hot类型和softmax之后的数据。若为二分类即01分布，则计算公式如下：
$$
-[P(i)*log(Q(i)) + (1 - P(i) * log(1-Q(i)))]
$$
​	CrossEntropy(model_output, lable)，前者model_output为网络输出，是one_hot类型，后者label是真实的label，不需要人为转换为one_hot类型，程序自身会进行转换。其中model_output和label的shape通常为：(N, num_classes)和(num_classes, )在图像分割中的shape为(N, num_classes, H, W)和(N, H, W)。

​	对应演示代码如下：

~~~python
import torch
import torch.nn as nn
import torch.nn.functional as F

model_output = torch.randn(size=(4, 21))
label = torch.empty(size=(4,), dtype=torch.long).random_(21)

# crossEntropy()
m = nn.CrossEntropyLoss()
print(m(model_output, label).item())

# softmax + log + NLLLoss()
m = nn.LogSoftmax(dim=1)
temp = m(model_output)
m = nn.NLLLoss()
print(m(temp, label).item())

class CrossEntropy(nn.Module):
    # 手动实现CrossEntropy函数
    def __init__(self):
        super(CrossEntropy, self).__init__()

    def forward(self, outputs, labels):
        # labels变为one_hot向量
        labels = F.one_hot(labels, num_classes=outputs.shape[1])
        # 对outputs进行softmax, log操作
        outputs = torch.log(torch.softmax(outputs, dim=1))  
        loss = -torch.sum(labels * outputs) / outputs.shape[0]  # 除以被分类对象的数量
        return loss

ce = CrossEntropy()
print(ce(model_output, label).item())

''' 代码输出
3.040743589401245
3.040743589401245
3.040743589401245
'''
~~~

​		调用crossEntropy()函数时，model_output的dtype最好是torch.float，label的dtype必须是torch.long。此外crossEntropy()函数所求的loss默认是average的，除以了被分类对象的数量（若为普通图像分类，则就是输入图像的数量N，若是图像分割，则除以 N * H * W）这一点和tensorflow不同，tensorflow默认计算的是sum总和，没有求平均值。

​		若不想要crossEntropy()函数计算mean，而要计算sum或者原本的loss向量，则可增加参数reduction，reduce和size_average参数未来将不再使用。

- reduce=True, size_average=True，默认如此，此时输出mean

- reduce=True, size_average=False，输出sum总和

- reduce=False, 则size_average参数将被忽略，输出的是loss组成的向量，该向量的shape和label的shape一样

  未来将采用reduction参数，该参数有三个str选项。

- reduction='none', 输出loss向量 loss vector

- reduction='mean'，输出mean

- reduction='sum'，输出sum总和

  对应演示代码如下

~~~python
model_output = torch.randn(size=(4, 21))
label = torch.empty(size=(4,), dtype=torch.long).random_(21)

# crossEntropy()
m = nn.CrossEntropyLoss(reduction='none')
print(m(model_output, label))

m = nn.CrossEntropyLoss(reduction='sum')
print(m(model_output, label).item())

m = nn.CrossEntropyLoss(reduction='mean')
print(m(model_output, label).item())

'''
tensor([4.2624, 3.7496, 3.0325, 3.0499])
14.094429016113281
3.5236072540283203
'''
~~~

​	面对图像分割这样的分类任务，CrossEntropy()函数和NLLLoss()函数的使用方法完全不变。在自定义CrossEntropy()函数方面，卡壳在如何生成label对应的one-hot向量上(等待解决, permute()函数更换通道，最终得到的loss值不一样)。

​	还有一点就是weight权重参数的作用， 分类任务中存在这样的情况：某些类别的对象出现的次数较多，某些类别的对象出现的次数较少，没有weight的原始CrossEntropy()函数，对所有分类对象都是平等看待，这样网络训练到最后即使准确率较高，但可能对出现次数较少的类别，其分类效果反而不好。因此引入weight矩阵，weight 是一个一维向量，长度等于种类数量，其中每一个数字代表该类别对应的权重，我们可以把出现次数较少的类别的权重设置的大一些，出现次数较多的类别的权重设置的小一些。

​	设置weight参数后，求loss的mean时，除以的数量不再是所有分类对象的数量了，而是每个分类对象对应类别的weight值的加和，举例如下。

~~~python
inputs = torch.FloatTensor([0, 0, 0, 1, 0, 1, 2, 0, 0, 0, 0, 0.5])
outputs = torch.LongTensor([0, 1, 2, 2])
inputs = inputs.view(size=(4, 3))  # 4个分类对象，3类
outputs = outputs.view(size=(4,))

weight_CE = torch.FloatTensor([1, 2, 3])  # 3个类别对应的weight分别为1, 2, 3

ce = nn.CrossEntropyLoss(weight=weight_CE)
loss = ce(inputs, outputs)
print(loss)

'''模型输出：
tensor(1.5472)

对应的手算过程为：
loss1 = 0 + ln(e0 + e0 + e0) = 1.098
loss2 = 0 + ln(e1 + e0 + e1) = 1.86
loss3 = 0 + ln(e2 + e0 + e0) = 2.2395
loss4 = -0.5 + ln(e0.5 + e0 + e0) = 0.7943
求平均 = (loss1 * 1 + loss2 * 2+loss3 * 3+loss4 * 3) / 9 = 1.5472

crossEntropy()计算公式，对model_out先进行softmax，后log，后和label相乘, 而label为one-hot向量，每一行中只有一个值为1，剩下的都为0，所以后面将label和output相乘时，和0相乘的部分结果都为0,所以反正结果为0，那我们只要softmax,log处理output中在label中对应位置为1的值就行 -ln(\frac{e^{x_i}}{\sum_{i=0}^{N}e^{x_i}})
'''
~~~

$$
-ln(\frac{e^{x_i}}{\sum_{i=0}^{N}e^{x_i}})
$$

---

CrossEntropy() 函数总结：

1. 交叉熵数学公式，彻底了解计算过程，自定义实现crossentropy()函数
2. reduction参数, none, mean, sum, weight 参数

参考文献：

https://blog.csdn.net/qq_27095227/article/details/103775032

---

### 二 MSE 函数

​	MSE损失函数没什么好说的，算的就是mean square error 平均方差，其中reduction默认的也是求解mean，若像要sum和none也可自行设置。

~~~python
import torch
import torch.nn as nn

model_output = torch.randint(low=0, high=255, size=(2, 256, 256)).float()
label = torch.empty(size=(2, 256, 256), dtype=torch.float).random_(255)

# MSE()函数
m = nn.MSELoss(reduction='mean')
print(m(model_output, label).item())

# 手动计算
img_shape = model_output.shape
final_ans = 0
for n in range(img_shape[0]):
    temp_ans = 0
    for row in range(img_shape[1]):
        for col in range(img_shape[2]):
            temp_ans += (model_output[n][row][col].item() - label[n][row][col].item()) ** 2
    final_ans += temp_ans / (img_shape[1] * img_shape[2])
print('final_ans: ', final_ans / img_shape[0])

''' 代码输出
10819.6376953125
final_ans:  10819.637344360352
'''
~~~

---

### 三 BCELoss 函数











