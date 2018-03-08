# AI-Container - Container Automation Detection System base on Tensorflow 

* 第一阶段：基于TF的集装箱自动识别系统 (Step 1： 箱号及size区域识别 container number & size position area detection) - Completed
  > [第一阶段测试结果](https://github.com/zdnet/AI-Container/wiki/Test-Case) 

* 第二阶段：基于TF的集装箱自动识别系统 (Step 2： 箱号及size文本识别 container number & size OCR recognition) - In-Progress

* 第三阶段：基于TF的集装箱自动识别系统 (Step 3： 性能精度优化及移动端+树莓派集成 Performance tuning & mobile integration) - TBC

* 第四阶段：基于TF的集装箱自动识别系统 (Step 4： 冻柜等特殊柜体智能检测应用 Reefer container status mornitoring) - TBC

* 第五阶段：基于TF的集装箱自动识别系统 (Step 5： 柜体破损自动检测，危险品柜自动检测 Container body damage area automation detection, danger container detection) - TBC

* 第六阶段：基于TF的集装箱自动识别系统 (Step 6： IoT connection with Android-Things, Ali-Things) - TBC


### 数据集的制作 Dataset preparing
我们从高雄港，广州港，上海港，长滩港采集了海量的高质量集装箱箱号图片作为训练集，分别采用了四种不同类别的算法有针对性地对图片做处理。此开放版本仅标注了集装箱箱号区域与箱尺寸区域。大约30%作为验证集，70%作为训练集。
此版本数据集综合考虑了各种集装箱规范，形状，尺寸，充分采样不同码头和港口的数据。相比于车牌识别，集装箱箱号的识别难度大，主要原因有：列印不规范，箱体破损，箱体被涂抹，逆光，暗光，角度限制，柜体层叠等等。我们的数据集充分考虑各个方面的因素，尽可能地对各种条件下的集装箱做照片采集。目前我们的采集工作还在进行当中。

We collected huge high quality container door picture from KAOCT, Guangzhou port, Shanghai port, Long Beach Port. This version only labeled CTN and size area, 70% images used for train dataset, 30% for test dataset.

![20180103_101246](https://github.com/zdnet/AI-Container/blob/master/pic/combine.jpg)

>  目前只采集了横印型箱号的图片，对竖印型箱号暂时还不支持。 Only support horizontal CTN in this version.

### 识别模型 Model
系统可按照需求和应用场景自动适配识别模型，在移动端需要快速定位及识别的我们采用SSD单层模型做区域识别，基于多层模型做文本识别。在后端或者以拍照方式采集数据的移动端，我们采用精度更高的多层模型识别，同时在识别前用opencv对图片做增强处理，进一步提高识别率。

System can auto adapt suitable model to run graph according to environment and real usage. In mobile side real time detection requires high FPS then it will run SSD trained model, if requires high accuracy then it will run 2-stages trained model with opencv enhanced.

### 数据扩展 Data Extension
为了进一步提升识别率，我们对这些海量图片进一步做了增强处理，主要为角度旋转，加噪点，颜色偏移等，大约拓展出一倍的照片加入到训练集中。

To improve the accuracy, we enahced the images such as angle rotate, add noise, color change, etc. extend at least x2 data to add into train dataset.

### 区域识别模型训练 Train Model
对区域识别做大约20个epoch的训练，目前区域识别率对角度比较好可以达到99%以上，对角度比较差的图片也有大约90%以上的准确率，之后会更新mAP数据。

For text area detection we have run 20+ epoch train, for good angle view we can reach 99%+ accuracy, for bad angle we also can reach 90%+ accuracy so far, will release out mAP data in next.

> 对箱号有涂抹的识别 (区域识别)
![2018-02-14_09_57_01-TensorBoard](https://github.com/zdnet/AI-Container/blob/master/pic/tf.png)

> More result: https://github.com/zdnet/AI-Container/wiki/Test-Case

### 文本识别模型训练 OCR recognize train
为了保证文本识别精度，特别是对箱号识别的准确度，我们制作了箱号字体生成器，对字体做3D变换。

> 生成模拟数据并做3D变换

![Output sample](https://github.com/zdnet/AI-Container/blob/master/pic/3D.gif)


> 箱号自动截取 automation CTN region crop

<img src="https://github.com/zdnet/AI-Container/blob/master/pic/id.png" width="600px" />

> size自动截取 automation size region crop

<img src="https://github.com/zdnet/AI-Container/blob/master/pic/size.png" width="600px" />

> 背景制作

<img src="https://github.com/zdnet/AI-Container/blob/master/pic/BG1.png" width="600px" />

### 移动端移植 Mobile integration
Android端采用pb压缩后的固化模型，ios端采用基于CoreML的转换后的模型。在保持较高帧率的条件下做高精度识别

<img src="https://github.com/zdnet/AI-Container/blob/master/pic/android.jpg" width="300px" />

>  关于移动端移植，可参考我另外一篇相关文章（TBC）

