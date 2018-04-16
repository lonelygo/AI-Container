# AI Container 集装箱智能检测及识别系统 
> Container automation detection system base on deep learning

>  该项目致力于解决集装箱箱号识别在移动端的解决方案. The main purpose of this project is focus on mobile solution for container OCR & detection

<img src="https://github.com/zdnet/AI-Container/blob/master/pic/port.jpg" width="600px" />

* 第一阶段：基于TF的集装箱自动识别系统 (Step 1： 箱号及size区域识别 container number & size position area detection) - Completed
  > [第一阶段测试结果 1st Phase testing result](https://github.com/zdnet/AI-Container/wiki/Test-Case) 

  > 目前还在加大数据集，下个批次的数据集加入了柜体侧面的照片

* 第二阶段：基于TF的集装箱自动识别系统 (Step 2： 箱号及size文本识别 container number & size OCR recognition) - Completed 90%

  > [第二阶段测试结果 2nd Phase testing result](https://github.com/zdnet/AI-Container/wiki/ctn_id_test_model) 
  
  > [Python 后端测试服务 Python backend testing](http://123.207.33.179:8000/container/index) - Last updated: 22/Mar 2018 by cody-chen-jf

* 第三阶段：基于TF的集装箱自动识别系统 (Step 3： 性能精度优化及移动端+树莓派集成 Performance tuning & mobile integration) - In-Progress

  > 第三阶段测试结果 3rd Phase testing result - by JcmeLs

* 第四阶段：基于TF的集装箱自动识别系统 (Step 4： 冻柜等特殊柜体智能检测应用 Reefer container status mornitoring) - TBC

* 第五阶段：基于TF的集装箱自动识别系统 (Step 5： 柜体破损自动检测，危险品柜自动检测 Container body damage area automation detection, danger container detection) - TBC

* 第六阶段：基于TF的集装箱自动识别系统 (Step 6： IoT connection with Android-Things, Ali-Things) - TBC

* 第七阶段：实际应用案例 (Step 7： Real case) - TBC
  - 集装箱洗柜修柜系统
  - 集装箱理货系统
  - 货代系统
  - 集装箱转运流程等


### 数据集的制作 Dataset preparing
我们从高雄港，广州港，上海港，长滩港采集了海量的高质量集装箱箱号图片作为训练集，分别采用了四种不同类别的算法有针对性地对图片做处理。此开放版本仅标注了集装箱箱号区域与箱尺寸区域。大约30%作为验证集，70%作为训练集。
此版本数据集综合考虑了各种集装箱规范，形状，尺寸，充分采样不同码头和港口的数据。相比于车牌识别，集装箱箱号的识别难度大，主要原因有：列印不规范，箱体破损，箱体被涂抹，逆光，暗光，角度限制，柜体层叠等等。我们的数据集充分考虑各个方面的因素，尽可能地对各种条件下的集装箱做照片采集。目前我们的采集工作还在进行当中。

We collected huge high quality container door picture from KAOCT, Guangzhou port, Shanghai port, Long Beach Port. This version only labeled CTN and size area, 70% images used for train dataset, 30% for test dataset.

<img src="https://github.com/zdnet/AI-Container/blob/master/pic/labeling.png" width="600px" />

>  目前只采集了横印型箱号的图片，对竖印型箱号暂时还不支持。 Only support horizontal printed number in this version.

### 识别模型 Model
系统可按照需求和应用场景自动适配识别模型，在移动端需要快速定位及识别的我们采用SSD单层模型做区域识别，基于多层模型做文本识别。在后端或者以拍照方式采集数据的移动端，我们采用精度更高的多层模型识别，同时在识别前用opencv对图片做增强处理，进一步提高识别率。

System can auto adapt suitable model to run graph according to environment and real usage. In mobile side real time detection requires high FPS then it will run SSD trained model, if requires high accuracy then it will run 2-stages trained model with opencv enhanced.

### 数据扩展 Data Extension
为了进一步提升识别率，我们对这些海量图片进一步做了增强处理，主要为角度旋转，加噪点，颜色偏移等，大约拓展出一倍的照片加入到训练集中。

To improve the accuracy, we enahced the images such as angle rotate, add noise, color range change, etc. We extend at least x2 data and  added into train dataset.

### 区域识别模型训练 Train Model
对区域识别做大约20个epoch的训练，目前区域识别率对角度比较好可以达到99%以上，对角度比较差的图片也有大约90%以上的准确率，之后会更新mAP数据。

For text area detection we have run 20+ epoch train, for good angle view we can reach 99%+ accuracy, for bad angle we also can reach 90%+ accuracy so far, will release out mAP data in next.

> 对箱号有涂抹的识别 (区域识别)
![2018-02-14_09_57_01-TensorBoard](https://github.com/zdnet/AI-Container/blob/master/pic/tf.png)

> More result: https://github.com/zdnet/AI-Container/wiki/Test-Case

### 文本识别模型训练 OCR recognize train
为了保证文本识别精度，特别是对箱号识别的准确度，我们制作了箱号字体生成器，对字体做3D变换。

To improve the accuracy, especially the container number in door recognize accuracy, we did a container id/size generator, add perspective transform for each image, and then paste font in background image to simulate real picture.

> 生成模拟数据并做3D变换 Generate dummy dataset and apply perspective transform

![Output sample](https://github.com/zdnet/AI-Container/blob/master/pic/3D.gif)


> 箱号自动截取 automation CTN region crop

<img src="https://github.com/zdnet/AI-Container/blob/master/pic/id.png" width="600px" />

> size自动截取 automation size region crop

<img src="https://github.com/zdnet/AI-Container/blob/master/pic/size.png" width="600px" />

> 背景制作 Background generator

<img src="https://github.com/zdnet/AI-Container/blob/master/pic/BG1.png" width="300px" />

文本识别模型我们采用了CRNN，虽然箱号和size是定长文本，但是经过比较后我们还是选择了CRNN作为识别的底层框架，第一个版本的识别准确度大约在85%左右，对有些易错字符如 Q,O | U,V | P,R 我们还会继续训练和处理。训练集大约有10多万的数据量，验证集有3万的数据量。对于不同集装箱字体我们还会继续有针对性的训练。

We used CRNN as the text recognition model after compared with CNN, the first version can reach 85% accuracy, to some easy-error char like Q,O | U,V | P,R we will continue to improve the algorithm. Train dataset have 10w+ records, eval dataset have 3w+ records. For differece container print font we will continue collect data to improve the accuracy.

### 移动端移植 Mobile integration
Android端采用pb压缩后的固化模型，ios端采用基于CoreML的转换后的模型。在保持较高帧率的条件下做高精度识别。对于第一个版本我们在app里集成了3个模型：区域识别，箱号识别和size识别，这样会导致我们的app很大，内存占用也多，之后我们会集成模型在一个model内。
对于不同模型的移动端移植，其实在TF层面还需要做很多工作，比如op_kernel的配置添加，.so重新打包, input, output的验证等等。

Android part we used frozen graph directly, iOS we used CoreML. To first version we integrated 3 models in app:region detection, container number recognize, size recognize. But it will make our app to a large size with high memory usage, in next we will combine 3 models in one model. 
For differece model mobile migration, we did many study and reseach, such as op_kernel add and re-build, re-package .so, verify input/ouput and so on. 

> 移动端移植 Mobile migration

<img src="https://github.com/zdnet/AI-Container/blob/master/pic/mobile_capture.jpg" width="800px" />

>  关于移动端移植，可参考我另外一篇相关文章（TBC）

