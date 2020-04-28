## build
node-gyp是node和计算机底层交互时经常要用到的包，在windows直接安装这个包也需要一些依赖，例如安装visual studio、python，因此windows-build-tools@4.0.0这个包就是用来解决其依赖问题的。

```npm i node-gyp windows-build-tools@4.0.0 -g```

安装后端版本tfjs
```npm i @tensorflow/tfjs-node```

npm包安装，用import引入，用parcel/webpack构建
```npm i @tensorflow/tfjs```

```npm i parcel-bundler -g  ```

```npm install```

## run
运行某页面
```parcel li*/*.html```

## link
- [tfjs Doc](https://js.tensorflow.org/api/latest)
- [google tfjs playground](http://playground.tensorflow.org)
- [image kernels网站了解图像卷积核](setosa.io/ev/image-kernels)
- [tfjs 官方预训练模型库](https://github.com/tensorflow/tfjs-models)

## list
- brand 商标识别
- brand-predict 商标识别保存&加载
- height-weight 身高体重预测-归一化
- iris 鸢尾花分类问题
- linear-regression 线性回归
- logistic-regression 逻辑回归
- mnist 手写数字识别
- mobilenet 图片识别预训练模型
- overfit 欠拟合&过拟合演示
- setup
- slider
- speech 语音识别
- speech-cn
- tensor 张量介绍
- xor 异或分类问题（多层神经网络，非线性问题）