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

## list
- tensor 张量介绍
- linear-regression 线性回归
- height-weight 身高体重预测-归一化
- logistic-regression 逻辑回归
- xor 异或分类问题（多层神经网络，非线性问题）
- iris 鸢尾花分类问题
- overfit 欠拟合&过拟合演示
- mnist 手写数字识别