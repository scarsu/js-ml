import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import { getInputs } from './data';
import { img2x, file2img } from './utils';

const MOBILENET_MODEL_PATH = 'http://127.0.0.1:8081/mobilenet/web_model/model.json';
const NUM_CLASSES = 3;
const BRAND_CLASSES = ['android', 'apple', 'windows'];

window.onload = async () => {
    const { inputs, labels } = await getInputs();
    console.log([inputs,labels])

    //将加载的图片素材可视化
    const surface = tfvis.visor().surface({ name: '输入示例', styles: { height: 250 } });
    inputs.forEach(img => {
        surface.drawArea.appendChild(img);
    });

    //加载预训练好的模型Mobilenet
    const mobilenet = await tf.loadLayersModel(MOBILENET_MODEL_PATH);
    
    //mobilenet的方法，给出其神经网络的概览(层、输出类型、参数)
    // mobilenet.summary();

    //获取中间层
    const layer = mobilenet.getLayer('conv_pw_13_relu');

    //定义一个截断模型truncatedMobilenet
    const truncatedMobilenet = tf.model({
        inputs: mobilenet.inputs,
        outputs: layer.output
    });

    //定义一个模型
    const model = tf.sequential();

    //添加一个flatten层（将截断模型提取的高维特征提取成一维向量，这一层没有参数，起转换作用
    model.add(tf.layers.flatten({
        inputShape: layer.outputShape.slice(1)
    }));

    //添加一个全链接层：用于训练我们的商标图片
    model.add(tf.layers.dense({
        units: 10,
        activation: 'relu'
    }));

    //添加一个全链接层：用于做多分类
    model.add(tf.layers.dense({
        units: NUM_CLASSES,
        activation: 'softmax'
    }));

    //设置损失函数：分类交叉熵损失函数，优化器为adam
    model.compile({ loss: 'categoricalCrossentropy', optimizer: tf.train.adam() });

    //训练数据 先经过截断模型，转为可以用于迁移模型的数据
    const { xs, ys } = tf.tidy(() => {
        const xs = tf.concat(inputs.map(imgEl => truncatedMobilenet.predict(img2x(imgEl))));
        const ys = tf.tensor(labels);
        return { xs, ys };
    });
    // console.log({ inputs, labels } )
    // console.log({ xs, ys } )

    //训练迁移模型
    await model.fit(xs, ys, {
        epochs: 20,
        callbacks: tfvis.show.fitCallbacks(
            { name: '训练效果' },
            ['loss'],
            { callbacks: ['onEpochEnd'] }
        )
    });

    window.predict = async (file) => {
        const img = await file2img(file);
        document.body.appendChild(img);
        const pred = tf.tidy(() => {
            const x = img2x(img);
            //截断模型先得到预测结果
            const input = truncatedMobilenet.predict(x);
            //再把数据传入迁移模型预测
            return model.predict(input);
        });

        const index = pred.argMax(1).dataSync()[0];
        setTimeout(() => {
            alert(`预测结果：${BRAND_CLASSES[index]}`);
        }, 0);
    };

    window.download = async () => {
        await model.save('downloads://model');
    };
};