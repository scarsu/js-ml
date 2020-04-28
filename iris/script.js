import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import { getIrisData, IRIS_CLASSES } from './data';

window.onload = async () => {
    const [xTrain, yTrain, xTest, yTest] = getIrisData(0.15);

    const model = tf.sequential();
    model.add(tf.layers.dense({
        units: 10,
        inputShape: [xTrain.shape[1]],
        activation: 'sigmoid'
    }));
    model.add(tf.layers.dense({
        units: 3, //输出有三类，因此输出层神经元个数必须设置为3
        activation: 'softmax'   //微积和函数适用于：输出有多类，且和为1
    }));

    model.compile({
        loss: 'categoricalCrossentropy',  //交叉熵损失函数
        optimizer: tf.train.adam(0.1), 
        metrics: ['accuracy']   //准确度度量
    });

    //训练过程是异步的
    await model.fit(xTrain, yTrain, {
        epochs: 100,
        validationData: [xTest, yTest], //验证集数据
        callbacks: tfvis.show.fitCallbacks(
            { name: '训练效果' },
            ['loss', 'test_loss', 'acc', 'test_acc'],
            { callbacks: ['onEpochEnd'] }
        )
    });

    window.predict = (form) => {
        const input = tf.tensor([[
            form.a.value * 1,
            form.b.value * 1,
            form.c.value * 1,
            form.d.value * 1,
        ]]);
        const pred = model.predict(input);
        alert(`预测结果：${IRIS_CLASSES[pred.argMax(1).dataSync(0)]}`);
    };
};