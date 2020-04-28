import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import { getData } from './data.js';

window.onload = async () => {
    const data = getData(400);

    tfvis.render.scatterplot(
        { name: 'XOR 训练数据' },
        {
            values: [
                data.filter(p => p.label === 1),
                data.filter(p => p.label === 0),
            ]
        }
    );

    //初始化一个神经网络连续型模型
    const model = tf.sequential();
    //添加一个隐藏层，神经元个数为4，神经元个数越多，训练越稳定越精确越慢
    model.add(tf.layers.dense({
        units: 5,
        inputShape: [2],
        activation: 'relu'  //设置激活函数
    }));
    //添加一个输出层，神经元格式为1,不需要设置inputShape，因为这一层的输入是上一层的输出
    model.add(tf.layers.dense({
        units: 1,
        activation: 'sigmoid' //隐藏层一定要加激活函数，否则所有的线性组合起来还是线性
    }));
    model.compile({
        loss: tf.losses.logLoss,
        optimizer: tf.train.adam(0.1)
    });

    const inputs = tf.tensor(data.map(p => [p.x, p.y]));
    const labels = tf.tensor(data.map(p => p.label));

    await model.fit(inputs, labels, {
        epochs: 10,
        callbacks: tfvis.show.fitCallbacks(
            { name: '训练效果' },
            ['loss']
        )
    });

    window.predict = (form) => {
        const pred = model.predict(tf.tensor([[form.x.value * 1, form.y.value * 1]]));
        alert(`预测结果：${pred.dataSync()[0]}`);
    };
};