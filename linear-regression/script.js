import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';	//tfjs的可视化库

window.onload = async () => {
	//模拟输入输出数据
    const xs = [1, 2, 3, 4];
    const ys = [1, 3, 5, 7];

	//散点图scatterplot
    tfvis.render.scatterplot(
        { name: '线性回归训练集' },
        { values: xs.map((x, i) => ({ x, y: ys[i] })) },
        { xAxisDomain: [0, 5], yAxisDomain: [0, 8] }
    );

	// 定义一个神经网络模型（连续的模型，这一层的输入一定是上一层的输出，绝大多数模型都是sequential
	const model = tf.sequential();
	//给模型添加一个dense(全链接)层(layer)，全链接层可以解决  *权重+偏执类型的问题
	//units代表神经元个数为1，inputShape描述了输入的张量结构为一维数组
	model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
	//给神经网络模型设置 损失函数为均方误差 优化器为sgd（随机梯度下降）
    model.compile({ loss: tf.losses.meanSquaredError, optimizer: tf.train.sgd(0.1) });

    const inputs = tf.tensor(xs);
	const labels = tf.tensor(ys);
	// 训练模型
    await model.fit(inputs, labels, {
        batchSize: 4,	//超参数：每个梯度 模型要去学习的样本数量
		epochs: 200,	//超参数：遍历训练数据数组的次数
		// 可视化训练过程
        callbacks: tfvis.show.fitCallbacks(
            { name: '训练过程' },
            ['loss']
        )
	});
	

	//使用训练模型进行预测
    const output = model.predict(tf.tensor([5]));
    alert(`如果 x 为 5，那么预测 y 为 ${output.dataSync()[0]}`);
};