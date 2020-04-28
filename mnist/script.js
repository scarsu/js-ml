import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import { MnistData } from './data';

window.onload = async () => {
	const data = new MnistData();
	
	//加载资源
	await data.load();
	
	//获取20组数据
    const examples = data.nextTestBatch(50);
	// console.log(examples)
	
    const surface = tfvis.visor().surface({ name: '输入示例' });
    for (let i = 0; i < 20; i += 1) {

		//截取出单个图片：从第1维的第i项+第二维的第1项开始截取，第一维截取长度是1，第二维截取长度是784
		//console.log(examples.xs.slice([i, 0], [1, 784]))

		//tf.tidy：用于优化webgl内存，防止tensor数据量过大导致内存泄漏

		//tf.browser.toPixels：转换成浏览器能识别的像素格式，传入二位参数就是黑白图片，三维就是彩色的

		//tensor.reshape：tensor格式转换
        const imageTensor = tf.tidy(() => {
            return examples.xs
                .slice([i, 0], [1, 784])  
                .reshape([28, 28, 1]);	//将一维数组转换成三维黑白图片格式
        });

        const canvas = document.createElement('canvas');
        canvas.width = 28;
        canvas.height = 28;
        canvas.style = 'margin: 4px';
        await tf.browser.toPixels(imageTensor, canvas);
        surface.drawArea.appendChild(canvas);
    }

	const model = tf.sequential();
	//添加一个二位卷积层
    model.add(tf.layers.conv2d({
        inputShape: [28, 28, 1],
        kernelSize: 5,	//卷积核的大小是5X5的矩阵
        filters: 8,	//应用8种图像卷积核
        strides: 1,	//移动步长，每一个像素单元都进行卷积操作
        activation: 'relu',	//激活函数，移除掉无用的特征（特征<0就废弃
        kernelInitializer: 'varianceScaling'	//可以不设置，设置了可以加快收敛速度
	}));
	
	//最大池化层
    model.add(tf.layers.maxPool2d({
        poolSize: [2, 2], //尺寸是2X2
        strides: [2, 2]	//移动步长，每隔两个像素单元进行一次卷积操作
	}));
	
	// 重复上述两个层
    model.add(tf.layers.conv2d({
        kernelSize: 5,
        filters: 16,	//需要提取更多特征
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }));
    model.add(tf.layers.maxPool2d({
        poolSize: [2, 2],
        strides: [2, 2]
	}));
	
	//flatten层用于将多维的特征数据，转换为一维的分类数据，传入dense层
	model.add(tf.layers.flatten());
	

    model.add(tf.layers.dense({
        units: 10,		//最终输出0-9十个分类
        activation: 'softmax',
        kernelInitializer: 'varianceScaling'
	}));
	
	//训练参数
    model.compile({
        loss: 'categoricalCrossentropy',	//交叉熵损失函数
        optimizer: tf.train.adam(),	//优化器
        metrics: ['accuracy']	//准确度度量
    });

	//训练集数据
    const [trainXs, trainYs] = tf.tidy(() => {
        const d = data.nextTrainBatch(1000);
        return [
            d.xs.reshape([1000, 28, 28, 1]),
            d.labels
        ];
    });

	//验证集数据
    const [testXs, testYs] = tf.tidy(() => {
        const d = data.nextTestBatch(200);
        return [
            d.xs.reshape([200, 28, 28, 1]),
            d.labels
        ];
    });

	//训练
    await model.fit(trainXs, trainYs, {
        validationData: [testXs, testYs],
        batchSize: 500,
        epochs: 100,
        callbacks: tfvis.show.fitCallbacks(
            { name: '训练效果' },
            ['loss', 'val_loss', 'acc', 'val_acc'],
            { callbacks: ['onEpochEnd'] }
        )
    });


    const canvas = document.querySelector('canvas');

    canvas.addEventListener('mousemove', (e) => {
		if (e.buttons === 1) {
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = 'rgb(255,255,255)';
            ctx.fillRect(e.offsetX, e.offsetY, 25, 25);
        }
    });

	//黑底画板
    window.clear = () => {
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = 'rgb(0,0,0)';
        ctx.fillRect(0, 0, 300, 300);
    };

    clear();

    window.predict = () => {
        const input = tf.tidy(() => {
            return tf.image.resizeBilinear(		//转换图像tensor尺寸
                tf.browser.fromPixels(canvas),	//canvas转换为tensor
                [28, 28],	//转换成28*28
                true
            ).slice([0, 0, 0], [28, 28, 1])	//canvas图片是彩色图片，通过slice转换为黑白图片
            .toFloat()	//训练数据进行过归一化，因此预测值也要归一化
            .div(255)	//归一化
            .reshape([1, 28, 28, 1]);	//和神经网络第一层的输入格式统一
        });
        const pred = model.predict(input).argMax(1);
        alert(`预测结果为 ${pred.dataSync()[0]}`);
    };
};