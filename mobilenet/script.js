import * as tf from '@tensorflow/tfjs';
import { IMAGENET_CLASSES } from './imagenet_classes';
import { file2img } from './utils';

const MOBILENET_MODEL_PATH = 'http://127.0.0.1:8080/mobilenet/web_model/model.json';

window.onload = async () => {
	//用tf的loadLayersModel加载模型
	const model = await tf.loadLayersModel(MOBILENET_MODEL_PATH);
	

    window.predict = async (file) => {

		//从文件转换为htmlElement
        const img = await file2img(file);
        document.body.appendChild(img);
        const pred = tf.tidy(() => {	//tidy优化webGl内存
            const input = tf.browser.fromPixels(img)	//转换为tensor
                .toFloat()	//整数转为浮点数
                .sub(255 / 2)	//归一化
                .div(255 / 2)
                .reshape([1, 224, 224, 3]);	//一个图片的格式
            return model.predict(input);	//预测
        });

		const index = pred.argMax(1).dataSync()[0];
		
		// setTimeout 0 使ui不被脚本阻塞
        setTimeout(() => {
            alert(`预测结果：${IMAGENET_CLASSES[index]}`);
        }, 0);
    };
};