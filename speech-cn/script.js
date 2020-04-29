import * as speechCommands from '@tensorflow-models/speech-commands';
import * as tfvis from '@tensorflow/tfjs-vis';

const MODEL_PATH = 'http://127.0.0.1:8080';
let transferRecognizer;

window.onload = async () => {
    const recognizer = speechCommands.create(
        'BROWSER_FFT',  //浏览器的傅里叶变换，将声音转为声谱数据
        null,
        MODEL_PATH + '/speech/model.json',
        MODEL_PATH + '/speech/metadata.json'
    );
    await recognizer.ensureModelLoaded();

    //使用createTransfer接口创建迁移模型
    transferRecognizer = recognizer.createTransfer('轮播图');
};

window.collect = async (btn) => {
    btn.disabled = true;
    const label = btn.innerText;
    //用collectExample接口收集语音数据，传入语音命令名称，背景音名称是固定的_background_noise_
    await transferRecognizer.collectExample(
        label === '背景噪音' ? '_background_noise_' : label
    );
    btn.disabled = false;

    //将收集的数据可视化
    document.querySelector('#count').innerHTML = JSON.stringify(transferRecognizer.countExamples(), null, 2);
};

window.train = async () => {
    //用train训练API训练收集的命令
    await transferRecognizer.train({
        epochs: 30,
        callback: tfvis.show.fitCallbacks(
            { name: '训练效果' },
            ['loss', 'acc'],
            { callbacks: ['onEpochEnd'] }
        )
    });
};

window.toggle = async (checked) => {
    if (checked) {
        //监听
        await transferRecognizer.listen(result => {
            const { scores } = result;
            const labels = transferRecognizer.wordLabels();
            const index = scores.indexOf(Math.max(...scores));
            document.body.appendChild(labels[index]);
        }, {
            overlapFactor: 0,
            probabilityThreshold: 0.75
        });
    } else {
        //停止监听
        transferRecognizer.stopListening();
    }
};

window.save = () => {
    //用serializeExamples接口将收集的样例数据序列化
    const arrayBuffer = transferRecognizer.serializeExamples();
    const blob = new Blob([arrayBuffer]);
    const link = document.createElement('a');
    link.href = window.URL.createObjectURL(blob);
    link.download = 'data.bin';
    link.click();
};