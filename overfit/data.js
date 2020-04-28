/* 
numSamples：要生成的样本的数量
variance：方差（变异），值越大，噪音越大
*/
export function getData(numSamples, variance) {
    let points = [];
  
    function genGauss(cx, cy, label) {
      for (let i = 0; i < numSamples / 2; i++) {
        let x = normalRandom(cx, variance);
        let y = normalRandom(cy, variance);
        points.push({ x, y, label });
      }
    }
  
    genGauss(2, 2, 1);
    genGauss(-2, -2, 0);
    return points;
  }
  
  /**
   * normalRandom：生成正态分布(高斯分布)的样本数据（正态分布两边低的数据看作噪音数据，中间高的数据视为正常数据）
   * 通过调整方差 调整噪音量
   * 方差越大 图形越矮胖 噪音数据越多
   * Samples from a normal distribution.
   * Uses the seedrandom library as the random generator.
   *
   * @param mean The mean. Default is 0.
   * @param variance The variance. Default is 1.
   */
  function normalRandom(mean = 0, variance = 1) {
    let v1, v2, s;
    do {
      v1 = 2 * Math.random() - 1;
      v2 = 2 * Math.random() - 1;
      s = v1 * v1 + v2 * v2;
    } while (s > 1);
  
    let result = Math.sqrt(-2 * Math.log(s) / s) * v1;
    return mean + Math.sqrt(variance) * result;
  }