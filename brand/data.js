const loadImg =(src)=>{
  return new Promise(resolve=>{
    const img = document.createElement('img')
    img.crossOrigin = 'anonymous'
    img.src = src
    img.width = 224   //以mobileNet为截断模型，其接收图片尺寸为224
    img.height = 224
    img.onload=()=>resolve(img)
  })
}

// 返回Promise
export const getInputs = async()=>{
  const loadImgs = []
  const labels = []
  for(let i=0;i<30;i+=1){
    ['android','apple','windows'].forEach(label=>{
      const imgP = loadImg(`http://127.0.0.1:8081/brand/train/${label}-${i}.jpg`)
      loadImgs.push(imgP)
      labels.push([
        label === 'android' ? 1 :0,
        label === 'apple' ? 1 :0,
        label === 'windows' ? 1 :0,
      ])
    })
  }
  // 利用promise.all一次性加载全部图片
  const inputs = await Promise.all(loadImgs)
  return{ 
    inputs, labels
  }
}