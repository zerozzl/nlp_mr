# 自然语言处理-机器阅读理解

对比常见模型在机器阅读理解任务上的效果，主要涉及以下几种模型：

- [MACHINE COMPREHENSION USING MATCH-LSTM AND ANSWER POINTER](https://arxiv.org/pdf/1608.07905.pdf)
- [BI-DIRECTIONAL ATTENTION FLOW FOR MACHINE COMPREHENSION](https://arxiv.org/pdf/1611.01603.pdf)
- [QANET: COMBINING LOCAL CONVOLUTION WITH GLOBAL SELF-ATTENTION FOR READING COMPREHENSION](https://arxiv.org/pdf/1804.09541.pdf)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

## Char-level 效果

### Match-LSTM

|-|Embed Rand|Embed Pretrained|Embed Fix|Embed Rand + Bigram|Embed Pretrained + Bigram|Embed Fix + Bigram|
|----|----|----|----|----|----|----|
|dureader|0.582|0.583|<b>0.588</b>|0.538|0.587|0.582|
|cmrc2018|0.114|0.154|0.147|0.062|<b>0.232</b>|0.229|

### BiDAF

|-|Embed Rand|Embed Pretrained|Embed Fix|Embed Rand + Bigram|Embed Pretrained + Bigram|Embed Fix + Bigram|
|----|----|----|----|----|----|----|
|dureader|<b>0.659</b>|0.604|0.6|0.536|0.462|0.479|
|cmrc2018|<b>0.7</b>|0.555|0.547|0.545|0.354|0.359|

### QANet

|-|Embed Rand|Embed Pretrained|Embed Fix|Embed Rand + Bigram|Embed Pretrained + Bigram|Embed Fix + Bigram|
|----|----|----|----|----|----|----|
|dureader|<b>0.596</b>|0.172|0.172|0.533|0.173|0.174|
|cmrc2018|<b>0.349</b>|0.109|0.11|0.302|0.109|0.113|

### BERT

|-|finetune|freeze|
|----|----|----|
|dureader|<b>0.77</b>|0.353|
|cmrc2018|<b>0.773</b>|0.077|

## Word-level 效果

### Match-LSTM

|-|Embed Rand|Embed Pretrained|Embed Fix|
|----|----|----|----|
|dureader|0.247|<b>0.314</b>|<b>0.314</b>|
|cmrc2018|0.025|<b>0.198</b>|0.177|

### BiDAF

|-|Embed Rand|Embed Pretrained|Embed Fix|
|----|----|----|----|
|dureader|0.299|0.319|<b>0.326</b>|
|cmrc2018|0.37|<b>0.391</b>|0.388|

### QANet

|-|Embed Rand|Embed Pretrained|Embed Fix|
|----|----|----|----|
|dureader|0.279|<b>0.305</b>|0.3|
|cmrc2018|0.193|<b>0.249</b>|0.245|
