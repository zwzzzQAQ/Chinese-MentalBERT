<!---
Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Chinese MentalBERT: Domain-Adaptive Pre-training on Social Media for Chinese Mental Health Text Analysis
This repository contains material associated to [this paper](https://arxiv.org/pdf/2402.09151.pdf).

It contains:

* link to domain adaptive pretrained models for Chinese mental health domain ([link](#Trained-ChineseMentalBERT))
* link to trained model for 4 evaluation tasks: two semantic recognition tasks [(link)](<#>), suicide classification ([link](<#>)), cognitive distortion ([link](<#>))
* code and material for domain adaptive pretraining ([link](#Domain-adaptive-Pre-training))
* code and material for downstream tasks finetuning and evaluation ([link](#Downstream-task-fine-tuning-and-evaluation))


## Download and install summary

### Material for domain adaptive pretraining

* **Download pretraining corpus**: 
  * **Sina Weibo Depression Dataset (SWDD) [5]** : https://github.com/ethan-nicholas-tsai/DepressionDetection
  * **Weibo User Depression Detection Dataset (WU3D) [6]**: https://github.com/aidenwang9867/Weibo-User-Depression-Detection-Dataset
* **Download the depression lexicon [7]**:  https://github.com/omfoggynight/Chinese-Depression-domain-Lexicon
* **Download the word segmentation tool LTP:** https://github.com/HIT-SCIR/ltp
* **Download the Chinese pre-trained BERT model (Chinese-BERT-wwm-ext) [3]**: https://huggingface.co/hfl/chinese-bert-wwm-ext

### Material for fine tuning on downstream task

* **Download the datasets**: 
  * **SMP2020-EWECT (Sentiment analysis tasks)**: https://github.com/BrownSweater/BERT_SMP2020-EWECT
  * **Suicide and cognitive distrotion tasks [4]:** https://github.com/HongzhiQ/SupervisedVsLLM-EfficacyEval
* **Download the pretrained model:** 
  * **Chinese MentalBERT:** https://huggingface.co/zwzzz/Chinese-MentalBERT


## Domain adaptive Pre-training

### Prepare the pretraining corpus and domain lexicon

We use two public data sets as pretraining corpus examples and the depression lexicon for guided mask mechanism, see [link](<#Material-for-domain-adaptive-pretraining>) for details. Feel free to add more related corpus or lexion to enrich your pretrain material. 


### Data pre-processing

In the `pre_processing.py`, it includes the following steps: 

1. **Data cleaning:** remove irrelevant information, which included URLs, user tags (e.g., @username), topic tags (e.g., #topic#), and we also removed special symbols, emoticons, and unstructured characters.
2. **Sentence concatenation:** Connect all cleaned sentences in their original sequence to form a continuous stream of text.
3. **Segmentation into 128-token samples**: Split the continuous text stream into multiple samples, each containing 128 tokens, to facilitate efficient processing and enable the model to learn long-distance dependencies in the text.

And put your data as [`TRAIN_FILE`](#Domain-adaptive-pre-training-on-the-corpus) when you run the pre-training.

### Configure word segmentation tool

Download the word segmentation tool LTP ([link](https://github.com/HIT-SCIR/ltp)) and put it as [`LTP_RESOURCE`](#Domain-adaptive-pre-training-on-the-corpus) when you run the pre-training.

### Download the foundational pre-trained model as start point

We utilized the Chinese pre-trained BERT model **([Chinese-BERT-wwm-ext](<https://huggingface.co/hfl/chinese-bert-wwm-ext>))** [3] in our experiment for the foundational pre-trained model. And put it as [`BERT_RESOURCE`](#Domain-adaptive-pre-training-on-the-corpus) when you run the pre-training.

### Domain adaptive pre-training on the corpus

You could run the following:

```bash
export TRAIN_FILE=/path/to/train/file
export LTP_RESOURCE=/path/to/ltp/tokenizer
export BERT_RESOURCE=/path/to/bert/tokenizer
export SAVE_PATH=/path/to/data/ref.txt

python run_chinese_ref.py \
    --file_name=$TRAIN_FILE \
    --ltp=$LTP_RESOURCE \
    --bert=$BERT_RESOURCE \
    --save_path=$SAVE_PATH
```

Then you can run the script like this: 


```bash
export TRAIN_FILE=/path/to/train/file
export VALIDATION_FILE=/path/to/validation/file
export TRAIN_REF_FILE=/path/to/train/chinese_ref/file
export VALIDATION_REF_FILE=/path/to/validation/chinese_ref/file
export OUTPUT_DIR=/tmp/test-mlm-wwm

python run_mlm_wwm.py \
    --model_name_or_path roberta-base \
    --train_file $TRAIN_FILE \
    --validation_file $VALIDATION_FILE \
    --train_ref_file $TRAIN_REF_FILE \
    --validation_ref_file $VALIDATION_REF_FILE \
    --do_train \
    --do_eval \
    --output_dir $OUTPUT_DIR
```
### Trained ChineseMentalBERT

Our trained model is public available at: https://huggingface.co/zwzzz/Chinese-MentalBERT. You can load it and use it to fine tune on your downstream task.

## Downstream task fine tuning and evaluation

Chinese MentalBERT is evaluated on four public datasets in the mental health domain, including two semantic recognition tasks ([link](<https://github.com/BrownSweater/BERT_SMP2020-EWECT>), suicide classification ([link](<https://github.com/HongzhiQ/SupervisedVsLLM-EfficacyEval>)), cognitive distortion ([link](<https://github.com/HongzhiQ/SupervisedVsLLM-EfficacyEval>)).
In the provided open source code, we use cognitive distortion multi-label classification as an example as a demonstration of finetuning and evaluation on downstream tasks.

### Prepare the experimental datasets

You can download the public dataset in our experiment as details in [link](# Material-for-fine-tuning-on-downstream-task). And put it on the `downstreamTasks` path

### Prepare the pretrained model

You can download the [pretrained model](<https://huggingface.co/zwzzz/Chinese-MentalBERT>) and set up

### Fine tuning for the downstream tasks

You could run the following:

```bash
python finetuning.py
```
### Performance evaluation

Then you can evaluate like this: 

```bash
python evaluate.py
```
## References
1. Genghao Li, Bing Li, Langlin Huang, Sibing Hou, et al. 2020. Automatic construction of a depressiondomain lexicon based on microblogs: text mining study. JMIR medical informatics, 8(6):e17650.

2. Wanxiang Che, Yunlong Feng, Libo Qin, Ting Liu. 2020. N-LTP: An open-source neural language technology platform for Chinese. arXiv preprint arXiv:2009.11616.

3. Yiming Cui, Wanxiang Che, Ting Liu, Bing Qin, and Ziqing Yang. 2021. Pre-training with whole word masking for chinese bert. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 29:3504–3514.

4. Hongzhi Qi, Qing Zhao, Changwei Song, Wei Zhai, Dan Luo, Shuo Liu, Yi Jing Yu, Fan Wang, Huijing Zou, Bing Xiang Yang, et al. 2023. Evaluating the efficacy of supervised learning vs large language models for identifying cognitive distortions and suicidal risks in chinese social media. arXiv preprint arXiv:2309.03564.
5. Cai, Yicheng, et al. "Depression detection on online social network with multivariate time series feature of user depressive symptoms." *Expert Systems with Applications* 217 (2023): 119538.
6. Wang, Yiding, et al. "A multitask deep learning approach for user depression detection on sina weibo." *arXiv preprint arXiv:2008.11708* (2020).
7. Li, Genghao, et al. "Automatic construction of a depression-domain lexicon based on microblogs: text mining study." *JMIR medical informatics* 8.6 (2020): e17650.
