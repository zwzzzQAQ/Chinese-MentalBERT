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

# Chinese MentalBERT
This repository contains material associated to [Chinese MentalBERT: Domain-Adaptive Pre-training on Social Media for Chinese Mental Health Text Analysis](https://arxiv.org/pdf/2402.09151.pdf).

We published our model on huggingface([link](https://huggingface.co/zwzzz/Chinese-MentalBERT)).

If you use this material, we would appreciate if you could cite the following reference.

## Pre-training
1.Prepare your own pretraining corpus

2.Data cleaning, connect all instances and then split according to 128 tokens

3.Download the depression dictionary([link](https://github.com/omfoggynight/Chinese-Depression-domain-Lexicon)) and place it in the pretrain path

4.Download the word segmentation tool LTP([link](https://github.com/HIT-SCIR/ltp)) and place it in the pretrain path

5.Download your favorite pre-training model as the starting point for your pretraining. In this article we chose 
Chinese-BERT-wwm-ext([link](https://github.com/ymcui/Chinese-BERT-wwm)). Put it under the pretrain path

6.You could run the following:

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
## Downstream task finetuning and evaluation
Chinese MentalBERT is evaluated on four public datasets in the mental health domain, including sentiment analysis, suicide detection, and cognitive distortion identification.
In the provided open source code, we use cognitive distortion multi-label classification as an example as a demonstration of finetuning and evaluation on downstream tasks.

1.Download the cognitive distortion multi-label classification dataset([link](https://github.com/HongzhiQ/SupervisedVsLLM-EfficacyEval))
 and place it in the downstreamTasks path

2.Place your own pretrained model or the Chinese MentalBERT downloaded from Huggingface([link](https://huggingface.co/zwzzz/Chinese-MentalBERT)) under the downstreamTasks path

3.You could run the following:

```bash
python finetuning.py
```
Then you can evaluate like this: 

```bash
python evaluate.py
```
## References
1.Genghao Li, Bing Li, Langlin Huang, Sibing Hou, et al. 2020. Automatic construction of a depressiondomain lexicon based on microblogs: text mining study. JMIR medical informatics, 8(6):e17650.

2.Wanxiang Che, Yunlong Feng, Libo Qin, Ting Liu. 2020. N-LTP: An open-source neural language technology platform for Chinese. arXiv preprint arXiv:2009.11616.

3.Yiming Cui, Wanxiang Che, Ting Liu, Bing Qin, and Ziqing Yang. 2021. Pre-training with whole word masking for chinese bert. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 29:3504–3514.

4.Hongzhi Qi, Qing Zhao, Changwei Song, Wei Zhai, Dan Luo, Shuo Liu, Yi Jing Yu, Fan Wang, Huijing Zou, Bing Xiang Yang, et al. 2023. Evaluating the efficacy of supervised learning vs large language models for identifying cognitive distortions and suicidal risks in chinese social media. arXiv preprint arXiv:2309.03564.
