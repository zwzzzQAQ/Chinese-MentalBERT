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

## Depression Dictionary Guided Whole Word Masking 
We have opened the [Chinese MentalBERT](https://huggingface.co/zwzzz/Chinese-MentalBERT) model on huggingface.

The following example will be executed on a corpus aggregated from four datasets: comments from the “Zoufan” Weibo treehole, 
the Weibo Depression “Chaohua” (a super topic on Weibo), the Sina Weibo Depression Dataset ([SWDD](https://github.com/ethan-nicholas-tsai/SWDD)), and 
the Weibo User Depression Detection Dataset ([WU3D](https://github.com/aidenwang9867/Weibo-User-Depression-Detection-Dataset)). Alternatively, you may utilize your own text files for training and validation purposes. Due to data privacy considerations, the "Zoufan" tree hole dataset and Weibo Depression "Chaohua" dataset we collected cannot be made public.

Cui et al. released [Pre-Training with Whole Word Masking for Chinese BERT](https://arxiv.org/pdf/1906.08101.pdf), first introduces the whole word mask (wwm) 
strategy of Chinese BERT. In this work, to enhance the model’s applicability to psychological text analysis, we integrated psychological lexicons into the 
pre-training masking mechanism. 

To better tailor our model to the specific needs of the mental health domain, we implemented a guided masking strategy utilizing a depression lexicon. This approach begins by identifying whether the pre-training text contains lexicon words; if so, these words are masked for prediction training. Should the proportion of text occluded fall below 20%, we augment the masked selection with additional, randomly chosen words. It’s important to note the distinct strategies required for word guidance and masking in English versus Chinese texts. While masking in word-level suffices in English, Chinese requires word segmentation to mask compound words accurately, ensuring complete concepts are expressed and understood by the model. Here we choose [LTP](https://github.com/HIT-SCIR/ltp) as our Chinese word segmentation tool. Our research investigates a lexicon-guided masking mechanism. We choose to use the depression dictionary developed by Li et al., available [here](https://github.com/omfoggynight/Chinese-Depression-domain-Lexicon). 

For the logical implementation of Chinese full-word masking, reference is made to the [transformers library](https://github.com/huggingface/transformers/tree/main/examples/research_projects/mlm_wwm).


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
## Downstream Task Finetuning and Evaluation
The proposed pretrained model underwent evaluation on four public datasets in the mental health domain, including sentiment analysis, suicide detection and cognitive distortion identification.

The sentiment analysis task is derived from [SMP2020-EWECT](https://github.com/BrownSweater/BERT_SMP2020-EWECT). This Weibo emotion classification evaluation task comprises two datasets: the first is a usual Weibo dataset, featuring randomly collected data on various topics; the second is an epidemic-related Weibo dataset. The Weibo data included all pertain to the COVID epidemic. The objective of the Weibo emotion classification task is to identify the emotions contained in Weibo posts. The input consists of a Weibo post, and the output is the identified emotion category contained in the Weibo post. In this task, the dataset categorizes Weibo posts into one of six categories based on the emotions they contain: positive, angry, sad, fearful, surprised, and no emotion.

Cognitive distortion multi-label datasets and high and low suicide risk datasets are available from [this](https://github.com/HongzhiQ/SupervisedVsLLM-EfficacyEval). The cognitive distortion task centers on the categories defined by Burns. Data were obtained by crawling comments from the “Zoufan” blog on the Weibo social platform. Subsequently, a team of qualified psychologists was recruited to annotate the data. Given that the data are publicly accessible, privacy concerns are not applicable. The classification labels in the cognitive distortion dataset include: all-or-nothing thinking, overgeneralization, mental filtering, demotivation, mind reading, fortune teller error, amplification, emotional reasoning, should statements, labeling and mislabeling, blaming yourself and blaming others. The suicide risk task aims to differentiate between high and low suicide risk, For the suicide detection data, the dataset contained 645 records with low suicide risk and 601 records with high suicide risk.


We provide code using cognitive distortion multi-label classification as an example.


You can finetuning with the following commands：


```bash
python finetuning.py
```
Then you can evaluate like this: 

```bash
python evaluate.py
```
## References
1.Yicheng Cai, Haizhou Wang, Huali Ye, Yanwen Jin, and Wei Gao. 2023. Depression detection on online social network with multivariate time series feature of user depressive symptoms. Expert Systems with Applications, 217:119538.

2.Yiming Cui, Wanxiang Che, Ting Liu, Bing Qin, and Ziqing Yang. 2021. Pre-training with whole word masking for chinese bert. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 29:3504–3514.

3.Genghao Li, Bing Li, Langlin Huang, Sibing Hou, et al. 2020. Automatic construction of a depressiondomain lexicon based on microblogs: text mining study. JMIR medical informatics, 8(6):e17650.

4.Hongzhi Qi, Qing Zhao, Changwei Song, Wei Zhai, Dan Luo, Shuo Liu, Yi Jing Yu, Fan Wang, Huijing Zou, Bing Xiang Yang, et al. 2023. Evaluating the efficacy of supervised learning vs large language models for identifying cognitive distortions and suicidal risks in chinese social media. arXiv preprint arXiv:2309.03564.
