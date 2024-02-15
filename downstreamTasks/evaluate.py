import warnings
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import pandas as pd
import numpy as np
from sklearn import metrics
warnings.filterwarnings("ignore")
def calculate_evaluation(prediction, true_label, type):
    recall_list = []
    precision_list = []
    f1_list = []
    for i in range(0, len(true_label)):
        recall = metrics.recall_score(true_label[i], prediction[i], average=type)
        recall_list.append(recall)
        precision = metrics.precision_score(true_label[i], prediction[i], average=type)
        precision_list.append(precision)
        f1 = metrics.f1_score(true_label[i], prediction[i], average=type)
        f1_list.append(f1)
    recall_list = np.array(recall_list)
    precision_list = np.array(precision_list)
    f1_list = np.array(f1_list)
    return np.mean(recall_list), np.mean(precision_list), np.mean(f1_list)

# 加载预训练模型和分词器
tokenizer = BertTokenizer.from_pretrained('Chinese-MentalBERT')
model = BertForSequenceClassification.from_pretrained('Chinese-MentalBERT', num_labels=12)
# 加载模型权重（确保路径正确）
model.load_state_dict(torch.load('best.pt'))
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

def read_tsv(file_path):
    df = pd.read_csv(file_path, delimiter='\t', header=None)
    labels = df.iloc[:, :12].values.tolist()  # 前12列是标签
    texts = df[12].tolist()  # 最后一列是文本
    return texts, labels

# 加载数据
val_texts, val_labels = read_tsv('test_data.tsv')
# 创建数据集和数据加载器
val_dataset = TextDataset(val_texts, val_labels, tokenizer)
val_loader = DataLoader(val_dataset, batch_size=8)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
predict = np.zeros((0, 12), dtype=np.int32)
gt = np.zeros((0, 12), dtype=np.int32)
for batch in val_loader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        logits_np = logits.cpu().numpy()
        predictions = np.where(logits_np >= 0.5, 1, 0)
        predict = np.concatenate((predict, predictions))
        gt = np.concatenate((gt, labels.cpu().numpy()))
recall, precision, f1 = calculate_evaluation(predict, gt, type='macro')
print('  F1:', f1, '  recall:', recall, '  precision:', precision)
