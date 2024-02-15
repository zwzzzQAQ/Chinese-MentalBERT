import warnings
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold
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

# 使用StratifiedKFold进行五折交叉验证
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
tokenizer = BertTokenizer.from_pretrained('Chinese-MentalBERT')
# 加载数据
texts, labels = read_tsv('train_data.tsv')
dataset = TextDataset(texts, labels, tokenizer)
best_f1 = 0
best_model_path = "best.pt"  # 定义最佳模型保存路径

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载预训练模型和分词器
# 使用BCEWithLogitsLoss作为损失函数
loss_fn = torch.nn.BCEWithLogitsLoss()

for fold, (train_idx, val_idx) in enumerate(kf.split(texts)):
    print(f"Starting fold {fold+1}")

    train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
    val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)

    train_loader = DataLoader(dataset, batch_size=16, sampler=train_subsampler)
    val_loader = DataLoader(dataset, batch_size=8, sampler=val_subsampler)

    # 重新初始化模型和优化器
    model = BertForSequenceClassification.from_pretrained('Chinese-MentalBERT', num_labels=12)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # 训练和验证模型
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}, Fold {fold + 1}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        print(f"Average Training Loss: {train_loss / len(train_loader)}")

        # 验证模型
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
        print(f'Fold: {fold+1}, Epoch: {epoch+1}, F1: {f1}, Recall: {recall}, Precision: {precision}')

        # 检查并保存最佳模型
        if best_f1 < f1:
            best_f1 = f1
            torch.save(model.state_dict(), best_model_path)
            print(f'New best model saved at epoch {epoch+1} of fold {fold+1} with F1: {best_f1}')

print(f'Best F1 across all folds: {best_f1}')
