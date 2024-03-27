import os
import tqdm
import xlrd
import torch
from torch import nn
import numpy as np

from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, BertConfig, AdamW
from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel
from torch.nn import CrossEntropyLoss


# dataloader
class MyDataset(Dataset):
    def __init__(self, tokenizer, max_len, path_file):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data_set = self.load_data(path_file)
        self.class_id = []

    def load_data(self, path_file):
        self.data_set = []

        books = xlrd.open_workbook(filename=path_file)
        sheets = books.sheets()[0]
        nrows, ncols = sheets.nrows, sheets.ncols
        title_name = sheets.row_values(0)
        assert "text" in title_name, ValueError(f"verify the key 'identity' in the list {sheets}.")
        assert "label" in title_name, ValueError(f"verify the key 'identity' in the list {sheets}.")

        for i in range(1, nrows - 1):
            sample = {}
            for k, v in zip(title_name, sheets.row_values(i)):
                sample[k] = v

            sentence = sample["text"]
            label = int(sample["label"])

            if label not in self.class_id:
                self.class_id.append(label)

            input_ids, token_type_ids, position_ids, attention_mask = self.convert_feature(sentence)
            self.data_set.append({"text": sentence,
                                  "input_ids": input_ids,
                                  "token_type_ids": token_type_ids,
                                  "attention_mask": attention_mask,
                                  "position_ids": position_ids,
                                  "label": label})
        return self.data_set

    def convert_feature(self, sentence):
        sentence_tokens = [i for i in sentence]
        tokens = ['[CLS]'] + sentence_tokens + ['[SEP]']
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        token_type_ids = [0] * len(input_ids)
        position_ids = [s for s in range(len(input_ids))]
        attention_mask = [1] * len(input_ids)

        return input_ids, token_type_ids, position_ids, attention_mask

    def get_class_num(self):
        return len(self.class_id)

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        instance = self.data_set[idx]
        return instance


def collate_func(batch_data):
    batch_size = len(batch_data)

    if batch_size == 0:
        return {}
    input_ids_list, token_type_ids_list, position_ids_list, attention_mask_list, label_list = [], [], [], [], []

    sample_list = []
    for instance in batch_data:
        input_ids_temp = instance["input_ids"]
        token_type_ids_temp = instance["token_type_ids"]
        position_ids_temp = instance["position_ids"]
        attention_mask_temp = instance["attention_mask"]
        label_temp = instance["label"]
        sample = {"text": instance["text"]}
        input_ids_list.append(torch.tensor(input_ids_temp, dtype=torch.long))
        token_type_ids_list.append(torch.tensor(token_type_ids_temp, dtype=torch.long))
        position_ids_list.append(torch.tensor(position_ids_temp, dtype=torch.long))
        attention_mask_list.append(torch.tensor(attention_mask_temp, dtype=torch.long))
        label_list.append(label_temp)
        sample_list.append(sample)

    return {"input_ids": pad_sequence(input_ids_list, batch_first=True, padding_value=0),
            "token_type_ids": pad_sequence(token_type_ids_list, batch_first=True, padding_value=0),
            "position_ids": pad_sequence(position_ids_list, batch_first=True, padding_value=0),
            "attention_mask": pad_sequence(attention_mask_list, batch_first=True, padding_value=0),
            "labels": torch.tensor(label_list, dtype=torch.long),
            "samples": sample_list}


# model
class BertPooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:,1]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class Text_Cls_Model(BertPreTrainedModel):
    base_model_prefix = "bert"

    def __init__(self, config, hidden_size, class_num):
        super(Text_Cls_Model, self).__init__(config)
        self.bert = BertModel(config)
        self.pooler = BertPooler(hidden_size)
        self.cls = nn.Linear(hidden_size, class_num)
        self.softmax = nn.Softmax()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, position_ids=None):
        outputs = self.bert.forward(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids,
                                    position_ids=position_ids)
        pooled_output = outputs[1]

        prediction_scores = self.cls(pooled_output)
        prediction_scores = self.softmax(prediction_scores)
        prediction_label = torch.argmax(prediction_scores, dim=1)
        outputs = (prediction_scores, prediction_label,)

        if labels is not None:
            loss_fct = CrossEntropyLoss()  # 交叉熵损失函数
            loss = loss_fct(prediction_scores.view(-1, 2), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs


# train and eval
def train(args):
    '''
    :param args:
    :return:
    '''
    device = torch.device("cuda" if torch.cuda.is_available() and int(args.device) >= 0 else "cpu")

    model = Text_Cls_Model.from_pretrained(args.pretrained_model_path)
    tokenizer = BertTokenizer.from_pretrained(args.vocab_path, do_lower_case=True)
    # dataloader
    train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)
    train_data = MyDataset(tokenizer, args.max_len, args.train_file_path)
    train_sampler = RandomSampler(train_data)
    train_data_loader = DataLoader(train_data,
                                   sampler=train_sampler,
                                   batch_size=train_batch_size,
                                   collate_fn=collate_func)
    test_data = MyDataset(tokenizer, args.max_len, args.test_file_path)

    model.to(device)
    optimizer = AdamW(model.parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    tr_loss = 0.0
    best_acc = 0.
    for iepoch in range(0, int(args.num_train_epochs)):
        model.train()
        for step, batch in enumerate(train_data_loader):
            input_ids = batch["input_ids"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            position_ids = batch["position_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model.forward(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids,
                                    position_ids=position_ids,
                                    labels=labels)
            loss = outputs[0]
            tr_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        eval_acc = evaluate(model, device, test_data, args)
        if best_acc < eval_acc:
            output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(iepoch))
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(output_dir)


def evaluate(model, device, dev_data, args):
    test_sampler = SequentialSampler(dev_data)
    test_data_loader = DataLoader(dev_data, sampler=test_sampler, batch_size=args.test_batch_size, collate_fn=collate_func)
    y_true = []
    y_predict = []
    for step, batch in enumerate(test_data_loader):
        model.eval()
        with torch.no_grad():
            labels = batch["labels"]
            input_ids = batch["input_ids"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            # attention_mask = batch["attention_mask"].to(device)
            # position_ids = batch["position_ids"].to(device)
            scores, prediction = model.forward(input_ids=input_ids,
                                               token_type_ids=token_type_ids)
            y_true.extend(labels.numpy().tolist())
            y_predict.extend(prediction.cpu().numpy().tolist())

    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    eval_acc = np.mean((y_true == y_predict))
    return eval_acc
