import numpy as np
import torch
from datasets import load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, \
    Trainer, AutoConfig

from util import read_geoqa_training_data

# checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
checkpoint = "bert-base-uncased"
output_dir = 'test-model'
model_output_dir = "./models/geo-classification-model"
l2i = {
    "Borders": 0,
    "Containment": 1,
    "Proximity": 2,
    "Crossing": 3
}
i2l = {
    "0": "Borders",
    "1": "Containment",
    "2": "Proximity",
    "3": "Crossing"
}
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
metric = load_metric("f1", "accuracy")


def train(merge=False):
    datastore = read_geoqa_training_data()
    config = AutoConfig.from_pretrained(checkpoint, label2id=l2i, id2label=i2l, num_labels=4)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, config=config)

    train_texts = [record["text"] for record in datastore["train"]]
    train_labels_str = [record["label"] for record in datastore["train"]]
    train_labels = [l2i.get(lbl) for lbl in train_labels_str]

    validation_texts = [record["text"] for record in datastore["validation"]]
    validation_labels_str = [record["label"] for record in datastore["validation"]]
    validation_labels = [l2i.get(lbl) for lbl in validation_labels_str]

    test_texts = [record["text"] for record in datastore["test"]]
    test_labels_str = [record["label"] for record in datastore["test"]]
    test_labels = [l2i.get(lbl) for lbl in test_labels_str]

    if merge:
        train_texts.extend(test_texts)
        train_labels.extend(test_labels)

    train_encodings = tokenizer(train_texts, padding=True, truncation=True)
    validation_encodings = tokenizer(validation_texts, truncation=True, padding=True)
    # test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    train_dataset = GeoQA201Dataset(train_encodings, train_labels)
    val_dataset = GeoQA201Dataset(validation_encodings, validation_labels)
    # test_dataset = GeoQA201Dataset(test_encodings, test_labels)

    training_args = TrainingArguments(
        output_dir=output_dir,  # output directory
        num_train_epochs=50,  # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        logging_steps=10,
        evaluation_strategy="epoch"
    )

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset,  # evaluation dataset
        compute_metrics=compute_metrics
    )

    trainer.train()
    model.save_pretrained(model_output_dir)


def tokenize_function(text):
    return tokenizer(text, truncation=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels, average="micro")


class GeoQA201Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    train(merge=True)
