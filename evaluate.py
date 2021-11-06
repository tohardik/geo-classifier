import datasets
import tensorflow as tf
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from train import i2l, l2i
from util import read_geoqa_training_data, read_benchmark_questions_with_labels

checkpoint = "./models/geo-classification-model"

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=4)

predictions = []
references = []
f1_metric = datasets.load_metric("f1")
# datastore = read_geoqa_training_data()
# question_with_labels = datastore["test"]
question_with_labels = read_benchmark_questions_with_labels()
for x in question_with_labels:
    test_sentence = x["text"]
    input_encodings = tokenizer(test_sentence, padding=True, truncation=True, return_tensors="pt")
    output = model(**input_encodings)
    output_probabilities = tf.math.softmax(output.logits.detach().numpy(), axis=-1)[0]
    # output_with_labels = {}
    max_prob = -1
    max_i = -1
    for i in range(len(output_probabilities)):
        prob_value = float(output_probabilities[i])
        if prob_value > max_prob:
            max_prob = prob_value
            max_i = i
        # output_with_labels[i2l.get(str(i))] = prob_value
    # print(test_sentence)
    # print(predictions_with_labels)
    # print()
    references.append(l2i.get(x["label"]))
    predictions.append(max_i)

results = f1_metric.compute(predictions=predictions, references=references, average="macro")
print("macro: ", results)

results = f1_metric.compute(predictions=predictions, references=references, average="micro")
print("micro: ", results)

results = f1_metric.compute(predictions=predictions, references=references, average="weighted")
print("weighted: ", results)
