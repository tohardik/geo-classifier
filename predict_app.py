import logging

import tensorflow as tf
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from train import i2l

app = Flask(__name__)

LOG = logging.getLogger("predict_app.py")

checkpoint = "./models/geo-classification-model"

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=4)


@app.route('/')
def home():
    return jsonify({"status": 200})


def process_input(input_text):
    input_encodings = tokenizer(input_text, padding=True, truncation=True, return_tensors="pt")
    output = model(**input_encodings)
    output_probabilities = tf.math.softmax(output.logits.detach().numpy(), axis=-1)[0]

    output_with_labels = {}
    for i in range(len(output_probabilities)):
        output_with_labels[i2l.get(str(i))] = float(output_probabilities[i])

    return output_with_labels


@app.route('/classify', methods=["POST", "GET"])
def classify():
    if request.method == "POST":
        input_text = request.form.get('input_text')
    else:
        input_text = request.args.get('input_text')

    if input_text is None:
        LOG.error("/classify,  Invalid parameters provided")
        return "Invalid parameters provided", 400
    else:
        LOG.info(f"/classify {request.method}, input={input_text}")
        result = process_input(input_text)
        return jsonify({
            "inputText": input_text,
            "result": result
        })


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=9091)
