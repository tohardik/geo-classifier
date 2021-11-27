import math
import random

CLASS_LABELS = ["Proximity", "Crossing ", "Containment", "Border", "Superlative", "Comparison"]


def read_benchmark_questions():
    with open("./benchmark.json", "r") as benchmark_file:
        import json
        benchmark_json = json.load(benchmark_file)
        questions_json = benchmark_json["questions"]
        questions_list = []
        for question in questions_json:
            questions_list.append(question["question"][0]["string"])
        return questions_list


def read_benchmark_questions_with_labels():
    with open("./GeoQA-annotated.tsv", "r") as file_content:
        all_lines = file_content.readlines()
        all_records = [{"label": line.split("\t")[0], "text": line.split("\t")[1].strip()} for line in all_lines if
                       len(line) > 0]
        return all_records


def read_geoqa_training_data():
    with open("./201Training.tsv", "r") as data201:
        all_lines = data201.readlines()
        all_records = [{"label": line.split("\t")[0], "text": line.split("\t")[1].strip()} for line in all_lines]

        bag = {}
        for x in all_records:
            if x["label"] in bag:
                bag[x["label"]].append(x)
            else:
                bag[x["label"]] = [x]

        train = []
        val = []
        test = []

        for x in bag:
            rows = bag[x]
            test_n = math.ceil(len(rows) * 0.3)
            train_n = len(rows) - test_n
            val_n = math.ceil(test_n / 2)
            test_n = test_n - val_n
            random.Random(13).shuffle(rows)

            train.extend(rows[:train_n])
            val.extend(rows[train_n:train_n + val_n])
            test.extend(rows[train_n + val_n:])

        train_test_valid_dataset = {
            'train': train,
            'test': test,
            'validation': val}

        return train_test_valid_dataset
