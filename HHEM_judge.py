import os

os.environ['HF_HOME'] = '/bigtemp/hpwang/huggingface/cache/'
from transformers import AutoModelForSequenceClassification
import pandas as pd
from tqdm import tqdm
import argparse
import json
from architectures import LLMCompletion
import numpy as np
from matplotlib import pyplot as plt

# Argument parsing
parser = argparse.ArgumentParser(description="QA judgment script.")
parser.add_argument("--model", type=str, default='Starling-LM-7B-alpha', help="Model name")
parser.add_argument('--form', type=str, default='semantic', help="Form of the data")
parser.add_argument("--answer_type", type=str, default='right', choices=['right', 'hallucinated'], help="Type of answer")
parser.add_argument("--knowledge_type", type=str, default='ground', choices=['ground', 'wiki'], help="Type of knowledge source")
parser.add_argument("--query_selection", type=int, default=-1, help="Index for the query to use")
args = parser.parse_args()

df = pd.read_json('data/qa_sampled_data.json', lines=True)

# Load query knowledge from the stored file
knowledge_right_file = f'results/qa/query_knowledge/Starling-LM-7B-alpha/right_ground_triplet.json'
with open(knowledge_right_file, 'r') as f:
    query_knowledges = json.load(f)
knowledge_hallucinated_file = f'results/qa/query_knowledge/Starling-LM-7B-alpha/hallucinated_ground_triplet.json'
with open(knowledge_right_file, 'r') as f:
    query_knowledges_1 = json.load(f)

# Extract right and hallucinated answers
# right_pairs = list(zip(df['knowledge'].tolist(), df['right_answer'].tolist()))
# hallucinated_pair = list(zip(df['knowledge'].tolist(), df['hallucinated_answer'].tolist()))

right_pairs = list(zip(query_knowledges, df['right_answer'].tolist()))
hallucinated_pair = list(zip(query_knowledges_1, df['hallucinated_answer'].tolist()))

# pairs = [ # Test data, List[Tuple[str, str]]
#     ("The capital of France is Berlin.", "The capital of France is Paris."), # factual but hallucinated
#     ('I am in California', 'I am in United States.'), # Consistent
#     ('I am in United States', 'I am in California.'), # Hallucinated
#     ("A person on a horse jumps over a broken down airplane.", "A person is outdoors, on a horse."),
#     ("A boy is jumping on skateboard in the middle of a red bridge.", "The boy skates down the sidewalk on a red bridge"),
#     ("A man with blond-hair, and a brown shirt drinking out of a public water fountain.", "A blond man wearing a brown shirt is reading a book."),
#     ("Mark Wahlberg was a fan of Manny.", "Manny was a fan of Mark Wahlberg.")
# ]

# Step 1: Load the model
model = AutoModelForSequenceClassification.from_pretrained(
    'vectara/hallucination_evaluation_model', trust_remote_code=True)


def predict_score(pairs, filter_datas, batch_size=2500):
    print("Predicting Pairs......")

    total_correct = 0
    total_processed = 0

    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]  # Process in batches of 2500
        filter_data = filter_datas[i:i + batch_size]
        res = model.predict(batch)  # Ensure model.predict() is used instead of direct model call

        for j, score in enumerate(res):
            data = filter_data[j]
            if "NONE" in data:
                continue  # Incorrect prediction if "NONE" is in filterdata
            elif score > 0.5:
                total_correct += 1  # Correct if score > 0.5
        total_processed += len(res)

    overall_accuracy = total_correct / total_processed if total_processed > 0 else 0
    print(f"Overall Accuracy: {total_correct} / {total_processed} = {overall_accuracy:.4f}")

    return total_correct, total_processed

with open('./results/qa/filter_hallucination/Starling-LM-7B-alpha/right.json', 'r') as f:
    rigth_filter_data = json.load(f)

with open('./results/qa/filter_hallucination/Starling-LM-7B-alpha/hallucinated.json', 'r') as f:
   hallucinated_filter_data = json.load(f)

right_pred, right_total = predict_score(right_pairs, rigth_filter_data)
hallucinated_pred, hallucinated_total = predict_score(hallucinated_pair, hallucinated_filter_data)

hallucinated_values = [hallucinated_pred, 0, hallucinated_total-hallucinated_pred]
right_values =[right_pred, 0, right_total-right_pred]
total_h = hallucinated_total
total_r = right_total

categories = ["Correct", "Inconclusive", "Incorrect"]
x = np.arange(len(categories))  # Label locations

hallucinated_percentages = [(v / total_h) * 100 if total_h > 0 else 0 for v in hallucinated_values]
right_percentages = [(v / total_r) * 100 if total_r > 0 else 0 for v in right_values]

# Plot Bar Chart
fig, ax = plt.subplots(figsize=(8, 5))
bar_width = 0.35

rects1 = ax.bar(x - bar_width/2, hallucinated_percentages, bar_width, label="Hallucinated")
rects2 = ax.bar(x + bar_width/2, right_percentages, bar_width, label="Right")

# Annotate bars with their percentages
for rect, perc in zip(rects1, hallucinated_percentages):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height, f'{perc:.1f}%', ha='center', va='bottom')

for rect, perc in zip(rects2, right_percentages):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height, f'{perc:.1f}%', ha='center', va='bottom')

ax.set_xlabel("Judgment Categories")
ax.set_ylabel("Percentage (%)")
ax.set_title("HHEM Comparison of Judgment Percentages (Hallucinated vs Right)")
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()
plt.savefig("./results/figure/HHEM_QA.png")

