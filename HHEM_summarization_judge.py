import os
os.environ['HF_HOME'] = '/bigtemp/hpwang/huggingface/cache/'
from transformers import AutoModelForSequenceClassification
import pandas as pd
from tqdm import tqdm
import argparse
import json
import numpy as np
from matplotlib import pyplot as plt
import torch
import time
start_time = time.time()
# Argument parsing
parser = argparse.ArgumentParser(description="QA judgment script.")
parser.add_argument("--model", type=str, default='Starling-LM-7B-alpha', help="Model name")
parser.add_argument('--form', type=str, default='semantic', help="Form of the data")
parser.add_argument("--answer_type", type=str, default='DeepSeek-R1-Distill-Llama-8B', help="Type of answer")
parser.add_argument("--query_selection", type=int, default=-1, help="Index for the query to use")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
args = parser.parse_args()

origin_df = pd.read_json('data/summarization_sampled_data.json', lines=True)
if args.answer_type in ['right', 'hallucinated']:
    judge_summaries = origin_df[f'{args.answer_type}_summary']
else:
    df = pd.read_json(f'./results/summarization/generated/{args.answer_type}/generated_summaries.json', lines=True)
    # Load query knowledge from the stored file
    judge_summaries = df["generated_summary"]


document = origin_df["document"].tolist()

knowledge_file = f'results/summarization/query_knowledge/{args.model}/{args.answer_type}_{args.form}_top2.json'
with open(knowledge_file, 'r') as f:
    query_knowledges = json.load(f)


# Extract right and hallucinated answers
# right_pairs = list(zip(df['document'].tolist(), df['right_summary'].tolist()))
# hallucinated_pair = list(zip(df['document'].tolist(), df['hallucinated_summary'].tolist()))

knowledge_pairs = list(zip(query_knowledges, document, judge_summaries.tolist()))
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


def predict_score(pairs):
    print("Predicting Pairs......")
    tensor_list = []
    for i in tqdm(range(len(pairs))):
        exp = pairs[i]  # Process in batches of 2500
        # if "wrong detail" in " ".join(exp[0]):
        #     tensor_list.append(torch.tensor((0, ), dtype=torch.float))
        # else:
        #     res_cur = model.predict([(exp[1], exp[2])])  # Ensure model.predict() is used instead of direct model call
        #     tensor_list.append(res_cur)
        flag = 0
        if "wrong detail" in " ".join(exp[0]):
            flag = 1
        res_cur = model.predict([(exp[1], exp[2])])  # Ensure model.predict() is used instead of direct model call
        if flag:
            res_cur = res_cur / 2
        tensor_list.append(res_cur)


    # Concatenate along dimension 0 to form a single 1D tensor of size 10*n
    result = torch.cat(tensor_list)
    return result


res = predict_score(knowledge_pairs)
total_correct = (res > 0.5).sum()
print(args.answer_type)
print(f"Accuracy: {total_correct} / {len(res)} = {total_correct / len(res):.3f}")
print(f"Hallucination Score: {1 - res.mean()}")

# Convert the tensor to a Python list
result_list = res.tolist()

# Save the list to a JSON file
json_filename = f"./results/summarization/judgment/{args.model}/{args.answer_type}_{args.form}_top2_curve.json"
with open(json_filename, "w") as json_file:
    json.dump(result_list, json_file, indent=4)

print(f"Result saved to {json_filename}")

# Mark the end time
end_time = time.time()
# Calculate the total elapsed time in seconds
total_seconds = int(end_time - start_time)
# Convert seconds to hours, minutes, and seconds
hours, remainder = divmod(total_seconds, 3600)
minutes, seconds = divmod(remainder, 60)

print(f"Total HHEM judge time: {hours:02d}:{minutes:02d}:{seconds:02d}")
