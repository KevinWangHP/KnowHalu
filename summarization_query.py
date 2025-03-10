import os
os.environ['HF_HOME'] = '/bigtemp/hpwang/huggingface/cache/'
import setGPU
import pandas as pd
from tqdm import tqdm
import argparse
import json
import numpy as np
from architectures import LLMCompletion
from utils import extract_query, split_summary_into_parts
from retrieve import SummaryRetriever
import time
# Mark the start time
start_time = time.time()

# Argument parsing
parser = argparse.ArgumentParser(description="Summary processing script.")
parser.add_argument("--model", type=str, default='Starling-LM-7B-alpha', help="Model name")
parser.add_argument('--form', type=str, default='semantic', help="Form of the data")
parser.add_argument("--topk", type=int, default=3, help="Top K results for document retrieval")
parser.add_argument("--answer_type", type=str, default='right', choices=['right', 'hallucinated'], help="Type of answer")
parser.add_argument("--query_selection", type=int, default=-1, help="Index for the query to use")
parser.add_argument("--save_freq", type=int, default=5, help="Frequency of saving checkpoints")
parser.add_argument("--count_limit", type=int, default=10, help="Limit for the count within the loop")
parser.add_argument("--generate_model", type=str, default='Llama-3.1-8B-Instruct', help="Model name")
parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
args = parser.parse_args()

df = pd.read_json('data/summarization_sampled_data.json', lines=True)
    
documents = df['document'].tolist()
if args.generate_model == 'origin':
    summaries = df[args.answer_type + '_summary'].tolist()
else:
    df_generate = pd.read_json(f'results/summarization/generated/{args.generate_model}/generated_summaries.json', lines=True)
    summaries = df_generate['generated_summary'].tolist()
retriever = SummaryRetriever(topk = args.topk)

# Read instructions
if args.query_selection != -1:
    suffix = f'_selection{args.query_selection}'
else:
    suffix = ''
    
with open(f'prompts/summarization/query_{args.form}{suffix}.txt', 'r', encoding="utf-8") as f:
    main_instruction = f.read()

knowledge_file = f'prompts/summarization/retrieve_{args.form}{suffix}.txt'
with open(knowledge_file, 'r', encoding="utf-8") as f:
    knowledge_instruction = f.read()
######################
    
stop_tokens = ['#Knowledge', '\n\n']
llm = LLMCompletion(args.model)

# Resume functionality
if args.generate_model == 'origin':
    file_name = f'results/summarization/query_knowledge/{args.model}/{args.answer_type}_{args.form}_top{args.topk}'
else:
    file_name = f'results/summarization/query_knowledge/{args.model}/{args.generate_model}_{args.form}_top{args.topk}'
    
if args.query_selection != -1:
    file_name += f'_q{args.query_selection}'
file_name += '.json'

directory = os.path.dirname(file_name)
if not os.path.exists(directory):
    os.makedirs(directory)
    
if args.resume:
    try:
        with open(file_name, 'r') as f:
            query_knowledge = json.load(f)
    except FileNotFoundError:
        print("No checkpoint file found, starting from scratch.")
else:
    query_knowledge = [[] for _ in range(len(documents))]

for i in tqdm(range(len(documents))):
    if query_knowledge[i] != []:
        continue
        
    for summary in split_summary_into_parts(summaries[i].strip()):
        count = 0
        prompt = main_instruction.format(summary=summary)
        prompt_length = len(prompt)
        prompt += '#Thought-1#:'
        current_output = llm(prompt, stop_tokens)
        last_output = current_output
        count += 1
        if args.model.startswith('gpt'):
            prompt += ' ' + current_output
        else:
            prompt += current_output

        while count < args.count_limit:
            if '\n\n' in current_output or '#Done#' in current_output or 'further queries' in current_output:
                break
            elif current_output.endswith('#Knowledge') or (args.model.startswith('gpt') and '\n' == current_output[-1:]) or ('Query-' in  current_output.split('\n')[-1]) or (args.model.startswith('gpt') and 'Query-' not in current_output):
                if 'Query-' in  current_output.split('\n')[-1]:
                    current_output += '\n'
                elif 'Query-' not in current_output and not current_output.endswith('\n'):
                    prompt += '\n'
                query = extract_query(current_output)
                if len(query) == 0:
                    last_newline_index = prompt.rfind('\n')
                    prompt = prompt[:last_newline_index]
                    prompt += f'\n#Query-{count}#:'
                    current_output = llm(prompt, stop_tokens).split('\n')[0]
                    if args.model.startswith('gpt'):
                        prompt += ' ' + current_output + '\n'
                    else:
                        prompt += current_output + '\n'
                    query = extract_query(f'#Query-{count}#:' + current_output)
                    if len(query) == 0:
                        import pdb; pdb.set_trace()

                knowledge = retriever.retrieve(documents[i], query)
                if args.query_selection != -1 or len(query) < 2:
                    knowledge_prompt = knowledge_instruction.format(question=query[0], knowledge=knowledge)
                else:
                    knowledge_prompt = knowledge_instruction.format(question=f'{query[0]} [{query[1]}]', knowledge=knowledge)
                knowledge_output = llm(knowledge_prompt).split('\n')[0]
                if args.model.startswith('gpt'):
                    if not prompt.endswith('\n'):
                        prompt += '\n'
                    prompt += f'#Knowledge-{count}#: ' + knowledge_output + f'\n#Thought-{count+1}#:'
                else:
                    if not prompt.endswith('\n#Knowledge'):
                        prompt += '\n#Knowledge'
                    prompt += f'-{count}#:' + knowledge_output + f'\n#Thought-{count+1}#:'
            else:
                break

            current_output = llm(prompt, stop_tokens)
            if current_output == last_output:
                break
            else:
                last_output = current_output
                
            count += 1
            if args.model.startswith('gpt'):
                prompt += ' ' + current_output
            else:
                prompt += current_output
                
        output = prompt[prompt_length:].strip()
        query_knowledge[i].append(output)
        print(output, "\n")
        
    # Save intermediate results
    if (i + 1) % args.save_freq == 0 or i == len(documents) - 1:
        with open(file_name, 'w') as f:
            json.dump(query_knowledge, f)


# Mark the end time
end_time = time.time()
# Calculate the total elapsed time in seconds
total_seconds = int(end_time - start_time)
# Convert seconds to hours, minutes, and seconds
hours, remainder = divmod(total_seconds, 3600)
minutes, seconds = divmod(remainder, 60)
# Print all arguments in a readable format
print("Parsed Arguments:")
for arg, value in vars(args).items():
    print(f"{arg}: {value}")
print(f"Total summarization query time: {hours:02d}:{minutes:02d}:{seconds:02d}")