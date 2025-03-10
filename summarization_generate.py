import os
os.environ['HF_HOME'] = '/bigtemp/hpwang/huggingface/cache/'
from pathlib import Path
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline, StoppingCriteria, StoppingCriteriaList
import pandas as pd
from openai import OpenAI
import argparse
import re
os.environ["OPENAI_API_KEY"] = 'sk-xxx'
client = OpenAI()
from token import access_token

parser = argparse.ArgumentParser(description="QA processing script.")
parser.add_argument("--model", type=str, default='internlm3-8b-instruct', help="Model name")
args = parser.parse_args()

class LLMCompletion(nn.Module):
    def __init__(self, model_name, max_new_tokens=1000, system_prompt=None):
        super(LLMCompletion, self).__init__()

        self.model_name = model_name
        self.models = {
            'Llama-2-7b-chat-hf': "meta-llama/Llama-2-7b-chat-hf",
            'Starling-LM-7B-alpha': "berkeley-nest/Starling-LM-7B-alpha",
            'Meta-Llama-3-8B-Instruct': "meta-llama/Meta-Llama-3-8B-Instruct",
            'Llama-3.1-8B-Instruct': "meta-llama/Llama-3.1-8B-Instruct",
            'DeepSeek-R1-Distill-Llama-8B': "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            'Qwen2.5-7B-Instruct': "Qwen/Qwen2.5-7B-Instruct",
            'gemma-7b-it': "google/gemma-7b-it",
            'Mistral-7B-Instruct-v0.3': "mistralai/Mistral-7B-Instruct-v0.3",
            'internlm3-8b-instruct': "internlm/internlm3-8b-instruct"
        }

        if model_name in self.models:
            model_path = self.models[model_name]
            self.generation_config = GenerationConfig.from_pretrained(model_path, token=access_token)
            self.generation_config.max_new_tokens = max_new_tokens
            self.generation_config.temperature = 0.
            self.generation_config.do_sample = False
            self.generation_config.top_p = 1.0
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, token=access_token, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.float16, device_map="auto", token=access_token, trust_remote_code=True
            ).eval()
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

    @torch.no_grad()
    def forward(self, prompt, stop_tokens=['\n'], split_token=None, return_prob=False):

        inputs = self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, return_tensors="pt").to('cuda')
        stopping_criteria = self.get_stopping_criteria(stop_tokens)
        if self.model_name in ["Llama-3.1-8B-Instruct", 'Mistral-7B-Instruct-v0.3']:
            pad_token_id = self.tokenizer.eos_token_id
        else:
            pad_token_id = self.tokenizer.pad_token_id
        output = self.model.generate(
            input_ids=inputs,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=pad_token_id,
            stopping_criteria=stopping_criteria,
            generation_config=self.generation_config,
            return_dict_in_generate=True,
            output_scores=True
        )

        response = self.tokenizer.decode(output['sequences'][0], skip_special_tokens=True)
        return response

    def get_stopping_criteria(self, stop_tokens):
        truncate_length = len(self.tokenizer(f'\n')['input_ids'])
        if stop_tokens:
            if self.model_name == "Starling-LM-7B-alpha":
                # Build stop_token_ids and filter out any empty ones.
                stop_token_ids = [
                    torch.LongTensor(self.tokenizer(f'\n{stop_token}')['input_ids'][truncate_length:]).cuda()
                    for stop_token in stop_tokens
                    if len(self.tokenizer(f'\n{stop_token}')['input_ids'][truncate_length:]) > 0
                ]

                class StopOnTokens(StoppingCriteria):
                    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                        # Check for empty input_ids
                        if input_ids.size(0) == 0:
                            return False

                        for stop_ids in stop_token_ids:
                            # Only check if input_ids is long enough
                            if input_ids.size(1) >= len(stop_ids):
                                if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                                    return True
                        return False

                return StoppingCriteriaList([StopOnTokens()])
            elif self.model_name == "Llama-3.1-8B-Instruct":
                # Build stop_token_ids and filter out any empty ones.
                stop_token_ids = [
                    torch.tensor(self.tokenizer.encode("\n" + stop_token, add_special_tokens=False)).to('cuda')
                    for stop_token in stop_tokens
                    if len(self.tokenizer.encode("\n" + stop_token, add_special_tokens=False)) > 0
                ]

                class StopOnTokens(StoppingCriteria):
                    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                        # Check for empty input_ids
                        if input_ids.size(0) == 0:
                            return False

                        for stop_ids in stop_token_ids:
                            # Only check if input_ids is long enough
                            if input_ids.size(1) >= len(stop_ids):
                                if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                                    return True
                        return False

                return StoppingCriteriaList([StopOnTokens()])
            else:
                # Build stop_token_ids and filter out any empty ones.
                stop_token_ids = [
                    torch.tensor(self.tokenizer.encode(stop_token, add_special_tokens=False)).to('cuda')
                    for stop_token in stop_tokens
                    if len(self.tokenizer.encode(stop_token, add_special_tokens=False)) > 0
                ]

                class StopOnTokens(StoppingCriteria):
                    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                        # Check for empty input_ids
                        if input_ids.size(0) == 0:
                            return False

                        for stop_ids in stop_token_ids:
                            # Only check if input_ids is long enough
                            if input_ids.size(1) >= len(stop_ids):
                                if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                                    return True
                        return False

                return StoppingCriteriaList([StopOnTokens()])
        else:
            return None

df = pd.read_json('data/summarization_sampled_data.json', lines=True)
documents = df['document'].tolist()


task = "summarization"
output_dir = Path(f'results/{task}/generated/{args.model}')
output_dir.mkdir(parents=True, exist_ok=True)

generated_summaries = []

llm = LLMCompletion(args.model)


for doc in df['document']:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Without any explanation, only give a concise and comprehensive summary of this article directly. (100 words max) Article: {doc}"},
    ]


    if args.model in ["Llama-3.1-8B-Instruct", "Qwen2.5-7B-Instruct", "internlm3-8b-instruct"]:
        response = llm(messages, ["\n\n\n"])
        response = response.split("assistant\n")[1]
    elif args.model == "DeepSeek-R1-Distill-Llama-8B":
        response = llm(messages, ["\n\n\n"])
        # Extracted text
        response =  response.split("</think>")[1]
    elif args.model in ["gemma-7b-it", "Mistral-7B-Instruct-v0.3"]:
        messages = messages[1:]
        response = llm(messages, ["\n\n\n"])
        # Extracted text
        response =  response.strip().split("\n")[-1]
    elif args.model == "Llama-2-7b-chat-hf":
        response = llm(messages, ["\n\n\n"])
        response = response
    else:
        raise NotImplementedError
    generated_summaries.append(response.strip())
    print(response.strip(), "\n")

df = df.assign(generated_summary=generated_summaries)
df[['generated_summary']].to_json(
    output_dir / 'generated_summaries.json',
    orient='records',
    lines=True,
    force_ascii=False
)