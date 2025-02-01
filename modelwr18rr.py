import os
import torch
import re
import json
import gdown
from datasets import Dataset
import pandas as pd
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig, GenerationConfig
from tqdm.auto import tqdm
from trl import DPOTrainer
from transformers import TrainerCallback
# from trl import SFTConfig
# from trl import SFTTrainer
# from trl import DPOConfig

import argparse
import json
import logging
import os
import re
import time
import random
from tqdm import tqdm 
import multiprocessing as mp
from collections import defaultdict
def process_output(output):

    
    while "[" in output or "]" in output:
        if "[" in output:
            output = output[1:]  # 去掉第一个字符
        if "]" in output:
            output = output[:-1]  # 去掉最后一个字符

    # print(33333)
    print(output)
    # 分割成多个实体
    entities = output.split(",")
    
    # 去掉每个实体的首尾空格
    entities = [entity.strip() for entity in entities]
    entities = [entity.strip(',.，。;!?！？') for entity in entities]
    entities = [entity.strip() for entity in entities]
    # print(444444444)
    # print(entities)
    # 将实体转换为整型ID
    ents = [text2ent.get(text, 99999) for text in entities]
    ids = [ent2id.get(entity,99999) for entity in ents]
    # ents = [text2ent(text) for text in entities]
    # ids = [ent2id(entity) for entity in ents]
    
    return ids
candidate_answers = []
id2ent = defaultdict(str)
ent2id = defaultdict(str)
rel2id= defaultdict(str)
text2ent = defaultdict(str)
ent2text = defaultdict(str)
all_candidate_answers = defaultdict(list)
rel2text_align = defaultdict(str)
rel2text = defaultdict(str)
scores = []    

def load_all_candidate_answers():
    global all_candidate_answers
    with open("./datasets/wn18rr" + "/retriever_candidate_tail" +".txt",'r') as load_f:
        all_candidate_answers=json.load(load_f)
        
def load_relation_text():
    global rel2text_align
    global rel2text
    with open("./datasets/wn18rr" + "/alignment/alignment_clean.txt",'r') as load_f:
        rel2text_align=json.load(load_f)  
    # with open("datasets/" + "fb15k-237" + "/relation2text.txt",'r') as load_f:
    #     rel2text=json.load(load_f)      
        
def load_rel_txt_to_id():
    global rel2id
    with open("./datasets/wn18rr" + '/get_neighbor/relation2id.txt', 'r') as file:
        relation_lines = file.readlines()
        for line in relation_lines:
            _name, _id = line.strip().split("\t")
            rel2id[_name] = _id
            
            
def load_ent_map_id():
    global ent2id
    global id2ent
    with open("./datasets/wn18rr" + '/get_neighbor/entity2id.txt', 'r') as file:
        entity_lines = file.readlines()
        for line in entity_lines:
            _name, _id = line.strip().split("\t")
            ent2id[_name] = _id
            id2ent[_id] = _name


def load_ent_to_text():
    global ent2text
    global text2ent
    with open("./datasets/wn18rr" + '/entity2text.txt', 'r') as file:
        entity_lines = file.readlines()
        for line in entity_lines:
            ent, text = line.strip().split("\t")
            ent2text[ent] = text
            text2ent[text] = ent


load_all_candidate_answers()
# print(len(all_candidate_answers))
prompt_data = []
with open("./datasets/wn18rr" + "/alignment/alignment.txt", 'r', encoding='utf-8') as file:
    for line in file:
        item = json.loads(line.strip())
        prompt_data.append(item)
def get_description(raw_value):
    for item in prompt_data:
        if item['Raw'] == raw_value:
            return item['Description']
    return "未找到对应的描述"
load_relation_text()

load_rel_txt_to_id()
load_ent_map_id()
load_ent_to_text()

def run_evaluation(model, tokenizer, generation_config, test_data, test_triplet, output_file):
    trained_model_response = []
    file_data = {}

    for data, sample in zip(test_data, test_triplet):
        tpe = sample['HeadEntity']
        question = sample['Question']
        headid = str(ent2id[tpe])
        relid = str(rel2id[question])
        key = f"{headid}\t{relid}"
        f=data["prompt"]
        # 生成模型输出
        candidate_ids = all_candidate_answers['\t'.join([str(ent2id[tpe]),str(rel2id[question])])]
        # print(candidate_ids)
        candidate_answers.clear()
        for id in candidate_ids[:20]:
            candidate_answers.append(ent2text[id2ent[str(id)]])
            # print(len(candidate_answers))
            # print(candidate_answers)
        candidiate_str = '[' + ','.join(candidate_answers)+ ']'
        
        inputs = tokenizer(rankk_formulate(data,candidiate_str,f), return_tensors="pt").to('cuda')
        output = model.generate(**inputs, generation_config=generation_config)
        output = tokenizer.batch_decode(output, skip_special_tokens=True)[0].split('[/INST] ')[1]
        trained_model_response.append(output)

        # 格式化输出为 ID 列表
        ids = process_output(output)
        file_data[key] = ids

    # 保存结果到文件
    with open(output_file, "w") as f:
        json.dump(file_data, f, indent=2)
# 调用评估指标
def evaluate_metrics(result_file, test_file):
    with open(result_file, 'r') as f:
        results = json.load(f)
    
    test_triplets = []
    with open(test_file, 'r') as f:
        for line in f:
            head, rel, tail = line.strip().split()
            test_triplets.append((head, rel, tail))
    
    total_samples = 0
    total_mrr = 0.0
    total_mr = 0.0
    hits1 = 0
    hits3 = 0
    hits10 = 0

    for head, rel, tail in test_triplets:
        key = f"{head}\t{rel}"
        if key in results:
            rankings = results[key]
            if tail in rankings:
                ranking = rankings.index(tail) + 1
            else:
                ranking = len(rankings) + 1
            
            total_samples += 1
            total_mrr += 1.0 / ranking
            total_mr += float(ranking)
            if ranking <= 1:
                hits1 += 1
            if ranking <= 3:
                hits3 += 1
            if ranking <= 10:
                hits10 += 1
    
    metrics = {
        'MRR': total_mrr / total_samples if total_samples > 0 else 0.0,
        'MR': total_mr / total_samples if total_samples > 0 else 0.0,
        'HITS@1': hits1 / total_samples if total_samples > 0 else 0.0,
        'HITS@3': hits3 / total_samples if total_samples > 0 else 0.0,
        'HITS@10': hits10 / total_samples if total_samples > 0 else 0.0,
    }
    return metrics
class CustomCallback(TrainerCallback):
    def __init__(self, model, tokenizer, test_data, test_triplet, generation_config, result_prefix="result"):
        self.model = model
        self.tokenizer = tokenizer
        self.test_data = test_data
        self.test_triplet = test_triplet
        self.generation_config = generation_config
        self.result_prefix = result_prefix

    def on_epoch_end(self, args, state, control, **kwargs):
        """在每个 epoch 结束时运行"""
        current_epoch = state.epoch
        
        if current_epoch % 10000 == 0:  # 每 100 个 epoch 运行测试
            print(f"\nRunning evaluation at epoch {int(current_epoch)}...")
            result_file = f"{self.result_prefix}_epoch{int(current_epoch)}.json"

            # 运行测试逻辑
            run_evaluation(self.model, self.tokenizer, self.generation_config, self.test_data, self.test_triplet, result_file)

            # 打印指标
            
            metrics = evaluate_metrics(result_file, "test2id.txt")
            epoch = int(current_epoch)

            file_name = f"metrics_epoch{epoch}_www.txt"

            # 打印指标到控制台
            print(f"Metrics at epoch {epoch}:")

            # 打开文件以写入模式
            with open(file_name, "w") as f:
                f.write(f"Metrics at epoch {epoch}:\n")
                
                # 将每个指标写入文件
                for key, value in metrics.items():
                    # 格式化并写入文件
                    f.write(f"{key}: {value:.4f}\n")
                    
                    # 也可以打印到控制台，如果需要
                    print(f"{key}: {value:.4f}")
#git clone https://github.com/Baiiiiiiiiii/GenAI_hw6_dataset.git
# Open and load the json dataset

num_epoch = 2
data_size = 3134
support_ratio = 1
with open("./qa_wn18rr.json", 'r') as jsonfile:
    full_data = json.load(jsonfile)

with open("./q_wn18rr.json", 'r') as jsonfile:
    test_data = json.load(jsonfile)

with open("./datasets/wn18rr" + "/test_answer.txt",'r') as load_f:
    test_triplet=json.load(load_f)
test_triplet = test_triplet[:data_size]
print("Totally %d test examples." % len(test_triplet))

model = AutoModelForCausalLM.from_pretrained(
    '/root/autodl-tmp/ori/Meta-Llama-3-8B-Instruct',
    device_map='auto',
    trust_remote_code=True,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4'
    )
)

tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/ori/Meta-Llama-3-8B-Instruct')
tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.eos_token

def data_formulate(data):
    messages = [
        # {"role": "system", "content": 'You are now an expert in sentence completion, and the answer is a word, or words, that you need to complete.What is [the answer]?'},
        {"role": "system", "content": 'You are an expert in sentence completion and your task is to provide the most accurate and contextually appropriate answer. The answer should be one or more words that best complete the given input. Ensure your response is concise, relevant, and formatted as a standalone output without additional explanation. Focus on understanding the context and providing the most logical answer to complete the sentence. Avoid including unnecessary details or rephrasing the input.'},
        {"role": "user", "content": data['prompt']},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt

def rank_formulate(data):


    messages =[
    {
    "role": "system",
    "content": "You are an assistant designed to reorder a list of candidate answers based on their likelihood of being correct. Your response must strictly adhere to the following requirements: \
    1. Provide the reordered list of candidate answers in the format: [most likely answer, ..., least likely answer]. \
    2. Do not include any additional text, explanation, or punctuation outside of the brackets. \
    3. Ensure that every candidate in the input list is included in the output. \
    4. Must contain []."
},
    {
    "role": "user",
    "content": ('The list of candidate answers is {candidiate_str}. The question is: {data_prompt}.').format(candidiate_str=candidiate_str, data_prompt=data["prompt"])
}
#  Please reorder all candidate answers based on their likelihood of being correct. Provide the output strictly in the format: [most likely answer, ..., least likely answer].
    # print(data["prompt"])
    # messages = [
    #     # {"role": "system", "content": 'The list of candidate answers is {candidiate_str}. And the question is {data["prompt"]}. Now, based on the previous examples and your own knowledge and thinking, sort the list to let the candidate answers which are more possible to be the true answer of the question more prior. Output the sorted order of candidate answers using the format \"[most possible answer | second possible answer | ... | least possible answer]\" and please start your response with \"The final order:\". Do not output anything except the final order. Note your output sorted order should contain all the candidates in the list but not add new answer to it.'},
    #  #   {"role": "system", "content": ('The list of candidate answers is {candidiate_str}. The question is: {data_prompt}.Sort all candidate answers by their likelihood of being correct. Ensure every candidate in the list is included in the output.Output the sorted order of all candidate answers in the format: \"[most likely answer , ... , least likely answer]\".Do not exclude any candidates. Start your response with \"The final order:\".Do not output anything except the final order.').format(candidiate_str=candidiate_str, data_prompt=data["prompt"])},
    #     # {"role": "system", "content": ('The list of candidate answers is {candidiate_str}. The question is: {data_prompt}.Sort all candidate answers by their likelihood of being correct. Ensure every candidate in the list is included in the output.Output the sorted order of all candidate answers in the format: \"[most likely answer | ... | least likely answer]\".Do not exclude any candidates and make sure there is only one [] and they are separated by | in [].').format(candidiate_str=candidiate_str, data_prompt=data["prompt"])},
    #     {"role": "system", "content": ('The list of candidate answers is {candidiate_str}. The question is: {data_prompt}.Sort all candidate answers by their likelihood of being correct. Ensure every candidate in the list is included in the output.Output the sorted order of all candidate answers in the format: \"[most likely answer , ... , least likely answer]\".Start your answer with [, end with ].This [] contains only the reordered list of candidate answers.').format(candidiate_str=candidiate_str, data_prompt=data["prompt"])},
    #     {"role": "user", "content": "What is the modified list of candidate answers?"},
    # ]

]



    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


# original_model_response = []
# for data in tqdm(test_data):
#     id = data['id']
#     # print(f'Question {id}:\n'+data['prompt'])
#     inputs = tokenizer(data_formulate(data), return_tensors="pt").to('cuda')
#     generation_config=GenerationConfig(
#             do_sample=False,
#             max_new_tokens = 200,
#             pad_token_id = tokenizer.pad_token_id
#     )
#     output = model.generate(**inputs, generation_config=generation_config)
#     output = tokenizer.batch_decode(output, skip_special_tokens=True)[0].split('[/INST] ')[1]
#     original_model_response.append(output)
#     print('Response from original model:\n'+output+'\n')

# num_epoch = 500
# data_size = 100
# support_ratio = 1

full_data=full_data[:data_size]
test_data=test_data[:data_size]
# Select part of the data for training
training_data = full_data[:data_size]

# Define the size of the support dataset
support_data_size = int(data_size * support_ratio)

# Prepare the data for the training dataset
prompt_list = [data_formulate(data) for data in training_data]
chosen_list = [data['support'] for data in training_data[:support_data_size]] + [data['oppose'] for data in training_data[support_data_size:]]
rejected_list = [data['oppose'] for data in training_data[:support_data_size]] + [data['support'] for data in training_data[support_data_size:]]
position_list = ['support' for _ in range(support_data_size)] + ['oppose' for _ in range(data_size - support_data_size)]

# Create the training dataset
train_dataset = Dataset.from_dict({'prompt': prompt_list, 'position': position_list, 'chosen': chosen_list, 'rejected': rejected_list})
pd.DataFrame(train_dataset).rename(columns={"chosen": "preferred", "rejected": "non-preferred"})
# training_args = TrainingArguments(
#     output_dir='./',
#     per_device_train_batch_size=4,
#     num_train_epochs=num_epoch,
#     gradient_accumulation_steps=8,
#     gradient_checkpointing=False,
#     learning_rate=2e-4,
#     # learning_rate=1e-7,
#     optim="paged_adamw_8bit",
#     logging_steps = 1,
#     warmup_ratio = 0.1,
#     report_to = 'none',

#     # model_init_kwargs='None'
# )
training_args = TrainingArguments(
    output_dir='./',                
    per_device_train_batch_size=2,       
    num_train_epochs=num_epoch,          
    gradient_accumulation_steps=8,       
    gradient_checkpointing=False,        
    # learning_rate=2e-4, 
    learning_rate=3e-4,     
    optim="paged_adamw_8bit",            
    logging_steps=1,                     
    warmup_ratio=0.1,                    
    report_to='none',                    
    # save_steps=100,                    # 每100步保存一次检查点
    # save_total_limit=3,                # 限制最多保存3个检查点
)
generation_config = GenerationConfig(
    do_sample=False,
    max_new_tokens=500,
    pad_token_id=tokenizer.pad_token_id
)
custom_callback = CustomCallback(
    model=model,
    tokenizer=tokenizer,
    test_data=test_data[:10],  # 只测试前 100 个样本
    test_triplet=test_triplet[:10],
    generation_config=generation_config,
    result_prefix="result"
)
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

dpo_trainer = DPOTrainer(
    model,
    args=training_args,
    beta=0.1,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    peft_config=peft_config,
    # callbacks=[custom_callback]  # 添加自定义回调
)


# training_args = DPOConfig(
#     output_dir='./',
#     per_device_train_batch_size=1,
#     num_train_epochs=num_epoch,
#     gradient_accumulation_steps=8,
#     gradient_checkpointing=False,
#     learning_rate=2e-4,
#     optim="paged_adamw_8bit",
#     logging_steps = 1,
#     warmup_ratio = 0.1,
#     report_to = 'none'
# )





# dpo_trainer = DPOTrainer(
#     model,
#     args=training_args,
#     beta=0.1,
#     train_dataset=train_dataset,
#     tokenizer=tokenizer,
#     peft_config=peft_config,
# )




dpo_trainer.train()

######################

# with open("/root/autodl-tmp/datasets/fb15k-237" + "/test_answer.txt",'r') as load_f:
#     test_triplet=json.load(load_f)
# test_triplet = test_triplet[:data_size]
# print("Totally %d test examples." % len(test_triplet))

output_data = []
output_question = []
id_counter = 1

trained_model_response_rank = []
trained_model_response = []
for data, sample in tqdm(zip(test_data, test_triplet), total=len(test_data)):
    tpe = sample['HeadEntity'] #if args.query == 'tail' else sample['Answer']
    question = sample['Question']
    support = sample['Answer']

    tpe_str = ent2text[tpe]
    # print(tpe_str)
    description = get_description(question)
    candidate_ids = all_candidate_answers['\t'.join([str(ent2id[tpe]),str(rel2id[question])])]
    # print(candidate_ids)
    candidate_answers.clear()
    for id in candidate_ids[:20]:
        candidate_answers.append(ent2text[id2ent[str(id)]])
        # print(len(candidate_answers))
        # print(candidate_answers)
    candidiate_str = '[' + ','.join(candidate_answers)+ ']'
    # print(candidiate_str)


    id = data['id']
    # print(f'Question {id}:\n'+data['prompt'])
    inputs = tokenizer(data_formulate(data), return_tensors="pt").to('cuda')
    inputrank = tokenizer(rank_formulate(data), return_tensors="pt").to('cuda')
    # generation_config=GenerationConfig(
    #         do_sample=False,
    #         max_new_tokens = 500,
    #         pad_token_id = tokenizer.pad_token_id
    # )
    output = model.generate(**inputs, generation_config=generation_config)
    output = tokenizer.batch_decode(output, skip_special_tokens=True)
    # print(f'1{output}')
    output = [text.split('assistant\n\n', 1)[1]  # 提取 assistant\n\n 后的内容
            .replace('.', '')              # 去掉句号
        for text in output]
    
    # print(output[0])
    trained_model_response.append(output[0])
    # print(f'2{output}')
    
    outputrank = model.generate(**inputrank, generation_config=generation_config)
    outputrank = tokenizer.batch_decode(outputrank, skip_special_tokens=True)
    # print(f'3{outputrank}')
    
    outputrank = [text.split('assistant\n\n', 1)[1]  # 提取 assistant\n\n 后的内容
            # .replace('.', '')              # 去掉句号
        for text in outputrank]
    # print(outputrank)
    if '[' in outputrank and ']' in outputrank: 
        outputrank = [text[text.index('[') + 1:text.index(']')] for text in outputrank]
        outputrank=outputrank[0]
    outputrank=outputrank[0]
    # print(outputrank)
    trained_model_response_rank.append(outputrank)


model_response = []
print(f'num_epoch: {num_epoch}\ndata_size: {data_size}\nsupport_ratio: {support_ratio}')
# print()

file_data = defaultdict(list)
output_file_path = "1jieduan.txt"
# file_data = {}
with open(output_file_path, "a") as file:
    for data,sample in zip(test_data,test_triplet):
        tpe = sample['HeadEntity'] #if args.query == 'tail' else sample['Answer']
        question = sample['Question']
        headid=str(ent2id[tpe])
        relid=str(rel2id[question])
        key = f"{headid}\t{relid}"

        # 获取训练和原始模型的输出
        id = data['id']
        output = trained_model_response[id - 1]
        # print(output)
        outputr = trained_model_response_rank[id-1]
        # print(outputr)
        # print(output)
        # 输出格式化为列表
        idsrank = process_output(outputr)
        # 方法一
        output = output.strip() 
        output = output.strip(',.，。;!?！？')
        output = output.strip() 
        
        ents = text2ent.get(output, 99999)
        ids = ent2id.get(ents,99999)
        
        file.write(f"{headid} {relid} {ids}\n")#方法1
        # 构建最终的JSON数据结构
        #counter = 1
        #new_key = key  # 初始时使用原始的 key

        # 检查当前 key 是否已经存在，如果已存在则增加计数器
        #while f"{new_key}_{counter}" in file_data:
          #  counter += 1
        
        # 构建新的 key
        #new_key = f"{key}_{counter}"
        file_data[key] = idsrank 

        # counter = 1
        # new_key = key  # 初始时使用原始的 key

        # # 检查当前 key 是否已经存在，如果已存在则增加计数器
        # while f"{new_key}_{counter}" in file_data:
        #     counter += 1
        
        # # 构建新的 key
        # new_key = f"{key}_{counter}"

        # # 将当前的 ids 添加到新的 key 中
        # file_data[new_key].append(ids)


        # ref_output = original_model_response[id-1]
        output = trained_model_response[id-1]
        print(f'Question {id}:\n'+data['prompt'])

        print('Response from trained model:\n'+output)
        print('Response from trained model rank:\n'+outputr)
        # print()
        # model_response.append({'id':data['id'], 'prompt':data['prompt'], 'response_from_original_model':ref_output, 'response_from_trained_model':output})
        model_response.append({'id':data['id'], 'prompt':data['prompt'], 'response_from_trained_model':output})

with open(f"epoch-{num_epoch}_size-{data_size}_ratio-{support_ratio}.json", "w", encoding='UTF-8') as outfile:
    json.dump(model_response, outfile, indent=4, ensure_ascii=False)
resultfile_name = f"result_epoch_{num_epoch}_size_{data_size}.json"
with open(resultfile_name, "w") as f:
    json.dump(file_data, f, indent=2)