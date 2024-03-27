import re
import os
import json
import pandas as pd
import logging
from openai import OpenAI
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
os.environ["http_proxy"] = "http://localhost:33210"
os.environ["https_proxy"] = "http://localhost:33210"   # 开梯子时要加上，33210是我电脑上梯子的端口
load_dotenv()
api_key = 'sk-60kV9nDoV5LSZfeq24Fc7f63F1Bc4216A091E7Fb45EeA604'
base_url = 'https://api.chatgptid.net/v1'
kwargs = {"temperature": 0.3}
num_round = 3
batch_size = 10
start_row = 0

class OpenAILLM():
    def __init__(self, api_key, base_url, kwargs):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.kwargs = kwargs
      
    def generate(self, prompt, **kwargs):
        messages = [{"role": "user", "content":prompt}]
        response = self.client.chat.completions.create(
            model = "gpt-3.5-turbo-1106",
            messages = messages,
            **kwargs
            )
        result = response.choices[0].message.content
        return result

def generate_conv(gpt, Q_prompt_template,A_prompt_template, row, num_round):
    '''
    为一行row生成num_rounds轮对话
    '''
    history = []
    convs = []
    for i in range(1,num_round+1):
        input = json.dumps({"post": row['post'], "history": history},ensure_ascii=False)
        Q_prompt = Q_prompt_template.replace('<input>', input)
        question = gpt.generate(Q_prompt)
        A_prompt = A_prompt_template.replace('<question>', question)
        answer = gpt.generate(A_prompt)
        conv_json = {
            "instruction": question,
            "output": answer,
            "history": history.copy()
        }
        convs.append(conv_json)
        history.append([question, answer])
    return convs  

def main():
    gpt = OpenAILLM(api_key,base_url,kwargs) 
    with open("prompts\\gen_Q_template.txt", 'r', encoding='utf-8') as prompt_file:
        Q_prompt_template = prompt_file.read()
    with open("prompts\\gen_A_template.txt", 'r', encoding='utf-8') as prompt_file:
        A_prompt_template = prompt_file.read()

    df_ = pd.read_csv(r'Data\filtered_questions.csv', encoding='utf-8')  
    result = []
    for index, row in df_.iloc[start_row:].iterrows():
        logging.info(f'Processing {index}...')
        convs = generate_conv(gpt, Q_prompt_template,A_prompt_template, row, num_round)
        result.extend(convs)
        
        if (index - start_row + 1) % batch_size == 0:
            with open(f'Data\\from_{index + 1 - batch_size}_to_{index}.json', 'w', encoding='utf-8') as file:
                json.dump(result, file, indent=4, ensure_ascii=False)
            result = []
        
if __name__ == "__main__":
    main()

"""
生成逻辑：逐帖子产生对话，即产生一个帖子的n轮，再产生下一个帖子的n轮，以此类推，每完成m个帖子的n轮对话，就保存一次
"""

# %%
import json
result = []
for file_name in os.listdir('Data'):
    if file_name.startswith('from_') and file_name.endswith('.json'):
        with open(f'Data/{file_name}', 'r', encoding='utf-8') as file:
            data = json.load(file)
            result.extend(data)

with open('Data\\online_QA.json', 'w', encoding='utf-8') as file:
    json.dump(result, file, indent=4, ensure_ascii=False)
    
# 计算output的平均长度
with open('Data\\online_QA.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

output_lengths = [len(item['output']) for item in data]
average_length = sum(output_lengths) / len(output_lengths)
print(f"Average output length: {average_length}")