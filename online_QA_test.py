import re
import os
import json
import pandas as pd
import logging
from openai import OpenAI
from dotenv import load_dotenv

os.environ["http_proxy"] = "http://localhost:33210"
os.environ["https_proxy"] = "http://localhost:33210"   # 开梯子时要加上，33210是我电脑上梯子的端口

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

def generate_conv(gpt, prompt_template, row, num_round):
    '''
    为一行row生成num_rounds轮对话
    '''
    history = []
    convs = []
    for i in range(1,num_round+1):
        input = json.dumps({"post": row['post'], "history": history},ensure_ascii=False)
        prompt = prompt_template.replace('<input>', input)
        question = gpt.generate(prompt)
        answer = gpt.generate('请你以财经专家或统计专家的身份回答问题：'+question)
        conv_json = {
            "instruction": question,
            "output": answer,
            "history": history.copy()
        }
        convs.append(conv_json)
        history.append([question, answer])
    return convs  

def main():
    load_dotenv()
    api_key = os.environ.get('api_key')
    base_url = os.environ.get('base_url')
    kwargs = {
        "temperature": 0.3
    }
    gpt = OpenAILLM(api_key,base_url,kwargs)
    num_round = 3
    batch_size = 10
    start_row = 0
    
    with open("prompts\\gen_Q_template.txt", 'r', encoding='utf-8') as prompt_file:
        prompt_template = prompt_file.read()
    
    df_ = pd.read_csv(r'Data\filtered_questions.csv', encoding='utf-8')  
    result = []
    for index, row in df_.iloc[start_row:].iterrows():  # 从选定的行开始迭代
        logging.info(f'Processing {index}...')
        convs = generate_conv(gpt, prompt_template, row, num_round)
        result.extend(convs)
        
        if (index - start_row + 1) % batch_size == 0:
            start_index = index + 1 - batch_size
            end_index = index
            with open(f'Data\\from_{start_index}_to_{end_index}.json', 'w', encoding='utf-8') as file:
                json.dump(result, file, indent=4, ensure_ascii=False)
            result = []
        
if __name__ == "__main__":
    main()

"""
生成逻辑：逐帖子产生对话，即产生一个帖子的n轮，再产生下一个帖子的n轮，以此类推，每完成m个帖子的n轮对话，就保存一次
"""
# %%
# # 计算'Data\\from_0_to_3.json'中output的平均长度
# import json
# import pandas as pd
# with open('Data\\from_0_to_20.json', 'r', encoding='utf-8') as file:
#     data = json.load(file)

# output_lengths = [len(item['output']) for item in data]
# average_length = sum(output_lengths) / len(output_lengths)
# print(f"Average output length: {average_length}")
# %%
