import re
import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
os.environ["http_proxy"] = "http://127.0.0.1:33210"
os.environ["https_proxy"] = "http://127.0.0.1:33210"

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

def get_prompt(prompt_template, help_post, *convs):
    '''
    如果convs为空，则返回第一轮对话的prompt
    如果convs长度为1，则返回第二轮对话的prompt
    如果convs长度为2，则返回第三轮对话的prompt，以此类推
    '''
    round_num = len(convs) + 1
    prompt = re.findall(fr'([\w\W]+对话{round_num}：)', prompt_template)[0]
    prompt = prompt.replace('{help_post}', help_post)
    for i, conv in enumerate(convs):
        prompt = prompt.replace(f'{{conv{i+1}}}', conv)
    return prompt

def generate_conv(gpt, prompt_template, df, num_round):
    '''
    生成第num_rounds轮对话
    '''
    prompts = []
    for i in range(len(df)):
        help_post = df['help_post'][i]
        convs = []
        for j in range(1,num_round+1):
            if j >1:
                convs.append(df[f'conv{j-1}'][i])
        prompt = get_prompt(prompt_template, help_post, *convs)
        prompts.append(prompt)
    
    responses = []
    for prompt in prompts:
        response = gpt.generate(prompt)
        responses.append(response)
    
    df[f'conv{num_round}'] = responses
    return df

def main():
    load_dotenv()
    api_key = os.environ.get("api_key")
    base_url = os.environ.get("base_url")
    kwargs = {
        "temperature": 0.3
    }
    gpt = OpenAILLM(api_key,base_url,kwargs)
    
    with open("prompt_template.txt", 'r', encoding='utf-8') as prompt_file:
        prompt_template = prompt_file.read()

    df_ = pd.read_csv(r'online_QA.csv', encoding='utf-8')
    df_ = generate_conv(gpt, prompt_template, df_, 1)
    df_ = generate_conv(gpt, prompt_template, df_, 2)
    df_ = generate_conv(gpt, prompt_template, df_, 3)
    df_.to_csv('online_QA.csv', encoding='utf-8', index=False)


if __name__ == "__main__":
    main()

"""
现在是逐轮产生对话，即产生完所有帖子的第一轮对话，再产生所有帖子的第二轮对话，以此类推
后面改成逐帖子产生对话可能更好，即产生一个帖子的n轮，再产生下一个帖子的n轮，以此类推，每完成m个帖子的n轮对话，就保存一次
"""
