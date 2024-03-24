# %%
import pandas as pd
import os
import logging
import multiprocessing
from openai import OpenAI
from dotenv import load_dotenv
from online_QA_test import OpenAILLM

os.environ["http_proxy"] = "http://localhost:33210"
os.environ["https_proxy"] = "http://localhost:33210"   # 开梯子时要加上，33210是我电脑上梯子的端口

batch_size = 20
api_key1 = 'sk-8R00Cji00bDcWyy895DfF9887104477b9015B51b3157C8E5'
api_key2 = 'sk-VktPzWjyo7Z3HYOw734a77C3AfD34f04B447BeCaA39dF274'
kwargs = {"temperature":0}
logging.basicConfig(level=logging.INFO)


def clf(df_,prompt_template,gpt):
    questions = df_['help_post'].tolist()
    responses = []
    for question in questions:
        prompt = prompt_template.replace('{question}', question)
        response = gpt.generate(prompt)
        responses.append(response)
    df_[f'质量高低'] = responses
    return df_

# gpt_api调用
def task(begin_line, api_key, base_url, kwargs, input_file):
    gpt = OpenAILLM(api_key, base_url, kwargs)
    with open("classifier_template.txt", 'r', encoding='utf-8') as prompt_file:
        prompt_template = prompt_file.read()
    
    df_ = pd.read_csv(input_file,encoding='utf-8')
    
    for i in range(begin_line, len(df_), batch_size):
        folder = os.path.dirname(input_file)
        logging.info(f'Processing {folder} batch {i // batch_size + 1}...')
        batch_df = df_.iloc[i:i + batch_size]
        batch_df = clf(batch_df,prompt_template,gpt)
        
        file_name = rf'{folder}\classification_{i // batch_size + 1}.csv'
        batch_df.to_csv(file_name, encoding='utf-8', index=False)

def main():
    load_dotenv()
    processes = []   
    base_url1 = 'https://api.132999.xyz/v1'
    base_url2 = 'https://api.chatgptid.net/v1'
    tasks_params = [
        (40,api_key1, base_url1, kwargs, r'classification_gpt\clf1\filtered_clf1.csv'),
        (40,api_key1, base_url1, kwargs, r'classification_gpt\clf2\filtered_clf2.csv'),
        (0,api_key1, base_url1, kwargs, r'classification_gpt\clf3\filtered_clf3.csv'),
        (0,api_key1, base_url1, kwargs, r'classification_gpt\clf4\filtered_clf4.csv')
    ]

    # 启动并行进程执行任务
    for begin_line, api_key, base_url, kw, input_file in tasks_params:
        process = multiprocessing.Process(target=task, args=(begin_line,api_key, base_url, kw, input_file))
        process.start()
        processes.append(process)

    # 等待所有进程执行完毕
    for process in processes:
        process.join()

if __name__ == '__main__':
    main()
