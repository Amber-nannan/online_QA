# %%
import pandas as pd
import os
import logging
import torch
import multiprocessing
from modelscope import AutoTokenizer, AutoModel
logging.basicConfig(level=logging.INFO)
torch.cuda.set_device(0)

batch_size = 20
# 加载模型和分词器
model_dir = '/home/chensiting/models/chatglm3-6b'
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).half().cuda()
model = model.eval()


def clf(df_,prompt_template):
    questions = df_['help_post'].tolist()
    responses = []
    for question in questions:
        prompt = prompt_template.replace('{question}', question)
        response, _ = model.chat(tokenizer, prompt, history=None)
        responses.append(response)
    df_[f'质量高低'] = responses
    return df_


# gpt_api调用
def task(begin_line, input_file):
    with open("classifier_template.txt", 'r', encoding='utf-8') as prompt_file:
        prompt_template = prompt_file.read()
    
    df_ = pd.read_csv(input_file,encoding='utf-8')
    
    for i in range(begin_line, len(df_), batch_size):
        folder = os.path.dirname(input_file)
        logging.info(f'Processing {folder} batch {i // batch_size + 1}...')
        batch_df = df_.iloc[i:i + batch_size]
        batch_df = clf(batch_df,prompt_template)
        
        file_name = rf'{folder}\classification_{i // batch_size + 1}.csv'
        batch_df.to_csv(file_name, encoding='utf-8', index=False)

def main():
    processes = []  
    tasks_params = [
        (0, '/home/chensiting/test/classification/clf1/filtered_clf1.csv'),
        (0, '/home/chensiting/test/classification/clf2/filtered_clf2.csv'),
        (0, '/home/chensiting/test/classification/clf3/filtered_clf3.csv'),
        (0, '/home/chensiting/test/classification/clf4/filtered_clf4.csv')
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

