# %%
# 把csv文件转换成json文件
import pandas as pd
import json
import re

# 正则表达式修改
def extract_contents(conv):
    conv_regex = r'\[Human\]: (.*?)\r\n\[AI\]: (.*)'
    matches = re.findall(conv_regex, conv)
    if matches:
        instruction = matches[0][0]
        output = matches[0][1]
        return instruction, output
    else:
        return None, None

def process_convs(row):
    convs = []
    for conv in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']:
        instruction, output = extract_contents(row[conv])
        if instruction and output:
            convs.append([instruction, output])
    return convs

def main():
    df_ = pd.read_csv('Data/online_QA_eco(1).csv',encoding='utf-8')
    result = []
    for index, row in df_.iterrows():
        convs = process_convs(row)
        for i, conv in enumerate(convs, start=1):
            if i == 1:
                history = []
            else:
                history = convs[:i-1]
            conv_json = {
                "instruction": conv[0],
                "output": conv[1],
                "history": history
            }
            result.append(conv_json)

    with open('Data/my_data.json', 'w', encoding='utf-8') as file:
        json.dump(result, file, indent=4, ensure_ascii=False)

if __name__ =='__main__':
    main()
