# %%
import pandas as pd
import os
def filter1(filepath):
    dirname = os.path.dirname(filepath)
    df_ = pd.read_csv(filepath,encoding='utf-8')
    keywords = ['哪里','哪儿','跪求','考.*?证','考.*?研','保研','外保','考博','就业','分享', '共享','作业', '答案', '讲义', 
                '有偿','职业','证书','考试','试卷','题库','复习','考生','硕士','博士','研究生','人大','专硕','学硕','PHD','phd',
                'CPA','CFA','ACCA','cpa','cfa','acca','会计师','税务师','含金量','国泰安','留学','中国人民大学','wind数据库',
                '在哪找','谁有','期刊','杂志','文献','投稿','终审','初审','外审','定稿','退稿','审稿','审阅','核心','会议','翻译',
                '求数据','求论文','求文献','求资料','求书','求.*?年.*?数据','资料','推荐','年.*?省','求.*?资料','求.*?书','大学(?![生])',
                '教材', '课本', '主编', '版', '试题', '书籍', '下载','统计年鉴','代码', '课后', '习题', '网课','支付.*?报酬']
    df_ = df_[~df_['help_post'].str.contains('|'.join(keywords))]
    df_ = df_[df_['help_post'].str.contains('[\u4e00-\u9fa5]')]
    df_ = df_[~df_['help_post'].str.contains('[a-zA-Z\s]{20,}')]  
    # help_post用正则表达式筛选，(有没有|求|查|获取|得到)*?(数据|论文|课件)
    df_ = df_[df_['help_post'].str.len() >= 15]
    df_ = df_[df_['help_post'].str.len() <= 400]
    df_ = df_.drop_duplicates(subset=['help_post'], keep='first')
    df_ = df_.dropna(subset=['help_post'])
    file_name = os.path.join(dirname, 'filtered_' + os.path.basename(filepath))
    df_.to_csv(file_name, encoding='utf-8', index=False)

filter1(r'F:\thesis\online_QA\classification_glm3\result（三轮）.csv')



# %%
# 合并数据
import os
import pandas as pd
folder_list =[
    r'classification_glm3\classification\财会',
    r'classification_glm3\classification\金融',
    r'classification_glm3\classification\统计',
    r'classification_glm3\classification\经济',
    r'classification_glm3\classification\国际贸易',
    r'classification_glm3\classification\管理',
]


final_dfs = []
for folder in folder_list:
    dfs = []
    file_list = os.listdir(folder)
    for file_name in file_list:
        if file_name.startswith('clf'):
            try:
                temp_df = pd.read_csv(os.path.join(folder, file_name), encoding='utf-8')
                # os.remove(os.path.join(folder, file_name))
                dfs.append(temp_df)
            except:
                pass
    final_df = pd.concat(dfs)
    final_df['学科'] = folder.split('\\')[-1]
    final_dfs.append(final_df)
final_dfs = pd.concat(final_dfs)
final_dfs.to_csv(r'classification_glm3\classification\glm3_merged_data.csv', encoding='utf-8', index=False)

