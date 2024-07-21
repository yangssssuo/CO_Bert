import pandas as pd
import numpy as np
def split_csv_file(file_name):
    # 读取csv文件
    df = pd.read_csv(file_name,header=None,index_col=None)
    file_path = '/home/yanggk/Data/CO_Bert/ElecProp/'
    eads_path = '/home/yanggk/Data/CO_Bert/Eads/'
    # 获取标题行
    print(df)
    # header = df.columns.tolist()
    # header = header[1:]

    # 遍历每一行
    for index, row in df.iterrows():
        # 使用第一列的值作为新的csv文件的文件名
        fea_name = str(row[0])
        old_feature = pd.read_csv(f'/home/yanggk/Data/CO_Bert/Property/{fea_name}.csv',index_col=0).values.squeeze().tolist()
        # print(old_feature.shape)
        eads_file_name = eads_path + str(row[0]) + '.csv'
        eads = old_feature[0]
        with open(eads_file_name,'w') as w:
            w.write(str(eads))

        old_feature = old_feature[1:]
        # print(type(old_feature))
        
        new_file_name = file_path + str(row[0]) + '.csv'
        # print(row)
        row = row[1:].to_list()
        # print(type(row))
        # print(row.shape)
        elec_feature = old_feature + row
        elec_feature = pd.DataFrame(elec_feature).T
        # print(elec_feature.T)
        # print(row.shape)
        # 将标题行和当前行保存为新的csv文件
        elec_feature.to_csv(new_file_name, index=False, header=False)

# 使用你的csv文件名替换'your_file.csv'
split_csv_file("/home/yanggk/Data/CO_Bert/test/electronic-prop.csv")
