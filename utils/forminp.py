import pandas as pd

aaa = pd.read_csv(f'/home/yanggk/Data/CO_Bert/new_data.csv')
# print(aaa)


for idx in range(len(aaa)):
    # with open()
    line = aaa.iloc[idx:idx+1,:].values.reshape(-1).tolist()
    name = line[0] + '-'+ line[1]
    struc = aaa.iloc[idx:idx+1,26:].to_csv(f'/home/yanggk/Data/CO_Bert/Structure/{name}.csv')
    prop = aaa.iloc[idx:idx+1,20:26].to_csv(f'/home/yanggk/Data/CO_Bert/Property/{name}.csv')
    # print(prop)
    
    # print(line)
    with open(f'/home/yanggk/Data/CO_Bert/IR/inps/{name}.inp','w') as wIR:
        wIR.write('6\t1\n')
        for i in range(2,8):
            wIR.write(str(line[i])+'\t'+str(line[i+6])+'\n')


    with open(f'/home/yanggk/Data/CO_Bert/Raman/inps/{name}.inp','w') as wRa:
        wRa.write('6\t1\n')
        for i in range(2,8):
            wRa.write(str(line[i])+'\t'+str(line[i+12])+'\n')

