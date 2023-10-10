import pandas as pd


aaa = pd.read_csv("/home/yanggk/Data/CO_Bert/data.csv")
# print(aaa)

# bbb = aaa.iloc[61498:61499,:].values
# bbb = bbb.reshape(-1).tolist()
# print(str(bbb))
# for i in bbb:
#     print(type(i))


with open("/home/yanggk/Data/CO_Bert/new_data.csv",'a') as w:
    for line in range(len(aaa)):
        xxx = aaa.iloc[line:line+1,:].values
        xxx = xxx.reshape(-1).tolist()
        handle = 0
        for i in xxx[2:]:
            try:
                float(i)
            except:
                print(line)
                handle = 1
                break
        if handle == 0:
            w.write(','.join(str(x) for x in xxx)+'\n')
        


        