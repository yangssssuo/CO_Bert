import pandas as pd
import torch
aaa = pd.read_csv("/home/yanggk/Data/CO_Bert/name.txt",header=None)
names = aaa.values.reshape(-1).tolist()
print(len(names))
# print(len(set(names)))
data_path = "/home/yanggk/Data/CO_Bert/new_data.csv"
bbb = pd.read_csv(data_path)
print(bbb)
# print(bbb['iz1'])
count = set()
for name in names:
    raman_path = '/home/yanggk/Data/CO_Bert/Raman/data/'+name+'.txt'
    data = pd.read_csv(raman_path,header=None,index_col=0,dtype='float32').values
    data = torch.tensor(data)
    # print(data.max())
    if data.max() != 100.0:
        count.add(name)
# for n,item in enumerate(bbb['izz6']):
#     if item >= 10000:
#         count.add(bbb['sys'][n]+'-'+bbb['name'][n])
# # print(len(count))
# for n,item in enumerate(bbb['izz5']):
#     if item >= 10000:
#         count.add(names[n])
# for n,item in enumerate(bbb['izz4']):
#     if item >= 10000:
#         count.add(names[n])
# for n,item in enumerate(bbb['izz3']):
#     if item >= 10000:
#         count.add(names[n])
# for n,item in enumerate(bbb['izz2']):
#     if item >= 10000:
#         count.add(names[n])
# for n,item in enumerate(bbb['izz1']):
#     if item >= 10000:
#         count.add(names[n])
for n,item in enumerate(bbb['Eads']):
    if item >= 10:
        count.add(bbb['sys'][n]+'-'+bbb['name'][n])

print(len(count))
washed = set(names) ^ count

print(len(washed))
with open("/home/yanggk/Data/CO_Bert/washed.txt",'w') as w:
    washed = list(washed)

    writelines = '\n'.join(washed)
    w.write(writelines)

# for name in names:
#     path = '/home/yanggk/Data/CO_Bert/Raman/data/' + name + '.txt'
#     data = pd.read_csv(path,header=None,index_col=0,dtype='float32').values
#     print(max(data))
#     if max(data) >=10000:
#         print(path)

# bbb = aaa.iloc[61498:61499,:].values
# bbb = bbb.reshape(-1).tolist()
# print(str(bbb))
# for i in bbb:
#     print(type(i))


# with open("/home/yanggk/Data/CO_Bert/new_data.csv",'a') as w:
#     for line in range(len(aaa)):
#         xxx = aaa.iloc[line:line+1,:].values
#         xxx = xxx.reshape(-1).tolist()
#         handle = 0
#         for i in xxx[2:]:
#             try:
#                 float(i)
#             except:
#                 print(line)
#                 handle = 1
#                 break
#         if handle == 0:
#             w.write(','.join(str(x) for x in xxx)+'\n')
        


        