import pickle

# rb是2进制编码文件，文本文件用r
f = open('demo/data/sunrgbd/sunrgbd_000017_infos.pkl','rb')
data = pickle.load(f)
print(data)
