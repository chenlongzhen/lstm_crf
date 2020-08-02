#-*-encoding=utf8-*-
import time
import requests
t=time.time()

r=requests.post('http://127.0.0.1:5002/ner?inputStr="格列喹酮通过与胰岛β细胞膜上磺脲受体结合促进胰岛素分泌[4],该结合作用短效可逆,避免了对细胞的持续刺激,降低了发生低血糖的风险。"')
print(r.text)