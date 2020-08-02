import os
import pandas as pd
import pickle
from collections import Counter
from data_process import split_text
from tqdm import tqdm
import jieba.posseg as psg
from cnradical import Radical,RunOption
import shutil
from random import shuffle
train_dir='ruijin_round1_train2_20181022'
def process_text(idx,split_method=None,split_name='train'):
    """
    读取文本  切割 然后打上标记  并提取词边界、词性、偏旁部首、拼音等文本特征
    :param idx: 文件的名字  不含扩展名
    :param split_method: 切割文本的方法   是一个函数
    :param split_name: 最终保存的文件夹名字
    :return:
    """
    data={}

    #------------------------------获取句子-----------------------------------
    if split_method is None:
        with open(f'datas/{train_dir}/{idx}.txt','r',encoding='utf-8') as f:
            texts=f.readlines()
    else:
        with open(f'datas/{train_dir}/{idx}.txt', 'r', encoding='utf-8') as f:
            texts=f.read()
            texts=split_method(texts)
    # data['word']=texts

    #---------------------------------获取标签----------------------------------
    tag_list=['O' for s in texts for x in s]
    tag =pd.read_csv(f'datas/{train_dir}/{idx}.ann',header=None,sep='\t')
    for i in range(tag.shape[0]):
        tag_item=tag.iloc[i][1].split(' ')#获取的实体类别以及起始位置
        cls,start,end=tag_item[0],int(tag_item[1]),int(tag_item[-1])#转换成对应的类型
        tag_list[start]='B-'+cls#其实位置写入B-实体类别
        for j in range(start+1,end):#后面的位置写I-实体类别
            tag_list[j]='I-'+cls
    assert len([x for s in texts for x in s])==len(tag_list)#保证两个序列长度一致

    text_list = ''
    for t in texts:
        text_list+=t
    textes = []
    tags = []
    start = 0
    end = 0
    max=len(tag_list)
    for s in texts:
        l = len(s)
        end += l
        if  end>=max or tag_list[end][0] != 'I':
            textes.append(text_list[start:end])
            tags.append(tag_list[start:end])
            start=end
    data['word']=textes
    data['label']=tags
    assert len([x for s in textes for x in s]) == len(tag_list)



    #-----------------------------提取词性和词边界特征----------------------------------
    word_bounds=['M' for item in tag_list]#首先给所有的字都表上B标记
    word_flags=[]#用来保存每个字的词性特征
    for text in textes:
        for word,flag in psg.cut(text):
            if len(word)==1:#判断是一个字的词
                start=len(word_flags)#拿到起始下标
                word_bounds[start]='S'#标记修改为S
                word_flags.append(flag)#将当前词的词性名加入到wordflags列表
            else:
                start=len(word_flags)#获取起始下标
                word_bounds[start]='B'#第一个字打上B
                word_flags+=[flag]*len(word)#将这个词的每个字都加上词性标记
                end=len(word_flags)-1#拿到这个词的最后一个字的下标
                word_bounds[end]='E'#将最后一个字打上E标记


    #--------------------------------------统一截断---------------------------------------
    bounds = []
    flags=[]
    start = 0
    end = 0
    for s in textes:
        l = len(s)
        end += l
        bounds.append(word_bounds[start:end])
        flags.append(word_flags[start:end])
        start += l
    data['bound'] = bounds
    data['flag']=flags


    #----------------------------------------获取拼音特征-------------------------------------
    radical=Radical(RunOption.Radical)#提取偏旁部首
    pinyin = Radical(RunOption.Pinyin)#用来提取拼音
    #提取偏旁部首特征  对于没有偏旁部首的字标上PAD
    data['radical']=[[radical.trans_ch(x) if radical.trans_ch(x) is not None else 'UNK' for x in s] for s in textes]
    # 提取拼音特征  对于没有拼音的字标上PAD
    data['pinyin'] = [[pinyin.trans_ch(x) if pinyin.trans_ch(x) is not None else 'UNK' for x in s] for s in textes]

    #------------------------------------------存储数据------------------------------------------------
    num_samples=len(textes)#获取有多少句话  等于是有多少个样本
    num_col=len(data.keys())#获取特征的个数 也就是列数

    dataset=[]
    for i in range(num_samples):
        records=list(zip(*[list(v[i]) for v in data.values()]))#解压
        dataset+=records+[['sep']*num_col]#每存完一个句子需要一行sep进行隔离
    dataset=dataset[:-1]#最后一行sep不要
    dataset=pd.DataFrame(dataset,columns=data.keys())#转换成dataframe
    save_path=f'data/prepare/{split_name}/{idx}.csv'

    def clean_word(w):
        if w=='\n':
            return 'LB'
        if w in [' ','\t','\u2003']:
            return 'SPACE'
        if w.isdigit():#将所有的数字都变成一种符号
            return 'num'
        return w
    dataset['word']=dataset['word'].apply(clean_word)
    dataset.to_csv(save_path,index=False,encoding='utf-8')


def multi_process(split_method=None,train_ratio=0.8):
    if os.path.exists('data/prepare/'):
        shutil.rmtree('data/prepare/')
    if not os.path.exists('data/prepare/trian/'):
        os.makedirs('data/prepare/train')
        os.makedirs('data/prepare/test')
    idxs=list(set([ file.split('.')[0] for file in os.listdir('datas/'+train_dir)]))#获取所有文件的名字
    shuffle(idxs)#打乱顺序
    index=int(len(idxs)*train_ratio)#拿到训练集的截止下标
    train_ids=idxs[:index]#训练集文件名集合
    test_ids=idxs[index:]#测试集文件名集合

    import multiprocessing as mp
    num_cpus=mp.cpu_count()#获取机器cpu的个数
    pool=mp.Pool(num_cpus)
    results=[]
    for idx in train_ids:
        result=pool.apply_async(process_text,args=(idx,split_method,'train'))
        results.append(result)
    for idx in test_ids:
        result=pool.apply_async(process_text,args=(idx,split_method,'test'))
        results.append(result)
    pool.close()
    pool.join()
    [r.get() for r in results]

def mapping(data,threshold=10,is_word=False,sep='sep',is_label=False):
    count=Counter(data)
    if sep is not None:
        count.pop(sep)
    if is_word:
        count['PAD']=100000001
        count['UNK']=100000000
        data = sorted(count.items(), key=lambda x: x[1], reverse=True)
        data=[ x[0]  for x in data if x[1]>=threshold]#去掉频率小于threshold的元素  未登录词
        id2item=data
        item2id={id2item[i]:i for i in range(len(id2item))}
    elif is_label:
        data = sorted(count.items(), key=lambda x: x[1], reverse=True)
        data = [x[0] for x in data]
        id2item = data
        item2id = {id2item[i]: i for i in range(len(id2item))}
    else:
        count['PAD'] = 100000001
        data = sorted(count.items(), key=lambda x: x[1], reverse=True)
        data = [x[0] for x in data]
        id2item = data
        item2id = {id2item[i]: i for i in range(len(id2item))}
    return id2item,item2id



def get_dict():
    map_dict={}
    from glob import glob
    all_w,all_bound,all_flag,all_label,all_radical,all_pinyin=[],[],[],[],[],[]
    for file in glob('data/prepare/train/*.csv')+glob('data/prepare/test/*.csv'):
        df=pd.read_csv(file,sep=',')
        all_w+=df['word'].tolist()
        all_bound += df['bound'].tolist()
        all_flag += df['flag'].tolist()
        all_label += df['label'].tolist()
        all_radical += df['radical'].tolist()
        all_pinyin += df['pinyin'].tolist()
    map_dict['word']=mapping(all_w,threshold=20,is_word=True)
    map_dict['bound']=mapping(all_bound)
    map_dict['flag']=mapping(all_flag)
    map_dict['label']=mapping(all_label,is_label=True)
    map_dict['radical']=mapping(all_radical)
    map_dict['pinyin']=mapping(all_pinyin)

    with open(f'data/prepare/dict.pkl','wb') as f:
        pickle.dump(map_dict,f)

if __name__ == '__main__':
    # print(process_text('0',split_method=split_text,split_name='train'))
    # multi_process()
    # print(set([ file.split('.')[0] for file in os.listdir('datas/'+train_dir)]))
    multi_process(split_text)
    get_dict()
    # with open(f'data/prepare/dict.pkl','rb') as f:
    #     data=pickle.load(f)
    # print(data['bound'])