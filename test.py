# import collections
import numpy as np
import re
poetry_file = 'data/poetry_all.txt'
special_character_removal = re.compile(r'[^\w。， ]', re.IGNORECASE)
# 诗集
poetrys = []
count=0
with open(poetry_file, "r", encoding='utf-8', ) as f:
    for line in f:
        try:
            title, content = line.strip().split(':')
            content = special_character_removal.sub('', content)
            content = content.replace(' ', '')
            if len(content) < 5:
                continue
            if(len(content)>12*6):
                content_list=content.split("。")
                for i in range(0,len(content_list)-1,2):
                    count += 1
                    content_temp='[' + content_list[i]+"。"+content_list[i+1] + '。]'
                    content_temp=content_temp.replace("。。","。")
                    if len(content_temp) < 5:
                        continue
                    poetrys.append(content_temp)
            else:
                content = '[' + content + ']'
                poetrys.append(content)
        except Exception as e:
            print(e)

# 按诗的字数排序
poetrys = sorted(poetrys, key=lambda line: len(line))
print('唐诗总数: ', len(poetrys))
print(poetrys[0])
print(count)
#
# all_words = []
# for poetry in poetrys:
#     all_words += [word for word in poetry]
# counter = collections.Counter(all_words)
# count_pairs = sorted(counter.items(), key=lambda x: -x[1])
# words, _ = zip(*count_pairs)
# # 取前多少个常用字
# words = words[:len(words)] + (' ',)
# # 每个字映射为一个数字ID
# word_num_map = dict(zip(words, range(len(words))))
# print(word_num_map)
# weights=np.random.normal(1,1,100)
# print(weights)
# t = np.cumsum(weights)
# print(t)
# s = np.sum(weights)
# print(s)
# a=np.searchsorted(t, np.random.rand(1)*s)
# import re
# special_character_removal = re.compile(r'[^\w， ]', re.IGNORECASE)
# content='元 邵亨贞	《八归 秋夜咏怀寄钱南金》	清蟾半露，惊乌三匝，城上漏水乍滴。微淳已透潇湘簟，还见小帘摇砌、澹镫垂壁。夜色迢迢人睡去，正想到、山阳吹笛。做弄得、客里文园，病后更无力。还是秋期过了，鸣蛩窗户，又对新诗相忆。片云天外，数峰江上，几误湘灵瑶瑟。叹流光过眼，宋玉多情共今夕。沧浪兴、扁舟容与，醉帽飘萧，亭皋清望极。'
# content = special_character_removal.sub('', content)
# print(content)