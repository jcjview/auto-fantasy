
fw=open("peoms_new.txt",'w',encoding='utf-8')
with open('qtrain1.txt',encoding='utf-8') as fr:
    for line in fr:
        line=line.replace('<R>','一').strip()
        lines=line.split("\t")
        poem=""
        for index,l in enumerate(lines):
            poem+=l.replace(' ','')
            if(index%2==1):
                poem+="。"
            else:
                poem+="，"
        fw.write("title:"+poem+"\n")
fw.close()
