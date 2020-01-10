import numpy as np
from random import randint
def bin_data_gen(seq_len=(5,20),list_len=20,file_path='output'):
    lol=[]
    with open(file_path,'w') as fn:
        for index in range(list_len):
            lol.append([(1 if randint(0,9) else 0) for i in range(randint(seq_len[0],seq_len[1]))])
            fn.write(''.join(str(x) for x in lol[index])+ ('\n' if index!=(list_len-1) else ''))

bin_data_gen(seq_len=(10,30),list_len=30,file_path='output')