import struct
import gzip
import numpy as np

def load_mnist():
    data_path=["train-images-idx3-ubyte.gz","train-labels-idx1-ubyte.gz","t10k-images-idx3-ubyte.gz","t10k-labels-idx1-ubyte.gz"]
    train_x=load_data(data_path[0])
    train_y = load_data(data_path[1])
    test_x = load_data(data_path[2])
    test_y = load_data(data_path[3])
    return train_x,train_y,test_x,test_y

def load_data(path):
    with gzip.open(path, 'rb') as f:
        temp = f.read()
        magic_num = struct.unpack_from('>I',temp,0)
        if magic_num[0]==2051:
            print("Loading image set.")
            _, img_num, row_num, col_num = struct.unpack_from('>IIII', temp, 0)
            images=np.zeros((img_num,row_num*col_num))
            im_index=16
            for i in range(img_num):
                images[i,:] = struct.unpack_from('>784B', temp, im_index)
                im_index += struct.calcsize('>784B')
            return images
        elif magic_num[0]==2049:
            print("Loading labels.")
            _, label_num=struct.unpack_from('>II', temp, 0)
            labels=np.zeros((label_num,1))
            label_index = struct.calcsize('>II')
            for i in range(label_num):
                labels[i,:]=struct.unpack_from('>B', temp, label_index)
                label_index += struct.calcsize('>B')
            return labels
        else:
            print("Unknown file")
            raise