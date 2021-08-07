import pandas as pd
import numpy as np
import os
import math

anno_source = os.path.join('D:','meronym','datasets','PASCAL-VOC','xybb-objects-new')
labels = os.listdir(anno_source)
data = []
for object_type in  labels:
    image_list = os.listdir(os.path.join(anno_source,object_type,'bbox'))
    ob_count = len(image_list)
    if ob_count>0:
        image_list = [name.split('.')[0] for name in image_list]
        label_list = [object_type]*ob_count
        obj_frame = pd.DataFrame(list(zip(label_list,image_list)), columns = ['label','image'])
        obj_frame.reindex(np.random.permutation(obj_frame.index))
        obj_frame['set'] = 'train'
        obj_frame.loc[math.ceil(ob_count*0.8):,'set'] = 'test'
        obj_frame.loc[math.ceil(ob_count*0.9):,'set'] = 'val'

        data.append(obj_frame)

data = pd.concat(data)
data.to_csv('dataset.csv')
    
    
                             
    
    
