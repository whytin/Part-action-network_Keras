
# coding: utf-8

# In[2]:


import numpy as np
import skimage.io
import os
import scipy.io as sio

class Dataset(object):
    def __init__(self, class_map=None):
        self._image_ids=[]
        self.image_info=[]
        self.annotations_info=[]
        for i in range(2326):
            self.annotations_info.append([])
        #Background is always the first class
        self.class_info=[{"source":"", "id":0, "name":"BG"}]
        self.source_class_ids={}
        self.action_classes=['baseball_pitch', 'baseball_swing', 'bench_press',
                'bowl', 'clean_and_jerk', 'golf_swing', 
                'jumping_jacks', 'jump_rope', 'pullup',
                'pushup', 'situp', 'squat',
                'strum_guitar', 'tennis_forehand', 'tennis_serve']
#        onehot_labels=list()
#        for i in range(15):
#            letter=[0 for _ in range(15)]
#            letter[i]=1
#            onehot_labels.append(letter)
        self.action_to_ind=dict(list(zip(self.action_classes, list(range(15)))))
    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source cannot contain a dot"
        for info in self.class_info:
            if info['source'] == source and info['id'] == class_id:
                return
        self.class_info.append({
            'source': source,
            'id': class_id,
            'name': class_name,
        })
    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            'id': image_id,
            'source': source,
            'path': path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)
    def image_reference(self, image_id):
        return ''
    def prepare(self, class_map=None):
        def clean_name(name):
            return ','.join(name.split(',')[:1])
        self.num_classes=len(self.class_info)
        self.class_ids=np.arange(self.num_classes)
        self.class_names=[clean_name(c['name']) for c in self.class_info]
        self.num_images=len(self.image_info)
        self._image_ids=np.arange(self.num_images)
        self.class_from_source_map={'{}.{}'.format(info['source'], info['id']):id
                                   for info, id in zip(self.class_info, self.class_ids)}
        self.sources = list(set([i['source'] for i in self.class_info]))
        self.source_class_ids={}
        for source in self.sources:
            self.source_class_ids[source]=[]
            for i, info in enumerate(self.class_info):
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)
    
    def map_source_class_id(self, source_class_id):
        return self.class_from_source_map[source_class_id]
    def get_source_class_id(self, class_id, source):
        info = self.class_info[class_id]
        assert info['source'] == source
        return info['id']
    def append_data(self, class_info, image_info):
        self.external_to_class_id={}
        for i, c in enumerate(self.class_info):
            for ds, id in c['map']:
                self.external_to_class_id[ds + str(id)]=i
        self.external_to_image_id={}
        for i, info in enumerate(self.image_info):
            self.external_to_image_id[info['ds']+str(info['id'])]=i
    

    @property
    def image_ids(self):
        return self._image_ids
    
    def source_image_link(self, image_id):
        return self.image_info[image_id]['path']
                
    def load_image(self, image_id):
        image = skimage.io.imread(self.image_info[image_id]['path'])
        
        if image.ndim !=3:
            image=skimage.color.gray2rgb(image)
        return image
    
    def load_mask(self, image_id):
        mask = np.empty([0,0,0])
        class_ids = np.empty([0], np.int32)
        return mask, class_ids
        
    def load_keypoints(self, image_id):
        keypoints = np.empty([0,0])
        mask = np.empty([0,0,0])
        class_ids = np.empty([0], np.int32)
        return keypoints, mask, class_ids
    
class PennAction(Dataset):
    def add_annotations(self, idx, image_id, bbox, width, height, x, y, visibility, action):
        annotations={
            'id':image_id,
            'bbox':bbox,
            'width':width,
            'height':height,
            'x':x,
            'y':y,
            'visibility':visibility,
            'action':action,
            'labels':self.action_to_ind[action[0]],
        }
        self.annotations_info[idx].append(annotations)
    def load_penn(self, subset):
        frames_path='/usr/local/datasets/Penn_Action/frames'
        labels_path='/usr/local/datasets/Penn_Action/labels'
        
        images_list=os.listdir(frames_path)
        images_list.sort(key=lambda x:int(x))
        num_images_dir=len(images_list)
        Train_ratio=0.9355
        division=int(num_images_dir*Train_ratio)
        
        if subset == 'train':
            images_list_path=[os.path.join(frames_path, x) for x in images_list[:division]]
        else:
            images_list_path=[os.path.join(frames_path, x) for x in images_list[division:]]
        
        labels_list=os.listdir(labels_path)
        labels_list.sort(key=lambda x:int(x[:-4]))
        
        if subset == 'train':
            labels_list_path=[os.path.join(labels_path, x) for x in labels_list[:division]]
        else:
            labels_list_path=[os.path.join(labels_path, x) for x in labels_list[division:]]
                
        class_ids = list(range(1,2))
        for i in class_ids:
            self.add_class('Penn', i, 'person')
        
        
        labels_data = [sio.loadmat(x) for x in labels_list_path] 
        if subset == 'train':
            image_idx=0
        else:
            image_idx=151664
        for i in range(len(labels_list_path)):
            for j in range(labels_data[i]['nframes'][0][0]):
                im_id = '%06d'%(j+1)
                im_path=os.path.join(images_list_path[i], im_id+'.jpg')
                image_idx += 1
                image_id = image_idx
                print(image_id)
                self.add_annotations(i, image_id, labels_data[i]['bbox'][j], labels_data[i]['dimensions'][0][1], labels_data[i]['dimensions'][0][0],
                               labels_data[i]['x'][j], labels_data[i]['y'][j], labels_data[i]['visibility'][j], labels_data[i]['action'])
                annotations=self.annotations_info[i]
                self.add_image('Penn', image_id=image_id, path=im_path, width=annotations[j]['width'], height=annotations[j]['height'], annotations=annotations[j])
                
    def load_keypoints(self, image_id):
        image_info = self.image_info[image_id]
        instance_masks=[]
        keypoint = []
        class_ids = []
        instance_masks=[]
        annotations = self.image_info[image_id]['annotations']
        human_nums = 1
        m = np.zeros([annotations['height'], annotations['width'], 13])
#        class_mask = np.zeros([human_nums, 13])
        keypoints_zip=zip(annotations['x'], annotations['y'], annotations['visibility'])
        keypoints = np.reshape(list(keypoints_zip), (13,3))
        for part_num, bp in enumerate(keypoints.astype(int)):
            if bp[2] == 1:
                m[bp[1], bp[0], part_num] = 1
#            class_mask[human_num, part_num]=bp[2]
        class_ids.append(1)
        instance_masks.append(m)
        keypoint.append(keypoints)
        if class_ids:
            mask = m
            class_ids = np.array(class_ids, dtype=np.int32)
            keypoint = np.array(keypoint, dtype=np.int32)
        return keypoint
        
        
    def load_labels(self, image_id):
        return self.image_info[image_id]['annotations']['labels']

    def load_bbox(self, image_id):
        return self.image_info[image_id]['annotations']['bbox']
        
        
        
        
        
        
        
        

