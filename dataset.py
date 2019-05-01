import numpy as np
import scipy.misc
import xml.etree.ElementTree as ET
import os

class Dataset(object):
    def __init__(self):
        self._image_ids=[]
        self.image_info=[]
        #self.annotations_info=[]
        self.action_classes=['applauding', 'blowing_bubbles', 'brushing_teeth',
                               'cleaning_the_floor', 'climbing', 'cooking',
                               'cutting_trees', 'cutting_vegetables', 'drinking',
                               'feeding_a_horse', 'fishing', 'fixing_a_bike',
                               'fixing_a_car', 'gardening', 'holding_an_umbrella',
                               'jumping', 'looking_through_a_microscope', 'looking_through_a_telescope',
                               'playing_guitar', 'playing_violin', 'pouring_liquid', 'pushing_a_cart',
                               'reading', 'phoning', 'riding_a_bike', 'riding_a_horse',
                               'rowing_a_boat', 'running', 'shooting_an_arrow', 'smoking',
                               'taking_photos', 'texting_message', 'throwing_frisby', 'using_a_computer',
                               'walking_the_dog', 'washing_dishes', 'watching_TV', 'waving_hands',
                               'writing_on_a_board', 'writing_on_a_book']
        self.action_to_ind=dict(list(zip(self.action_classes, list(range(40)))))

    def add_image(self, image_id, path, **kwargs):
        image_info={
            'id': image_id,
            'path': path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)
    def prepare(self):
        self.num_images=len(self.image_info)
        self._image_ids = np.arange(self.num_images)

    def image_ids(self):
        return self._image_ids
    def source_image_link(self, image_id):
        return self.image_info[image_id]['path']
    def load_image(self, image_id):
        image=scipy.misc.imread(self.image_info[image_id]['path'])
        return image
    def load_bbox(self, image_id):
        image = self.load_image(image_id)
        bbox = self.image_info[image_id]['annotation']['bbox']
        image_bbox = image[(bbox[1]-1):(bbox[3]-1),(bbox[0]-1):(bbox[2]-1)]
        return image_bbox
    def load_label(self, image_id):
        return self.image_info[image_id]['annotation']['label']
    # def add_annotations(self, idx, image_id, bbox, width, height, action):
    #     annotations={
    #         'id':image_id,
    #         'bbox':bbox,
    #         'width':width,
    #         'height':height,
    #         'action':action,
    #         'label':self.action_classes[action],
    #     }
    #     self.annotations_info[idx].append(annotations)
    def load_stan(self, subset):
        frames_path='/usr/local/datasets/stanford40/JPEGImages'
        labels_path='/usr/local/datasets/stanford40/XMLAnnotations'
        train_txt = '/usr/local/datasets/stanford40/ImageSplits/train.txt'
        test_txt = '/usr/local/datasets/stanford40/ImageSplits/test.txt'

        image_list = []
        anno_list = []
        if subset == 'train':
            with open(train_txt) as f:
                for line in f:
                    image_list.append(line[:-1])
                    anno_list.append(line[:-4]+'xml')
        else:
            with open(test_txt) as f:
                for line in f:
                    image_list.append(line[:-1])
                    anno_list.append(line[:-4] + 'xml')

        for i in range(len(image_list)):
            im_id = i+1
            im_path=os.path.join(frames_path, image_list[i])
            action, bbox, size = parser(os.path.join(labels_path, anno_list[i]))
            annotation={
                'id':im_id,
                'bbox':bbox,
                'action':action,
                'label':self.action_to_ind[action],
            }
            self.add_image(image_id=im_id, path=im_path, width=size[1], height=size[0], annotation=annotation)






def parser(file):
    tree = ET.parse(file)
    root = tree.getroot()
    #filename = root[0].text
    action = root[2][1].text
    xmax = int(root[2][2][0].text)
    xmin = int(root[2][2][1].text)
    ymax = int(root[2][2][2].text)
    ymin = int(root[2][2][3].text)
    depth = int(root[3][0].text)
    height = int(root[3][1].text)
    width = int(root[3][2].text)

    bbox = [xmin, ymin, xmax, ymax]
    size = [height, width, depth]

    return action, bbox, size