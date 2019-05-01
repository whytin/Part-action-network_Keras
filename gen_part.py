import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image, ImageDraw
import Penn

def gen_part_bbox(location, height, width):
    left_ear=location[0]
    neck = location[1]
    left_shoulder=location[2]
    left_elbow = location[3]
    left_finger = location[4]
    right_shoulder = location[5]
    right_elbow = location[6]
    right_finger = location[7]
    left_bone = location[8]
    right_bone = location[9]
    left_foot = location[10]
    right_foot = location[11]
    head=expand50([left_ear[0],2*left_ear[1]-neck[1], 2*neck[0]-left_ear[0], neck[1]], height, width)
    left_arm = expand50([min(left_shoulder[0],left_elbow[0],left_finger[0]),min(left_shoulder[1],left_elbow[1],left_finger[1]),\
                         max(left_shoulder[0],left_elbow[0],left_finger[0]),max(left_shoulder[1],left_elbow[1],left_finger[1])], height, width)
    left_hand= expand50([min(1.5*left_finger[0]-0.5*left_elbow[0], left_finger[0]), min(1.5*left_finger[1]-0.5*left_elbow[1], left_finger[1]), \
                        max(1.5*left_finger[0]-0.5*left_elbow[0], 0.75*left_finger[0]+0.25*left_elbow[0]), max(1.5*left_finger[1]-0.5*left_elbow[1], 0.75*left_finger[1]+0.25*left_elbow[1])], height, width)
    right_arm= expand50([min(right_shoulder[0], right_elbow[0], right_finger[0]), min(right_shoulder[1], right_elbow[1], right_finger[1]), \
         max(right_shoulder[0], right_elbow[0], right_finger[0]), max(right_shoulder[1], right_elbow[1], right_finger[1])], height, width)
    right_hand = expand50([min(1.5 * right_finger[0] - 0.5 * right_elbow[0], right_finger[0]),
                         min(1.5 * right_finger[1] - 0.5 * right_elbow[1], right_finger[1]), \
                         max(1.5 * right_finger[0] - 0.5 * right_elbow[0], 0.75*right_finger[0]+0.25*right_elbow[0]),
                         max(1.5 * right_finger[1] - 0.5 * right_elbow[1], 0.75*right_finger[1]+0.25*right_elbow[1])], height, width)
    terso = expand50([min(left_shoulder[0],left_bone[0]), min(left_shoulder[1],right_shoulder[1]), max(right_shoulder[0], right_bone[0]), max(left_bone[1], right_bone[1])], height, width)
    lower_body = expand50([min(left_bone[0],left_foot[0]), min(left_bone[1],right_bone[1]), max(right_bone[0], right_foot[0]), max(left_foot[1], right_foot[1])], height, width)

    return head, left_arm, left_hand, right_arm, right_hand, terso, lower_body
def expand50(bbox, height, width):
    min_x = max(1.3*bbox[0]-0.3*bbox[2],0)
    min_y = max(1.3*bbox[1]-0.3*bbox[3],0)
    max_x = min(1.3 * bbox[2] - 0.3 * bbox[0], width)
    max_y = min(1.3 * bbox[3] - 0.3 * bbox[1], height)

    return [min_x, min_y, max_x, max_y]

# file_path='/usr/local/datasets/stanford40/applauding_001.jpg'
# location =[[100,119], [132,159], [57,189], [52,305],[120,218],[200,166],[236,278],[177,199],[88,327],[202,322],[103,388],[194,386]]
#
# img = Image.open(file_path)
# w, h = img.size
# draw = ImageDraw.Draw(img)
# head, left_arm, left_hand, right_arm, right_hand, terso, lower_body = gen_part_bbox(location,400,300)
#


def load(id):
    penn_dataset = Penn.PennAction()
    penn_dataset.load_penn('train')
    penn_dataset.prepare()
    image = penn_dataset.load_image(id)
    keypoint = penn_dataset.load_keypoints(id)
    label = penn_dataset.load_labels(id)
    bbox = penn_dataset.load_bbox(id)

    return image, bbox, keypoint, label

image, bbox, keypoint, label=load(1864)
x=keypoint[0,:,0]
y=keypoint[0,:,1]
neck=[0.5*(x[1]+x[2]), 0.5*(y[1]+y[2])]
ear =[0.5*(neck[0]+x[1]), y[0]]
location=[ear, neck , [x[1],y[1]], [x[3], y[3]], [x[5], y[5]], [x[2],y[2]], [x[4],y[4]], [x[6],y[6]], [x[7], y[7]], [x[8],y[8]], [x[11],y[11]],[x[12], y[12]]]
shape=np.shape(image)
head, left_arm, left_hand, right_arm, right_hand, terso, lower_body= gen_part_bbox(location, shape[0], shape[1])

plt.imshow(image)


currentAxis=plt.gca()
rect0=patches.Rectangle((head[0], head[1]),head[2]-head[0],head[3]-head[1],linewidth=1,edgecolor='r',facecolor='none')
currentAxis.add_patch(rect0)
rect1=patches.Rectangle((left_arm[0], left_arm[1]),left_arm[2]-left_arm[0],left_arm[3]-left_arm[1],linewidth=1,edgecolor='r',facecolor='none')
currentAxis.add_patch(rect1)
rect2=patches.Rectangle((left_hand[0], left_hand[1]),left_hand[2]-left_hand[0],left_hand[3]-left_hand[1],linewidth=1,edgecolor='r',facecolor='none')
currentAxis.add_patch(rect2)
rect3=patches.Rectangle((right_arm[0], right_arm[1]),right_arm[2]-right_arm[0],right_arm[3]-right_arm[1],linewidth=1,edgecolor='r',facecolor='none')
currentAxis.add_patch(rect3)
rect4=patches.Rectangle((right_hand[0], right_hand[1]),right_hand[2]-right_hand[0],right_hand[3]-right_hand[1],linewidth=1,edgecolor='r',facecolor='none')
currentAxis.add_patch(rect4)
rect5=patches.Rectangle((terso[0], terso[1]),terso[2]-terso[0],terso[3]-terso[1],linewidth=1,edgecolor='r',facecolor='none')
currentAxis.add_patch(rect5)
rect6=patches.Rectangle((lower_body[0], lower_body[1]),lower_body[2]-lower_body[0],lower_body[3]-lower_body[1],linewidth=1,edgecolor='r',facecolor='none')
currentAxis.add_patch(rect6)
for i in range(13):
    currentAxis.scatter(x[i], y[i],c='r', s=20, alpha=1)
plt.show()