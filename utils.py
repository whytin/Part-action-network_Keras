import numpy as np
import random
import scipy.misc
import tensorflow as tf



def load_train(dataset, image_id, augment=True):
    image = dataset.load_image(image_id)
    bbox = dataset.load_bbox(image_id)
    if augment:
        if random.randint(0,1):
            image = np.fliplr(image)
            bbox = np.fliplr(bbox)
        scale=random.randint(140,224)
        image=scale_jitter(image, scale)
        bbox=scale_jitter(bbox, scale)
        image=random_crop(image, 224)
        bbox=random_crop(bbox, 224)
        #image = resize_crop(image, 224)
        #bbox = resize_crop(bbox, 224)
    concat = np.concatenate([image,bbox],axis=0)
    return concat

def load_test(dataset, image_id, augment=True):
    image = dataset.load_image(image_id)
    bbox = dataset.load_bbox(image_id)
    if augment:
        if random.randint(0, 1):
            image = np.fliplr(image)
            bbox = np.fliplr(bbox)
        #scale = random.randint(140, 224)
        image = scale_jitter(image, 224)
        bbox = scale_jitter(bbox, 224)

        image = random_crop(image, 224)
        bbox = random_crop(bbox, 224)

    concat = np.concatenate([image, bbox], axis=0)
    return concat

def scale_jitter(image, scale):
    base_width = np.shape(image)[0]
    base_height = np.shape(image)[1]
    new_width = int(224*base_width/scale)
    new_height = int(224*base_height/scale)
    image_resize = scipy.misc.imresize(image, (new_height,new_width))
    h_off = random.randint(0,new_height-base_height)
    w_off = random.randint(0,new_width-base_width)
    image_jit = image_resize[h_off:h_off+base_height, w_off:w_off+base_width,:]
    return image_jit


def resize_crop(image, size):
    image = mold_image(image)
    crop = scipy.misc.imresize(image, (size,size))
    return crop

def random_crop(image, size):
    image = mold_image(image)
    height = np.shape(image)[0]
    width = np.shape(image)[1]
    if width>=size and height>=size:
        h_off = random.randint(0, height-size)
        w_off = random.randint(0, width-size)
        cropped = image[h_off:h_off+size, w_off:w_off+size, :]
    else:
        cropped = scipy.misc.imresize(image, (size,size))
    return cropped
def mold_image(image):
    return image-[128,128,128]


def load_location(keypoint):
    nose = keypoint[0]
    neck = keypoint[1]
    right_shoulder=keypoint[2]
    right_elbow = keypoint[3]
    right_hand = keypoint[4]
    left_shoulder = keypoint[5]
    left_elbow = keypoint[6]
    left_hand = keypoint[7]
    right_bone = keypoint[8]
    right_knee = keypoint[9]
    right_foot = keypoint[10]
    left_bone = keypoint[11]
    left_knee = keypoint[12]
    left_foot = keypoint[13]
    right_eye = keypoint[14]
    left_eye = keypoint[15]
    right_ear = keypoint[16]
    left_ear = keypoint[17]

    ear_new=neck_new=left_shoulder_new=left_elbow_new=left_hand_new=right_shoulder_new=\
        right_elbow_new=right_hand_new=right_bone_new=left_bone_new=right_foot_new=left_foot_new=[0,0]

    if right_ear[0]!=0 and right_ear[0]<=neck[0]:
        ear_new = right_ear
    elif left_ear[0]!=0 and left_ear[0]<=neck[0]:
        ear_new = left_ear
    elif right_eye[0]!=0 and right_eye[0]<=neck[0]:
        ear_new = right_eye
    elif left_eye[0]!=0 and left_eye[0]<=neck[0]:
        ear_new = left_eye
    else:
        ear_new = nose
    if right_shoulder[0]!=0 and right_shoulder[0]<left_shoulder[0]:
        left_shoulder_new = right_shoulder
        left_elbow_new = right_elbow
        left_hand_new = right_hand
        right_shoulder_new = left_shoulder
        right_elbow_new = left_elbow
        right_hand_new = left_hand
    else:
        right_shoulder_new = right_shoulder
        right_elbow_new = right_elbow
        right_hand_new = right_hand
        left_shoulder_new = left_shoulder
        left_elbow_new = left_elbow
        left_hand_new = left_hand
    if right_bone[0]!=0 and right_bone[0]<left_bone[0]:
        left_bone_new = right_bone
        right_bone_new = left_bone
    else:
        left_bone_new = left_bone
        right_bone_new = right_bone
    if right_foot[0]!=0 and right_foot[0]<left_foot[0]:
        left_foot_new = right_foot
        right_foot_new = left_foot
    elif left_foot[0]!=0 and left_foot[0]<right_foot[0]:
        right_foot_new = right_foot
        left_foot_new = left_foot
    elif right_knee[0]< left_knee[0]:
        left_foot_new = right_knee
        right_foot_new = left_knee
    else:
        left_foot_new = left_knee
        right_foot_new = right_knee
    neck_new=neck

    return [ear_new, neck_new, left_shoulder_new, left_elbow_new, left_hand_new, \
            right_shoulder_new, right_elbow_new, right_hand_new, left_bone_new, right_bone_new, left_foot_new, right_foot_new]

