import os
import sys
import math
import numpy as np
import cv2
import tensorflow as tf
from mrcnn.config import Config
import mrcnn.model as modellib
import glob
import time
import json



home = os.path.expanduser("~")
ftp_path = os.path.join(home, 'projects/freshness')
pattern_json = "*.json"

def where_json(file_name):
    return os.path.exists(file_name)

def init(root_dir):
    # Root directory of the project
    ROOT_DIR = os.path.abspath(root_dir)

    # Import Mask RCNN
    sys.path.append(ROOT_DIR)  # To find local version of the library


def classify_instances(classification_model, image, boxes, masks, class_ids, class_names, labels,
                       scores=None, classify_size=224):
    N = boxes.shape[0]  # Number of instances

    result = dict(zip(labels, [[] for i in range(len(labels))]))
    result_json = dict()
    result_list = []
    # result_json['image_info'] = {'image_height': image.shape[0], 'image_weight': image.shape[1], 'Number_of_Products': N}

    for i in range(N):
        result_json_g = dict()
        # due to merge, just to be sure masks actually have that mask value
        if (i + 1) in masks:
            (y, x) = np.where(masks == (i + 1))  # locations for that object's mask
            bbox = [np.min(x), np.min(y), np.max(x) + 1 - np.min(x), np.max(y) + 1 - np.min(y)]  # x, y, width, height

            fruit = np.ones(image.shape, dtype='uint8') * 255
            fruit[y, x] = image[y, x]  # put object on a white background
            fruit = fruit[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]  # crop only the object

            # if cropped image is larger than classify_size, resize it
            if fruit.shape[0] > classify_size or fruit.shape[1] > classify_size:
                fruit = cv2.resize(fruit, (classify_size, classify_size), interpolation=cv2.INTER_AREA)

            center = classify_size//2
            fruit_half_row = fruit.shape[0]/2
            fruit_half_col = fruit.shape[1]/2
            instance = np.ones((1, classify_size, classify_size, 3))
            instance[0, center - math.floor(fruit_half_row):center + math.ceil(fruit_half_row),
            center - math.floor(fruit_half_col): center + math.ceil(fruit_half_col)] = fruit/255  # object is centered

            predictions = classification_model.predict(instance)  # predict
            top_x = normalize_value(float(bbox[0]), image.shape[1])
            top_y = normalize_value(float(bbox[1]), image.shape[0])
            bottom_x = normalize_value(float(bbox[0]) + float(bbox[2]), image.shape[1])
            bottom_y = normalize_value(float(bbox[1]) + float(bbox[3]), image.shape[0])

            pred_index = int(np.argmax(predictions[0], axis=0))
            fruit_list = result[labels[pred_index]]
            fruit_list.append([predictions[0][pred_index], bbox])
            result_json_g = {"top_y": top_y, "top_x": top_x, "bottom_y": bottom_y, "bottom_x": bottom_x, "probability": float(predictions[0][pred_index]), "tag_name": labels[pred_index]}
            result_list.append(result_json_g)
    result_json["results"] = result_list

    return result, result_json

def normalize_value(value, max):

    min = 0
    value = (value-min)/(max-min)

    return value



def seg_and_classify(seg_model, classification_model, src, seg_class_names, classification_labels,
                     classify_size=224, padding_value=256, stride=175):
    src = np.pad(src, padding_value, constant_values=255)[padding_value:, padding_value:,
          padding_value:padding_value + 3]  # src is assumed to be RGB

    # to store all rois/class_ids/scores/masks and update them if necessary
    rois = np.zeros((1, 4), dtype='int32')
    class_ids = np.zeros((1,), dtype='int32')
    scores = np.zeros((1,), dtype='int32')
    masks = np.zeros((src.shape[0], src.shape[1]), dtype='uint16')
    next_piece = 1

    # shift by stride, but class_ids/scores/rois values might not be %100 correct due to merge
    for i in range(src.shape[0] // stride):
        for j in range(src.shape[1] // stride):
            row = i * stride
            col = j * stride

            # image to segment, a part of the src (of size (padding_value, padding_value, 3))
            detect_image = src[row:row + padding_value, col:col + padding_value, :]
            results = seg_model.detect([detect_image], verbose=1)[0]  # results for the 1st img (only 1 img was fed)

            current_masks = np.zeros((src.shape[0], src.shape[1], results['masks'].shape[-1]), dtype='uint8')
            current_masks[row:row + padding_value, col:col + padding_value, :] = results['masks']

            delete_indices = []
            for k in range(current_masks.shape[-1]):
                x, y = np.where(current_masks[:, :, k] != 0)
                center = len(x) // 2
                piece_value = next_piece

                # if there is another object on the center of the current mask, then merge them by
                # taking that 'another object's mask number (integer)
                if masks[x[center], y[center]] != 0:
                    piece_value = masks[x[center], y[center]]
                masks[x, y] = piece_value

                # if above condition didn't hold, increment next_piece since there is another new object
                if piece_value == next_piece:
                    next_piece += 1
                else:
                    delete_indices.append(k)  # to delete this from other information (roi etc.)

            current_rois = np.copy(results['rois'])
            # fix locations considering to whole image
            # (current_rois had locations in between 0 - (padding_value - 1) for rows and cols)
            current_rois[:, [0, 2]] += row
            current_rois[:, [1, 3]] += col
            current_rois = np.delete(current_rois, delete_indices, axis=0)  # delete the merged ones

            rois = np.concatenate((rois, current_rois), axis=0)
            class_ids = np.concatenate((class_ids, np.delete(results['class_ids'], delete_indices)), axis=0)
            scores = np.concatenate((scores, np.delete(results['scores'], delete_indices)), axis=0)

    # classify using created masks
    return classify_instances(classification_model, src, rois[1:], masks, class_ids[1:],
                       seg_class_names, classification_labels, scores[1:], classify_size=classify_size)


def visualize_result(image, result, labels):
    freshness_txt = open("Freshness_process.txt", "w")
    colors = []
    for _ in labels:
        colors.append((int(np.random.choice(256, 1)), int(np.random.choice(256, 1)), int(np.random.choice(256, 1))))

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for key, value in result.items():
        key = classification_labels.index(key)
        '''
        if key == 2 or key == 3:
            for score, bbox in value:
                cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), colors[key], 1)
                freshness_txt.write(f'{bbox}\t{score}\t{classification_labels[key]}\n')
        '''
        for score, bbox in value:
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), colors[key], 1)
            freshness_txt.write(f'{bbox}\t{score}\t{classification_labels[key]}\n')
    freshness_txt.close()
    cv2.imshow("result", image)
    cv2.waitKey(0)


def create_models(model_path, classification_model_path):
    inference_config = InferenceConfig()
    seg_model = modellib.MaskRCNN(mode="inference",
                                  config=inference_config,
                                  model_dir="")
    seg_model.load_weights(model_path, by_name=True)

    classification_model = tf.keras.models.load_model(classification_model_path)
    return seg_model, classification_model


# this is the same as the one for training
class FruitsConfig(Config):
    NAME = "fruits"
    DETECTION_MIN_CONFIDENCE = 0.7
    GPU_COUNT = 1
    IMAGES_PER_GPU = 5
    NUM_CLASSES = 1 + 2  # BG + apple + tomato
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    TRAIN_ROIS_PER_IMAGE = 32
    VALIDATION_STEPS = 5


class InferenceConfig(FruitsConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


root_dir = "D:/Motiwe/CaurfurDubai/Mask_RCNN/"
model_path = "models/mask_rcnn_fruits.h5"
classification_model_path = "models/freshness.h5"
classification_labels = ["freshapples", "freshtomatoes", "rottenapples", "rottentomatoes"]
seg_class_names = ['BG', 'apple', 'tomato']

init(root_dir)  # to find local version of the mask-rcnn library
seg_model, classification_model = create_models(model_path, classification_model_path)

# load image
images_path = "D:/Motiwe/CaurfurDubai/images/test/"
image_path = os.listdir(images_path)[0]
image = cv2.imread(os.path.join(images_path, image_path))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)







result, result_json = seg_and_classify(seg_model, classification_model, image, seg_class_names, classification_labels)
visualize_result(image, result, classification_labels)


for directory, _, _ in os.walk(ftp_path):
    with open(directory + '\sonuc.json', 'w') as outfile:
        json.dump(result_json, outfile)

