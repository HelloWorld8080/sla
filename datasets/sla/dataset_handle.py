import xml.etree.ElementTree as ET
import os
from os import getcwd
import shutil

sets = ['train', 'val', 'test']
classes = ["car", "slagcar"]  # 改成自己的类别
abs_path = os.getcwd()
abs_path = abs_path if "project" in os.getcwd() else "F:/deeplearn/object_check/yolo_v5/sla/datasets/sla"
or_imgs_path = "/home/data" if "project" in os.getcwd() else "F:/deeplearn/object_check/yolo_v5/sla/data/or_imgs"
print('fdf', abs_path, or_imgs_path)


def move_img(ori_img_path):
    for i, j, k in os.walk(ori_img_path):
        for k_i in k:
            full_file = os.path.join(i, k_i)
            new_full_file = os.path.join(abs_path + '/image_sets', k_i)
            shutil.copy(full_file, new_full_file)


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def convert_annotation(image_id):
    in_file = open(abs_path + '/image_sets/%s' % (image_id), encoding='UTF-8')
    out_file = open(abs_path + '/labels/%s.txt' % (image_id.split('.')[0]), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        # difficult = obj.find('difficult').text
        # difficult = obj.find('Difficult').text
        cls = obj.find('name').text
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        b1, b2, b3, b4 = b
        # 标注越界修正
        if b2 > w:
            b2 = w
        if b4 > h:
            b4 = h
        b = (b1, b2, b3, b4)
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


# print(len(os.listdir('./image_sets')))

import shutil

#/home/data
move_img(or_imgs_path)
xml_count = 0
img_count = 0

train_file = open(abs_path + '/train.txt', 'w')
val_file = open(abs_path + '/val.txt', 'w')
test_file = open(abs_path + '/test.txt', 'w')
for file in os.listdir(abs_path + '/image_sets'):
    if file.split('.')[1] == 'xml':
        full_file = os.path.join(abs_path + '/image_sets', file)
        new_full_file = os.path.join(abs_path + '/Annotations', file)
        shutil.copy(full_file, new_full_file)
        convert_annotation(file)
        # if xml_count < 100:
        #     pass
        #     # convert_annotation(file)
        # elif xml_count < 150:
        #     pass
        #     # convert_annotation(file)
        xml_count += 1
    elif file.split('.')[1] == 'jpg':
        full_file = os.path.join(abs_path + '/image_sets', file)
        new_full_file = os.path.join(abs_path + '/images', file)
        shutil.copy(full_file, new_full_file)
        if img_count < 100:
            train_file.write(abs_path + '/images/%s\n' % (file))
        elif img_count < 150:
            val_file.write(abs_path + '/images/%s\n' % (file))
        else:
            test_file.write(abs_path + '/images/%s\n' % (file))
        img_count += 1
train_file.close()
val_file.close()
test_file.close()