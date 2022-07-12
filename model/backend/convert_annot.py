import os
from camvid_dataset import Dataset, visualize
import json
from label_studio_converter.brush import encode_rle, image2annotation, mask2rle
import numpy as np
import uuid

DATA_DIR = '../data/CamVid/'

x_train_dir = os.path.join(DATA_DIR, 'train')
y_train_dir = os.path.join(DATA_DIR, 'trainannot')

dataset = Dataset(x_train_dir, y_train_dir, classes=['sky', 'building', 'pole', 'road', 'pavement', 
               'tree', 'signsymbol', 'fence', 'car', 
               'pedestrian', 'bicyclist', 'unlabelled'])

for image, mask, filename in dataset:
    height, width, channel = mask.shape

    sky_mask = mask[:,:,0] * 255
    sky_mask = sky_mask.astype(np.uint8)
    sky_rle = mask2rle(sky_mask)

    building_mask = mask[:,:,1] * 255
    building_mask = building_mask.astype(np.uint8)
    building_rle = mask2rle(building_mask)

    pole_mask = mask[:,:,2] * 255
    pole_mask = pole_mask.astype(np.uint8)
    pole_rle = mask2rle(pole_mask)

    road_mask = mask[:,:,3] * 255
    road_mask = road_mask.astype(np.uint8)
    road_rle = mask2rle(road_mask)

    pavement_mask = mask[:,:,4] * 255
    pavement_mask = pavement_mask.astype(np.uint8)
    pavement_rle = mask2rle(pavement_mask)

    tree_mask = mask[:,:,5] * 255
    tree_mask = tree_mask.astype(np.uint8)
    tree_rle = mask2rle(tree_mask)

    signsymbol_mask = mask[:,:,6] * 255
    signsymbol_mask = signsymbol_mask.astype(np.uint8)
    signsymbol_rle = mask2rle(signsymbol_mask)

    fence_mask = mask[:,:,7] * 255
    fence_mask = fence_mask.astype(np.uint8)
    fence_rle = mask2rle(fence_mask)

    car_mask = mask[:,:,8] * 255
    car_mask = car_mask.astype(np.uint8)
    car_rle = mask2rle(car_mask)

    pedestrian_mask = mask[:,:,9] * 255
    pedestrian_mask = pedestrian_mask.astype(np.uint8)
    pedestrian_rle = mask2rle(pedestrian_mask)

    bicyclist_mask = mask[:,:,10] * 255
    bicyclist_mask = bicyclist_mask.astype(np.uint8)
    bicyclist_rle = mask2rle(bicyclist_mask)

    unlabelled_mask = mask[:,:,11] * 255
    unlabelled_mask = unlabelled_mask.astype(np.uint8)
    unlabelled_rle = mask2rle(unlabelled_mask)

    task = {
        'data': {'image': f'/data/local-files/?d=train/{filename}'},
        'annotations': [{
            "result": [{
                "id": str(uuid.uuid4())[0:8],
                "type": "brushlabels",
                "value": {
                    "rle": sky_rle,
                    "format": "rle",
                    "brushlabels": ["sky"]
                },
                "origin": "manual",
                "to_name": 'image',
                "from_name": 'tag',
                "image_rotation": 0,
                "original_width": width,
                "original_height": height
            },{
                "id": str(uuid.uuid4())[0:8],
                "type": "brushlabels",
                "value": {
                    "rle": building_rle,
                    "format": "rle",
                    "brushlabels": ["building"]
                },
                "origin": "manual",
                "to_name": 'image',
                "from_name": 'tag',
                "image_rotation": 0,
                "original_width": width,
                "original_height": height
            },{
                "id": str(uuid.uuid4())[0:8],
                "type": "brushlabels",
                "value": {
                    "rle": pole_rle,
                    "format": "rle",
                    "brushlabels": ["pole"]
                },
                "origin": "manual",
                "to_name": 'image',
                "from_name": 'tag',
                "image_rotation": 0,
                "original_width": width,
                "original_height": height
            },{
                "id": str(uuid.uuid4())[0:8],
                "type": "brushlabels",
                "value": {
                    "rle": road_rle,
                    "format": "rle",
                    "brushlabels": ["road"]
                },
                "origin": "manual",
                "to_name": 'image',
                "from_name": 'tag',
                "image_rotation": 0,
                "original_width": width,
                "original_height": height
            },{
                "id": str(uuid.uuid4())[0:8],
                "type": "brushlabels",
                "value": {
                    "rle": pavement_rle,
                    "format": "rle",
                    "brushlabels": ["pavement"]
                },
                "origin": "manual",
                "to_name": 'image',
                "from_name": 'tag',
                "image_rotation": 0,
                "original_width": width,
                "original_height": height
            },{
                "id": str(uuid.uuid4())[0:8],
                "type": "brushlabels",
                "value": {
                    "rle": tree_rle,
                    "format": "rle",
                    "brushlabels": ["tree"]
                },
                "origin": "manual",
                "to_name": 'image',
                "from_name": 'tag',
                "image_rotation": 0,
                "original_width": width,
                "original_height": height
            },{
                "id": str(uuid.uuid4())[0:8],
                "type": "brushlabels",
                "value": {
                    "rle": signsymbol_rle,
                    "format": "rle",
                    "brushlabels": ["signsymbol"]
                },
                "origin": "manual",
                "to_name": 'image',
                "from_name": 'tag',
                "image_rotation": 0,
                "original_width": width,
                "original_height": height
            },{
                "id": str(uuid.uuid4())[0:8],
                "type": "brushlabels",
                "value": {
                    "rle": fence_rle,
                    "format": "rle",
                    "brushlabels": ["fence"]
                },
                "origin": "manual",
                "to_name": 'image',
                "from_name": 'tag',
                "image_rotation": 0,
                "original_width": width,
                "original_height": height
            },{
                "id": str(uuid.uuid4())[0:8],
                "type": "brushlabels",
                "value": {
                    "rle": car_rle,
                    "format": "rle",
                    "brushlabels": ["car"]
                },
                "origin": "manual",
                "to_name": 'image',
                "from_name": 'tag',
                "image_rotation": 0,
                "original_width": width,
                "original_height": height
            },{
                "id": str(uuid.uuid4())[0:8],
                "type": "brushlabels",
                "value": {
                    "rle": pedestrian_rle,
                    "format": "rle",
                    "brushlabels": ["pedestrian"]
                },
                "origin": "manual",
                "to_name": 'image',
                "from_name": 'tag',
                "image_rotation": 0,
                "original_width": width,
                "original_height": height
            },{
                "id": str(uuid.uuid4())[0:8],
                "type": "brushlabels",
                "value": {
                    "rle": bicyclist_rle,
                    "format": "rle",
                    "brushlabels": ["bicyclist"]
                },
                "origin": "manual",
                "to_name": 'image',
                "from_name": 'tag',
                "image_rotation": 0,
                "original_width": width,
                "original_height": height
            },{
                "id": str(uuid.uuid4())[0:8],
                "type": "brushlabels",
                "value": {
                    "rle": unlabelled_rle,
                    "format": "rle",
                    "brushlabels": ["unlabelled"]
                },
                "origin": "manual",
                "to_name": 'image',
                "from_name": 'tag',
                "image_rotation": 0,
                "original_width": width,
                "original_height": height
            }]
        }]
    }
    json.dump(task, open(filename.replace('.png', '.json'), 'w'))
    print(filename)
# visualize(image=image, mask=mask)

# prepare Label Studio Task


# json.dump(task, open('task.json', 'w'))

