import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import numpy as np
import requests
import io
import hashlib
import urllib
import cv2
import json
import logging

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import albumentations as albu

import segmentation_models_pytorch as smp

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_single_tag_keys, get_choice, is_skipped
from label_studio_converter.brush import decode_rle, mask2rle

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
d = os.path.dirname(__file__)
os.environ.setdefault('TORCH_HOME', os.path.join(d, '..\pretrained'))
logging.info('TORCH_HOME=' + os.environ.get('TORCH_HOME'))
ws_dir = os.environ.get('LABEL_STUDIO_WORKSPACE')

ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation
DEVICE = 'cuda'

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    # _transform = [
    #     albu.PadIfNeeded(384, 480),
    #     albu.Lambda(image=preprocessing_fn),
    #     albu.Lambda(image=to_tensor, mask=to_tensor),
    # ]
    _transform = [
        albu.PadIfNeeded(192, 640),
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

def get_postprocessing(width, height):
    _transform = [albu.CenterCrop(height, width)]
    return albu.Compose(_transform)

image_transforms = get_preprocessing(preprocessing_fn)

def get_transformed_image(url, result, labels):
    is_local_file = url.startswith('/data/local-files/')

    if is_local_file:
        filename = url.replace('/data/local-files/?d=', '')
        filepath = os.path.join(ws_dir, filename)
    else:
        filename = url.replace('/data', 'media')
        filepath = os.path.join(ws_dir, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(filepath)
    else:
        image = cv2.imread(filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape
        if result is None:
            return image_transforms(image=image), width, height
        else:
            mask_dict = {annot['value']['brushlabels'][0]: decode_rle(annot['value']['rle']).reshape(height, width, 4)[:,:,0] / 255 for annot in result}
            masks = []
            for label in labels:
                if label in mask_dict:
                    masks.append(mask_dict[label])
                else:
                    masks.append(np.zeros((height, width)))
            mask = np.stack(masks, axis=-1).astype('float')
            return image_transforms(image=image, mask=mask), width, height


class TrainDataset(Dataset):

    def __init__(
            self, 
            images_urls, 
            annot_results, 
            labels
    ):
        self.images_urls = images_urls
        self.annot_results = annot_results
        self.labels = labels
        self.samples = [get_transformed_image(url, result, self.labels) for url, result in zip(self.images_urls, self.annot_results)]
    
    def __getitem__(self, i):
        sample, *_ = self.samples[i]
        return sample['image'], sample['mask']
        
    def __len__(self):
        return len(self.images_urls)


class Model(object):
    def __init__(self, model_path, num_classes, freeze_extractor=False):
        self.model_path = model_path

        if os.path.exists(self.model_path):
            self.model = torch.load(self.model_path)
        else:
            # create segmentation model with pretrained encoder
            self.model = smp.FPN(
                encoder_name=ENCODER, 
                encoder_weights=ENCODER_WEIGHTS, 
                classes=num_classes, 
                activation=ACTIVATION,
            )

    def save(self):
        torch.save(self.model, self.model_path)

    def load(self):
        self.model = torch.load(self.model_path)

    def predict(self, image_urls, annot_results, labels):
        samples = [get_transformed_image(url, result, labels) for url, result in zip(image_urls, annot_results)]
        pr_masks = []
        for sample, width, height in samples:
            x_tensor = torch.from_numpy(sample['image']).to(DEVICE).unsqueeze(0)
            pr_mask = self.model.predict(x_tensor)
            pr_mask = pr_mask.squeeze(dim=0).cpu().numpy().round() * 255
            pr_mask = pr_mask.astype(np.uint8)
            image_postforms = get_postprocessing(width, height) 
            croped_mask = [image_postforms(image=mask)['image'] for mask in pr_mask]
            pr_masks.append(np.stack(croped_mask))   
        return pr_masks

    def eval(self, image, result_prefix):
        try:
            input_image = cv2.imread(image)
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            height, width, _ = input_image.shape
            sample = image_transforms(image=input_image)
            input_tensor = torch.from_numpy(sample['image']).to(DEVICE).unsqueeze(0)
            pr_mask = self.model.predict(input_tensor)
            pr_mask = pr_mask.squeeze(dim=0).cpu().numpy().round() * 255
            pr_mask = pr_mask.astype(np.uint8)
            image_postforms = get_postprocessing(width, height) 

            for i, mask in enumerate(pr_mask):
                croped_mask = image_postforms(image=mask)['image']
                cv2.imwrite(f'{result_prefix}_{i}.png', croped_mask)

            return {'rc': 0}
        except:
            return {'rc': 1}

    def train(self, dataloader, num_epochs=5):
        loss = smp.utils.losses.DiceLoss()
        metrics = [
            smp.utils.metrics.IoU(threshold=0.5),
        ]

        optimizer = torch.optim.Adam([ 
            dict(params=self.model.parameters(), lr=0.0001),
        ])

        # create epoch runners 
        # it is a simple loop of iterating over dataloader`s samples
        train_epoch = smp.utils.train.TrainEpoch(
            self.model, 
            loss=loss, 
            metrics=metrics, 
            optimizer=optimizer,
            device=DEVICE,
            verbose=True,
        )

        for i in range(num_epochs):
            logging.info('\nEpoch: {}'.format(i))
            train_epoch.run(dataloader)

class ModelBackend(LabelStudioMLBase):

    def __init__(self, freeze_extractor=False, **kwargs):
        super(ModelBackend, self).__init__(**kwargs)
        if len(self.parsed_label_config.items()) == 0:
            self.model = None
            return
        
        self.save_label_config()
        from_name, schema = list(self.parsed_label_config.items())[0]
        self.from_name = from_name
        self.to_name = schema['to_name'][0]
        self.labels = schema['labels']
        self.freeze_extractor = freeze_extractor
        if self.train_output:
            self.model = Model(self.get_model_path(), len(self.labels), freeze_extractor)
        else:
            self.model = Model(self.get_model_path(), len(self.labels), freeze_extractor)

    def get_model_path(self):
        workspace = os.environ.get('LABEL_STUDIO_WORKSPACE')
        project_id = self.project.split('.', 1)[0]
        return os.path.join(workspace, f'model_param/{project_id}.pth')
    
    def save_label_config(self):
        workspace = os.environ.get('LABEL_STUDIO_WORKSPACE')
        project_id = self.project.split('.', 1)[0]
        filepath = os.path.join(workspace, f'model_param/{project_id}.json')
        label_config = {
            'project': self.project,
            'hostname': self.hostname,
            'access_token': self.access_token,
            'label_config': self.label_config
        }
        with open(filepath, mode='w', encoding='utf-8') as file_obj:
            json.dump(label_config, file_obj)

    def reset_model(self):
        self.model = Model(len(self.labels), self.freeze_extractor)

    def eval(self, image, result_prefix):
        return self.model.eval(image, result_prefix)

    def predict(self, tasks, **kwargs):
        if self.model is None: return

        image_urls = [task['data']['image'] for task in tasks]
        try:
            annot_results = [task['annotations'][0]['result'] for task in tasks]
        except:
            annot_results = [None] * len(image_urls)

        pr_masks = self.model.predict(image_urls, annot_results, self.labels)
        
        predictions = []
        for pr_mask in pr_masks:
            result = []

            for label, mask in zip(self.labels, pr_mask):
                rle = mask2rle(mask)
                r = {
                    'from_name': self.from_name,
                    'to_name': self.to_name,
                    'type': 'brushlabels',
                    "origin": "manual",
                    "image_rotation": 0,
                    "original_width": mask.shape[1],
                    "original_height": mask.shape[0],
                    'value': {
                        'format': 'rle',
                        'brushlabels': [label],
                        'rle': rle
                    }
                }
                result.append(r)

            # expand predictions with their scores for all tasks
            predictions.append({'result': result})

        return predictions

    def fit(self, completions, workdir=None, batch_size=8, num_epochs=100, **kwargs):
        if self.model is None: return

        image_urls, annot_results = [], []
        logging.info('Collecting annotations...')
        if len(completions) == 0:
            return {}
        
        for completion in completions:
            if is_skipped(completion):
                continue
            image_urls.append(completion['data']['image'])
            annot_results.append(completion['annotations'][0]['result'])

        logging.info('Creating dataset...')
        dataset = TrainDataset(image_urls, annot_results, self.labels)
        dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

        logging.info('Train model...')
        # self.reset_model()
        self.model.train(dataloader, num_epochs=num_epochs)

        logging.info('Save model...')
        self.model.save()

        return {'model_path': self.get_model_path()}
