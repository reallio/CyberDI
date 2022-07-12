import torch.nn as nn
import torch
import glob
import os
import numpy as np
import cv2
import logging
from torch.utils.data import Dataset 
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as VF
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from label_studio_ml.model import LabelStudioMLBase
from label_studio_converter.brush import decode_rle, mask2rle
import json

d = os.path.dirname(__file__)
os.environ.setdefault('TORCH_HOME', os.path.join(d, '..\pretrained'))
logging.info('TORCH_HOME=' + os.environ.get('TORCH_HOME'))
ws_dir = os.environ.get('LABEL_STUDIO_WORKSPACE')

def get_raw_data(url, result):
    is_local_file = url.startswith('/data/local-files/')
    is_ng = 1.0

    if is_local_file:
        filename = url.replace('/data/local-files/?d=', '')
        filepath = os.path.join(ws_dir, filename)
    else:
        filename = url.replace('/data', 'media')
        filepath = os.path.join(ws_dir, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(filepath)
        
    img  = Image.open(filepath).convert("RGB")
    width = img.width
    height = img.height

    if len(result) > 0:
        annot = result[0] 
        mat = decode_rle(annot['value']['rle']).reshape(height, width, 4)[:,:,0]
        kernel = np.ones((5, 5), np.uint8)
        matD = cv2.dilate(mat, kernel)
        mask = Image.fromarray(matD)               # image2 is a PIL image 
    else:
        mat = np.zeros((height, width), np.uint8)
        mask = Image.fromarray(mat)
        is_ng = 0.0

    is_ng = torch.tensor([is_ng], dtype=torch.float32)
    return {"img":img, "mask":mask, "is_ng":is_ng, "width":width, "height":height}

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("Linear") != -1:
        torch.nn.init.constant_(m.weight.data, 0.0)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)


class SegmentNet(nn.Module):
    def __init__(self, in_channels=3, init_weights=True):
        super(SegmentNet, self).__init__()
        
        self.layer1 = nn.Sequential(
                            nn.Conv2d(in_channels, 32, 5, stride=1, padding=2),
                            nn.BatchNorm2d(32),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(32, 32, 5, stride=1, padding=2),
                            nn.BatchNorm2d(32),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(2)  
                        )

        self.layer2 = nn.Sequential(
                            nn.Conv2d(32, 64, 5, stride=1, padding=2),
                            nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(64, 64, 5, stride=1, padding=2),
                            nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(64, 64, 5, stride=1, padding=2),
                            nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(2)
                        )

        self.layer3 = nn.Sequential(
                            nn.Conv2d(64, 64, 5, stride=1, padding=2),
                            nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(64, 64, 5, stride=1, padding=2),
                            nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(64, 64, 5, stride=1, padding=2),
                            nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(64, 64, 5, stride=1, padding=2),
                            nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(2)
                        )
    
        self.layer4 = nn.Sequential(
                            nn.Conv2d(64, 1024, 15, stride=1, padding=7),
                            nn.BatchNorm2d(1024),
                            nn.ReLU(inplace=True)
                        )

        # self.layer5 = nn.Sequential(
        #                     nn.Conv2d(1024, 1, 1),
        #                     nn.ReLU(inplace=True)
        #                 )

        self.layer5 = nn.Sequential(
                            nn.Conv2d(1024, 1, 1),
                            nn.Sigmoid()
                        )

        # self.layer5 = nn.Sequential(
        #                     nn.Conv2d(1024, 1, 1)
        #                 )

        if init_weights == True:
            pass

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        
        return {"f":x4, "seg":x5}


class DecisionNet(nn.Module):
    
    def __init__(self, init_weights=True):
        super(DecisionNet, self).__init__()

        self.layer1 = nn.Sequential(
                            nn.MaxPool2d(2),
                            nn.Conv2d(1025, 8, 5, stride=1, padding=2),
                            nn.BatchNorm2d(8),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(2),
                            nn.Conv2d(8, 16, 5, stride=1, padding=2),
                            nn.BatchNorm2d(16),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(2),
                            nn.Conv2d(16, 32, 5, stride=1, padding=2),
                            nn.BatchNorm2d(32),
                            nn.ReLU(inplace=True)
                        )

        self.fc =  nn.Sequential(
                            nn.Linear(66, 1, bias=False),
                            nn.Sigmoid()
                        )

        if init_weights == True:
            pass

    def forward(self, f, s):
        xx = torch.cat((f, s), 1)
        x1 = self.layer1(xx)
        x2 = x1.view(x1.size(0), x1.size(1), -1)
        s2 = s.view(s.size(0), s.size(1), -1)

        x_max, x_max_idx = torch.max(x2, dim=2)
        x_avg = torch.mean(x2, dim=2)
        s_max, s_max_idx = torch.max(s2, dim=2)
        s_avg = torch.mean(s2, dim=2)

        y = torch.cat((x_max, x_avg, s_avg, s_max), 1)
        y = y.view(y.size(0), -1)

        return self.fc(y)

class TrainDataset(Dataset):
    def __init__(self, images_urls, annot_results, labels):
        self.images_urls = images_urls
        self.annot_results = annot_results
        self.labels = labels
        self.raw_data = [get_raw_data(url, result) for url, result in zip(self.images_urls, self.annot_results)]
        self.len = len(self.images_urls)

        self.transform = transforms.Compose([
            #transforms.Resize((self.raw_data[0]['height'], self.raw_data[0]['width']), transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((self.raw_data[0]['height'] // 8, self.raw_data[0]['width'] // 8)),
            transforms.ToTensor(),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __getitem__(self, index):
        idx = index %  self.len
        mask = self.raw_data[idx]['mask']
        img = self.raw_data[idx]['img']
        is_ng = self.raw_data[idx]['is_ng']

        if np.random.rand(1) > 0.5:
            mask = VF.hflip(mask)
            img  = VF.hflip(img)
            
        if np.random.rand(1) > 0.5:
            mask = VF.vflip(mask)
            img  = VF.vflip(img)

        img = self.transform(img)
        mask = self.mask_transform(mask)

        return {"img":img, "mask":mask, "is_ng":is_ng}

    def __len__(self):
        return self.len


class Model(object):
    def __init__(self, model_segment_path, model_decision_path, num_classes, freeze_extractor=False):
        self.model_segment_path = model_segment_path
        self.model_decision_path = model_decision_path

        if os.path.exists(self.model_segment_path):
            self.load_segment()
        else:
            self.model_segment = SegmentNet()
        
        self.model_segment = self.model_segment.cuda()

        if os.path.exists(self.model_decision_path):
            self.load_decision()
        else:
            self.model_decision = DecisionNet()

        self.model_decision = self.model_decision.cuda()

        # Optimizer
        self.optimizer_seg = torch.optim.Adam(self.model_segment.parameters(), lr=0.0005, betas=(0.5, 0.999))

        # Loss function
        self.criterion_segment = torch.nn.BCELoss().cuda()

        # Optimizer
        self.optimizer_dec = torch.optim.Adam(self.model_decision.parameters(), lr=0.01, betas=(0.5, 0.999))

        # Loss function
        self.criterion_decision = torch.nn.BCELoss().cuda()


    def save_segment(self):
        torch.save(self.model_segment, self.model_segment_path)
    
    def save_decision(self):
        torch.save(self.model_decision, self.model_decision_path)

    def save(self):
        self.save_segment()
        self.save_decision()

    def load_segment(self):
        self.model_segment = torch.load(self.model_segment_path)

    def load_decision(self):
        self.model_decision = torch.load(self.model_decision_path)

    def predict(self, image_urls, labels):
        raw_data = [get_raw_data(url, []) for url in image_urls]
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        pr_masks = []
        for sample in raw_data:
            raw_img = sample['img']
            width = sample['width']
            height = sample['height']
            img = transform(raw_img)
            img = img.unsqueeze(0).cuda()
            self.model_segment.eval()
            with torch.no_grad():
                seg_result = self.model_segment(img)

            f = seg_result["f"]
            seg = seg_result["seg"]

            self.model_decision.eval()
            with torch.no_grad():
                dec_result = self.model_decision(f, seg)
        
            resize = transforms.Resize((height, width), transforms.InterpolationMode.BICUBIC)
            enlarged_seg = resize(seg.squeeze(0))
            enlarged_seg = enlarged_seg.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            pr_masks.append(enlarged_seg)


        return pr_masks

    def eval(self, image, result_prefix):
        try:
            input_image  = Image.open(image).convert("RGB")
            width = input_image.width
            height = input_image.height

            transform = transforms.Compose([
                transforms.ToTensor()
            ])
            img = transform(input_image).unsqueeze(0).cuda()

            self.model_segment.eval()
            with torch.no_grad():
                seg_result = self.model_segment(img)

            f = seg_result["f"]
            seg = seg_result["seg"]

            self.model_decision.eval()
            with torch.no_grad():
                dec_result = self.model_decision(f, seg)
        
            resize = transforms.Resize((height, width), transforms.InterpolationMode.BICUBIC)
            enlarged_seg = resize(seg.squeeze(0))
            save_image(enlarged_seg, f'{result_prefix}_mask.png')

            return {'rc': 0, 'confidence': dec_result.item()}
        except:
            return {'rc': 1 }

    def train(self, dataloader, num_epochs=5):
        self.train_segment(dataloader, num_epochs)
        self.train_decision(dataloader, num_epochs)
        
    def train_segment(self, dataloader, num_epochs):
        for epoch in range(num_epochs):
            data_iter = dataloader.__iter__()
            self.model_segment.train()

            # train *****************************************************************
            for i in range(len(dataloader)):
                batch_data = data_iter.__next__()
                img = batch_data["img"].cuda()
                mask = batch_data["mask"].cuda()

                self.optimizer_seg.zero_grad()

                rst = self.model_segment(img)
                seg = rst["seg"]

                loss_seg = self.criterion_segment(seg, mask)
                loss_seg.backward()
                self.optimizer_seg.step()

                # logging.info(
                #     "\r [Epoch %d/%d]  [Batch %d/%d] [loss %f]"
                #     %(
                #         epoch,
                #         num_epochs,
                #         i,
                #         len(dataloader),
                #         loss_seg.item()
                #     ))

                print("\r [Epoch %d/%d]  [Batch %d/%d] [loss_seg %f]"
                    %(
                        epoch,
                        num_epochs,
                        i,
                        len(dataloader),
                        loss_seg.item()
                    ))

    def train_decision(self, dataloader, num_epochs):
        for epoch in range(num_epochs):
            data_iter = dataloader.__iter__()
            self.model_decision.train()

            # train *****************************************************************
            for i in range(len(dataloader)):
                batch_data = data_iter.__next__()
                img = batch_data["img"].cuda()
                is_ng = batch_data["is_ng"].cuda()

                self.model_segment.eval()
                with torch.no_grad():
                    rst = self.model_segment(img)

                f = rst["f"]
                seg = rst["seg"]

                self.optimizer_dec.zero_grad()

                rst_d = self.model_decision(f, seg)

                loss_dec = self.criterion_decision(rst_d, is_ng)

                loss_dec.backward()
                self.optimizer_dec.step()
                print("\r [Epoch %d/%d]  [Batch %d/%d] [loss_dec %f]"
                    %(
                        epoch,
                        num_epochs,
                        i,
                        len(dataloader),
                        loss_dec.item()
                    ))


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
            self.model = Model(self.get_model_segment_path(), self.get_model_decision_path(), len(self.labels), freeze_extractor)
        else:
            self.model = Model(self.get_model_segment_path(), self.get_model_decision_path(), len(self.labels), freeze_extractor)

    def get_model_segment_path(self):
        workspace = os.environ.get('LABEL_STUDIO_WORKSPACE')
        project_id = self.project.split('.', 1)[0]
        return os.path.join(workspace, f'model_param/{project_id}_seg.pth')

    def get_model_decision_path(self):
        workspace = os.environ.get('LABEL_STUDIO_WORKSPACE')
        project_id = self.project.split('.', 1)[0]
        return os.path.join(workspace, f'model_param/{project_id}_dec.pth')
    
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
        pr_masks = self.model.predict(image_urls, self.labels)
        
        predictions = []
        for pr_mask in pr_masks:
            result = []
            label = self.labels[0]
            rle = mask2rle(pr_mask.squeeze())
            r = {
                'from_name': self.from_name,
                'to_name': self.to_name,
                'type': 'brushlabels',
                "origin": "manual",
                "image_rotation": 0,
                "original_width": pr_mask.shape[1],
                "original_height": pr_mask.shape[0],
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

        return {}