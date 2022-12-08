import json
import logging
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import requests
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))
JSON_CONTENT_TYPE = 'application/json'
JPEG_CONTENT_TYPE = 'image/jpeg'


def Net():
    model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False   

    model.fc = nn.Sequential(
                      nn.Linear(num_features, 256),
                      nn.ReLU(),
                      nn.Dropout(0.4),
                      nn.Linear(256, 133),                   
                      nn.LogSoftmax(dim=1))
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        checkpoint = torch.load(f , map_location =device)
        model.load_state_dict(checkpoint)
    model.eval()
    return model




def input_fn(request_body, content_type=JPEG_CONTENT_TYPE):
    
    if content_type == JPEG_CONTENT_TYPE: return Image.open(io.BytesIO(request_body))
    logger.debug('SO loded JPEG content')
    # process a URL submitted to the endpoint
    
    if content_type == JSON_CONTENT_TYPE:
        #img_request = requests.get(url)
        logger.debug(f'Request body is: {request_body}')
        request = json.loads(request_body)
        logger.debug(f'Loaded JSON object: {request}')
        url = request['url']
        img_content = requests.get(url).content
        return Image.open(io.BytesIO(img_content))
    
    raise Exception('Requested unsupported ContentType in content_type: {}'.format(content_type))

# inference
def predict_fn(input_object, model):
    logger.info('In predict fn')
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
    ])
    logger.info("transforming input")
    input_object=test_transform(input_object)
    
    with torch.no_grad():
        logger.info("Calling model")
        prediction = model(input_object.unsqueeze(0))
    return prediction
