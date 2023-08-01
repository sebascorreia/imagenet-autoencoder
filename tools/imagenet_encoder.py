import argparse
import torch
from torchvision.transforms import transforms
from PIL import Image
import os
import numpy as np
import pickle
import sys
sys.path.append("./")
# Add necessary imports for the dataset
from datasets import load_dataset
from io import BytesIO
from imagenetautoencoder import utils
from imagenetautoencoder.models import builer as builder


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_args():
    # parse the args
    print('=> parse the args ...')
    parser = argparse.ArgumentParser(description='Encoder for autoencoder')
    parser.add_argument('--dataset', default='teticio/audio-diffusion-256', type=str)
    parser.add_argument('--arch', default='vgg16', type=str, help='backbone architecture')
    parser.add_argument('--resume', type=str)
    args = parser.parse_args()
    args.parallel = 0
    args.batch_size = 1
    args.workers = 0

    return args

def encode(model, img):
    with torch.no_grad():
        code = model.module.encoder(img).cpu().numpy()
    return code

def main(args):
    print('=> torch version : {}'.format(torch.__version__))
    utils.init_seeds(1,cuda_deterministic= False)
    print('=> modeling the network ...')
    model = builder.BuildAutoEncoder(args)
    
    total_params = sum(p.numel() for p in model.parameters())
    print('=> num of params: {} ({}M)'.format(total_params, int(total_params * 4 / (1024*1024))))
    utils.load_dict(args.resume,model)

    # Load the dataset
    dataset = load_dataset(args.dataset , split='train') # Replace with the actual path to your dataset

    # Create the image transform
    trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    encodings = {}
    counter =0
    for data in dataset:
        
        # Extract the melspectrogram image bytes and convert to PIL image
        img = data['image'].convert("RGB")
        img = trans(img).unsqueeze(0).cuda()
        if encodings.get(data['audio_file']) is None:
            encodings[data['audio_file']] = {}
            
        model.eval()
        encodings[data['audio_file']][data['slice']] = encode(model, img)
        # To do: Save or process the encoded code
    pickle.dump(encodings, open("encodings.pkl", "wb"))

if __name__ == '__main__':
    args = get_args()
    main(args)
