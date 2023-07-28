import argparse
import torch
from torchvision.transforms import transforms
from PIL import Image
import os
import numpy as np

# Add necessary imports for the dataset
from datasets import load_dataset
from io import BytesIO

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_args():
    # parse the args
    print('=> parse the args ...')
    parser = argparse.ArgumentParser(description='Encoder for autoencoder')
    parser.add_argument('--dataset', default='teticio/audio-diffusion-256', type=str)
    parser.add_argument('--arch', default='vgg16', type=str, help='backbone architecture')
    parser.add_argument('--resume', type=str)
    args = parser.parse_args()

    return args

def encode(model, img):
    with torch.no_grad():
        code = model.module.encoder(img).cpu().numpy()
    return code

def main(args):
    print('=> torch version : {}'.format(torch.__version__))

    # Rest of your code

    # Load the dataset
    dataset = load_dataset(args.dataset) # Replace with the actual path to your dataset

    # Create the image transform
    trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    for data in dataset:
        # Extract the melspectrogram image bytes and convert to PIL image
        img = data['image'].convert("RGB")
        img = trans(img).unsqueeze(0).cuda()

        model.eval()
        code = encode(model, img)
        print(code.shape)

        # To do: Save or process the encoded code

if __name__ == '__main__':
    args = get_args()
    main(args)
