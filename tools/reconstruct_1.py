import argparse
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import transforms, ToTensor
import sys
sys.path.append("./")
import utils
import models.builer as builder

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_args():
    # parse the args
    print('=> parse the args ...')
    parser = argparse.ArgumentParser(description='Trainer for auto encoder')
    parser.add_argument('--arch', default='vgg16', type=str, help='backbone architecture')
    parser.add_argument('--resume', type=str)
    parser.add_argument('--image_path', type=str, required=True, help='Path to the image to be reconstructed')

    args = parser.parse_args()
    args.parallel = 0
    args.batch_size = 1
    args.workers = 0

    return args

def main(args):
    print('=> torch version : {}'.format(torch.__version__))
    utils.init_seeds(1, cuda_deterministic=False)

    print('=> modeling the network ...')
    model = builder.BuildAutoEncoder(args)     
    total_params = sum(p.numel() for p in model.parameters())
    print('=> num of params: {} ({}M)'.format(total_params, int(total_params * 4 / (1024*1024))))

    print('=> loading pth from {} ...'.format(args.resume))
    utils.load_dict(args.resume, model)

    # Load the image directly
    input_image = transforms.Compose([ToTensor()])(plt.imread(args.image_path)).unsqueeze(0)

    plt.figure(figsize=(8, 4))

    model.eval()
    print('=> reconstructing ...')
    with torch.no_grad():
      input = input_image.cuda(non_blocking=True)
      output = model(input)

      input = transforms.ToPILImage()(input.squeeze().cpu())
      output_img = transforms.ToPILImage()(output.squeeze().cpu())

    # Save the reconstructed image to the drive
      output_img.save('figs/reconstructed_image.jpg')

      plt.subplot(1,2,1, xticks=[], yticks=[])
      plt.imshow(input)
      plt.title("Original")

      plt.subplot(1,2,2, xticks=[], yticks=[])
      plt.imshow(output_img)
      plt.title("Reconstructed")

  plt.savefig('figs/single_reconstruction.jpg')

if __name__ == '__main__':
    args = get_args()
    main(args)
