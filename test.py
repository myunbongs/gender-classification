import torch
import matplotlib.pyplot as plt
import argparse

from data import make_celeba_dataloaders
from utils import imshow


def get_parser():
    parser = argparse.ArgumentParser(description="transfer learing assignment")
    parser.add_argument("--path", type=str, default='celeba-gender-dataset/',
                        help="Path to the directory to dataset.")
    parser.add_argument("--pretraiend", type=str, default='checkpoints/model.pt',
                        help="Path to the pretraiend model file.")
    parser.add_argument("--num_images", type=int, default=4,
                        help="Number of test images")
    return parser

class Tester():
    def __init__(self,  model, dataloaders, num_images, device):
        self.model = model
        self.dataloaders = dataloaders
        self.num_images = num_images
        self.device = device
    
    def test(self):
        was_training = self.model.training
        self.model.eval()
        images_so_far = 0
        fig = plt.figure()

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.dataloaders['test']):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

                for j in range(inputs.size()[0]):
                    images_so_far += 1
                    ax = plt.subplot(self.num_images//2, 2, images_so_far)
                    ax.axis('off')
                    ax.set_title(f'predicted: {[preds[j]]}')
                    imshow(inputs.cpu().data[j])

                    if images_so_far == self.num_images:
                        self.model.train(mode=was_training)
                        return
            self.model.train(mode=was_training)

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = get_parser()
    args = parser.parse_args()

    celeba_dataloaders, __ = make_celeba_dataloaders(path=args.path)

    tester = Tester(args.pretraiend, celeba_dataloaders, args.num_images, device)
    tester.test()

if __name__ == '__main__':
    main()

    

        
