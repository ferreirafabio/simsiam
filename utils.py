from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import argparse
import torch

class SummaryWriterCustom(SummaryWriter):
    def __init__(self, out_path, batch_size):
        # super().__init__()
        self.batch_size = batch_size
        self.writer = SummaryWriter(out_path)

    def write_image_grid(self, tag, images, original_images, epoch, global_step):
        fig = image_grid(images=images, original_images=original_images, epoch=epoch, plot_size=self.batch_size)
        self.writer.add_figure(tag, fig, global_step=global_step)
        
    def write_theta_heatmap(self, tag, theta, epoch, global_step):
        fig = theta_heatmap(theta, epoch)
        self.writer.add_figure(tag, fig, global_step=global_step)
    
    def write_scalar(self, tag, scalar_value, global_step):
        self.writer.add_scalar(tag=tag, scalar_value=scalar_value, global_step=global_step)

    def close(self):
        self.writer.close()


class GradientReverse(torch.autograd.Function):
    scale = 1.0

    @staticmethod
    def forward(ctx, x):
        #  autograd checks for changes in tensor to determine if backward should be called
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return GradientReverse.scale * grad_output.neg()

def grad_reverse(x, scale=1.0):
    GradientReverse.scale = scale
    return GradientReverse.apply(x)


def image_grid(images, original_images, epoch, plot_size=16):
    """Return a 5x5 grid of the MNIST images as a matplotlib figure."""
    # Create a figure to contain the plot.
    figure = plt.figure(figsize=(20, 50))
    figure.tight_layout()
    num_images = min(len(original_images), plot_size)
    plt.subplots_adjust(hspace=0.5)

    v1 = images[0]
    v2 = images[1]

    titles = [f"orig@{epoch} epoch", "view 1", "view 2"]
    total = 0
    for i in range(num_images):  # orig_img in enumerate(original_images, 1):
        orig_img = original_images[i]
        v1_img = v1[i]
        v2_img = v2[i]

        all_images = [orig_img, v1_img, v2_img]
        for j in range(3):
            total += 1

            plt.subplot(num_images, 3, total, title=titles[j])
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)

            img = all_images[j].cpu().detach().numpy()

            if img.shape[0] == 3:
                # CIFAR100 and ImageNet case
                img = np.moveaxis(img, 0, -1)
            else:
                # MNIST case
                img = img.squeeze()

            plt.imshow(np.clip(img, 0, 1))

    return figure



def theta_heatmap(theta, epoch):
    figure, ax = plt.subplots()
    # figure.tight_layout()
    sns.heatmap(theta[0], annot=True)
    ax.set_title(f'Theta @ {epoch} epoch')
    return figure


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def custom_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]