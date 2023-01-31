from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import argparse
import torch
import os

class SummaryWriterCustom(SummaryWriter):
    def __init__(self, out_path, plot_size):
        # super().__init__()
        self.plot_size = plot_size
        self.writer = SummaryWriter(out_path)

    def write_image_grid(self, tag, stn_images, original_images, epoch):
        fig = image_grid(stn_images=stn_images, original_images=original_images, epoch=epoch, plot_size=self.plot_size)
        self.writer.add_figure(tag, fig, global_step=epoch)
        
    def write_theta_heatmap(self, tag, theta, epoch):
        fig = theta_heatmap(theta, epoch)
        self.writer.add_figure(tag, fig, global_step=epoch)
    
    def write_scalar(self, tag, scalar_value, epoch):
        self.writer.add_scalar(tag=tag, scalar_value=scalar_value, global_step=epoch)

    def write_stn_info(self, stn_images, images, thetas, epoch, tag_images, tag_thetas):
        theta_v1 = thetas[0][0].cpu().detach().numpy()
        theta_v2 = thetas[1][0].cpu().detach().numpy()
        self.write_image_grid(tag=tag_images, stn_images=stn_images, original_images=images, epoch=epoch)
        self.write_theta_heatmap(tag=tag_thetas + " v1", theta=theta_v1, epoch=epoch)
        self.write_theta_heatmap(tag=tag_thetas + " v2", theta=theta_v2, epoch=epoch)
        try: # TODO: analyse "numpy.linalg.LinAlgError: SVD did not converge" error
            theta_eucl_norm = np.linalg.norm(np.double(theta_v1 - theta_v2), 2)
        except Exception as e:
            print(e)
            theta_eucl_norm = 0.
        self.write_scalar(tag="theta eucl. norm.", scalar_value=theta_eucl_norm, epoch=epoch)

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


def image_grid(stn_images, original_images, epoch, plot_size=16):
    """Return a 5x5 grid of the MNIST images as a matplotlib figure."""
    # Create a figure to contain the plot.
    figure = plt.figure(figsize=(20, 50))
    figure.tight_layout()
    num_images = min(len(original_images), plot_size)
    plt.subplots_adjust(hspace=0.5)

    v1 = stn_images[0]
    v2 = stn_images[1]

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
    sns.heatmap(theta, annot=True)
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


def print_gradients(stn, args):
    print("-------------------------sanity check stn grads-------------------------------")
    
    # backbone weights
    if args.separate_localization_net:
        print(stn.module.transform_net.localization_net.backbones[1].conv2d_2.weight)
    else:
        print(stn.module.transform_net.localization_net.backbones[0].conv2d_2.weight)

    # backbone grads
    if args.separate_localization_net:
        print(stn.module.transform_net.localization_net.backbones[1].conv2d_2.weight.grad)
    else:
        print(stn.module.transform_net.localization_net.backbones[0].conv2d_2.weight.grad)

    # head weights
    if args.separate_localization_net:
        print(stn.module.transform_net.localization_net.backbones[1].conv2d_2.weight)
    else:
        print(stn.module.transform_net.localization_net.backbones[0].conv2d_2.weight)

    # head grads
    print(stn.module.transform_net.localization_net.heads[0].linear2.weight.grad)
    print(stn.module.transform_net.localization_net.heads[1].linear2.weight.grad)

    print(f"CUDA MAX MEM:           {torch.cuda.max_memory_allocated()}")
    print(f"CUDA MEM ALLOCATED:     {torch.cuda.memory_allocated()}")