import os, glob
import matplotlib.pyplot as plt


def remove_files(path):
    for f in glob.glob(path):
        os.remove(f)


def generate_images(stack, path):
    remove_files(path + '*')
    n_planes = stack.shape[-1]
    for plane_idx in range(n_planes):
        plt.imshow(stack[:, :, plane_idx], cmap='gray')
        plt.colorbar()
        plt.title('Focal stack {}/{}'.format(plane_idx + 1, n_planes)), plt.xticks([]), plt.yticks([])
        plt.savefig(path + 'img_{}.png'.format(plane_idx))
        plt.close()