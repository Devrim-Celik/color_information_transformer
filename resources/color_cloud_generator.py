from scipy.ndimage import imread
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_color_cloud(path, size=(352, 240), show=True,
    gen_img=True, path_to_save="../images"):
    """
    Given an image, it maps all its pixels onto a three-dimensional space
    using its (R,G,B) values and saves this color cloud if wanted.

    Args
        path: path to an image
        size: size to resize the image to
        show: whether to display the color cloud image via matplotlib
        gen_imag: whether to generate and save the image of the color cloud
        path_to_save: path to where to save the color_cloud

    Returns
        dictionary TODO also give position TODO list of all pixel color, mapped between 0 and 1
    """

    im = Image.open(path)
    image = np.array(im.resize(size, Image.ANTIALIAS))
    f_image = image.reshape(size[0]*size[1],3).astype(float)
    # put rgb values between 0 and 1
    for i in range(3):
        f_image[:,i] = f_image[:,i]/255

    # plot them an save them
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(f_image[:,0], f_image[:,1], f_image[:,2], c=f_image, marker='o')
    ax.set_xlim3d(0, 1)
    ax.set_ylim3d(0, 1)
    ax.set_zlim3d(0, 1)
    plt.savefig(path_to_save + "/color_cloud.png")
    if show:
        plt.show()

if __name__=="__main__":
    generate_color_cloud("../example/img2.jpg")
