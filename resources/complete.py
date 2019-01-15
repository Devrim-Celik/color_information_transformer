from scipy.ndimage import imread
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from minimal_weight_perfect_matching import minimal_weight_matching

def complete(path1, path2, size=(35, 24), path_to_save="../images"):
        im1 = Image.open(path1)
        image1 = np.array(im1.resize(size, Image.ANTIALIAS))
        shape = image1.shape
        f_1 = image1.reshape(shape[0]*shape[1],3)

        im2 = Image.open(path2)
        image2 = np.array(im2.resize(size, Image.ANTIALIAS))
        f_2 = image2.reshape(shape[0]*shape[1],3)

        """
        # bin the pixels to multiples of 5
        for row in range(image1.shape[0]):
                for col in range(image1.shape[1]):
                        x = image1[row, col]
                        print(row, col, x, ((x+2)//5)*5)


        f_1 = np.array([((x+2)//5)*5 for x in f_1])
        f_2 = np.array([((x+2)//5)*5 for x in f_2])
        plt.figure("IMAGE 1")
        plt.imshow(f_1.reshape(size[1], size[0], 3))
        plt.figure("IMAGE 2")
        plt.imshow(f_2.reshape(size[1], size[0], 3))
        plt.show()
        """
        population, best, his1, his2 = minimal_weight_matching(f_1, f_2,  population_size= 40, nr_iterations=1000, m_rate=0.5)



        f_1_n = np.zeros((size[0]*size[1], 3)).astype(int)
        f_2_n = np.zeros((size[0]*size[1], 3)).astype(int)

        for x,y in enumerate(best[0]):
                f_1_n[x] = f_2[y]
                f_2_n[y] = f_1[x]

        plt.figure("HISTORY BEST")
        plt.plot(his1)
        plt.savefig("../images/history_best.jpg")

        plt.figure("HISTORY AVG")
        plt.plot(his2)
        plt.savefig("../images/history_avg.jpg")

        plt.figure("IMAGE 1")
        plt.imshow(f_1.reshape(size[1], size[0], 3))
        plt.savefig("../images/img1.jpg")

        plt.figure("IMAGE 2")
        plt.imshow(f_2.reshape(size[1], size[0], 3))
        plt.savefig("../images/img2.jpg")

        plt.figure("NEW IMAGE 1")
        plt.imshow(f_1_n.reshape(size[1], size[0], 3))
        plt.savefig("../images/created1.jpg")

        plt.figure("NEW IMAGE 2")
        plt.imshow(f_2_n.reshape(size[1], size[0], 3))
        plt.savefig("../images/created2.jpg")

        plt.show()



if __name__=="__main__":
        path1 = "../example/img1.jpg"
        path2 = "../example/img2.jpg"
        #path1 = "../example/test1.jpg"
        #path2 = "../example/test2.jpg"
        complete(path1, path2)
