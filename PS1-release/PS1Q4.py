import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import color, io

class Prob4():
    def __init__(self):
        """Load input color image indoor.png and outdoor.png here as class variables."""

        self.indoor = None
        self.outdoor = None
        ###### START CODE HERE ######
        self.indoor = io.imread("indoor.png")
        self.outdoor = io.imread("outdoor.png")
        ###### END CODE HERE ######

    def rgb_split(self, img):
        return img[...,0], img[..., 1], img[..., 2]
    
    def prob_4_1(self):
        """Plot R,G,B channels separately and also their corresponding LAB space channels separately for both the indoor and outdoor image.
           Use the "gray" colormap options for plotting each channel."""
        
        ###### START CODE HERE ######
        images = [("Indoor", self.indoor), ("Outdoor", self.outdoor)]
        fig, axes = plt.subplots(4, 3, figsize=(12, 14))

        for i, (name, img) in enumerate(images):
            if img.ndim == 3 and img.shape[2] == 4: # sanity check for channels (debug)
                img = img[..., :3]

            R, G, B = self.rgb_split(img)
            
            lab = color.rgb2lab(img)
            l, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

            row_rgb, row_lab = i * 2, i * 2 + 1

            # RGB rows
            for ax, ch, title in zip(
                axes[row_rgb], [R, G, B],
                [f"{name} - R", f"{name} - G", f"{name} - B"]
            ):
                ax.imshow(ch, cmap="gray")
                ax.set_title(title)
                ax.axis("off")

            # LAB rows
            for ax, ch, title in zip(
                axes[row_lab], [l, a, b],
                [f"{name} - l", f"{name} - a", f"{name} - b"]
            ):
                ax.imshow(ch, cmap="gray")
                ax.set_title(title)
                ax.axis("off")

        plt.tight_layout()
        plt.show()
        ###### END CODE HERE ######
        return

    def prob_4_2(self):
        """
        Convert the loaded RGB image to HSV and return HSV matrix without using inbuilt functions. Return the HSV image as HSV. Plot the HSV image.
        Make sure to use a 3 channeled RGB image with floating point values lying between 0 - 1 for the conversion to HSV.

        Returns:
            HSV image (3 channeled image of size H x W x 3 with floating point values lying between 0 - 1 in each channel)
        """
        
        HSV = None
        ###### START CODE HERE ######
        image = io.imread("inputPS1Q4.jpg").astype(np.float32)
        
        # sanity checks
        if image.ndim == 3 and image.shape[-1] == 4:
            image = image[..., :3]
        if image.max() > 1.0:
            image /= 255.0
            
        r,g,b = image[...,0], image[...,1], image[...,2]
        max = np.max(image, axis=2)
        min = np.min(image, axis=2)
        chroma = max - min
        
        # VALUE (ez)
        value = max
        
        # SATURATION
        saturation = np.zeros_like(value)
        nozero_val = value > 0
        saturation[nozero_val] = chroma[nozero_val] / value[nozero_val]
        
        # HUE
        hue = np.zeros_like(value)
        nozero_chroma = chroma > 1e-8 # avoid DIV 0
        rmax = (max == r) & nozero_chroma
        gmax = (max == g) & nozero_chroma
        bmax = (max == b) & nozero_chroma
        
        hue[rmax] = ((g - b)[rmax] / chroma[rmax]) % 6
        hue[gmax] = ((b - r)[gmax] / chroma[gmax]) + 2
        hue[bmax] = ((r - g)[bmax] / chroma[bmax]) + 4
        
        hue = (hue / 6.0) % 1.0 # normalizzeeeeee
        
        HSV = np.stack([hue, saturation, value], axis=2).astype(np.float32)
        
        plt.figure(figsize=(6,6))
        plt.imshow(HSV)
        plt.title("HSV")
        plt.axis("off")
        plt.show()
        ###### END CODE HERE ######
        return HSV

        
if __name__ == '__main__':
    
    p4 = Prob4()
    p4.prob_4_1()
    HSV = p4.prob_4_2()
