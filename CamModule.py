from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
from matplotlib import pyplot as plt

def imprep(img):
    img = img / 2 + 0.5     # Unnormalize
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    #plt.imshow(npimg)
    #plt.show()
    return npimg

def get_cam(model, target_layers, input_tensor, img_name):

    # Display a single image from the batch
    rgb_img = imprep(input_tensor[0])
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)

    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True, image_weight=0.7)
    plt.imshow(visualization)
    plt.savefig(f"/home/christian/Desktop/Continual/GradCAM/images/{img_name}.png", bbox_inches='tight')
