import torch
import torchvision
import torchvision.transforms as transforms
import torch.hub
import numpy as np
from CamModule import get_cam

# Load pretrained model
model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
model.eval()  # Set the model to evaluation mode
target_layers = [model.layer3[2].conv2]

# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize images
])

batch_size = 3
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
test_iterator = iter(testloader)


def imshow(img):
    img = img / 2 + 0.5     # Unnormalize
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    #plt.imshow(npimg)
    #plt.show()
    return npimg

for x in range(30):
    input_tensor, labels = next(test_iterator)
    get_cam(model, target_layers, input_tensor, x)
