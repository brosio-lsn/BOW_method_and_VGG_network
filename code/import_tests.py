# import_test.py
import numpy
import numpy as np
import tensorboard
import tensorboardX
import torch
import torchvision
import tqdm
import cv2

def test_numpy():
    # Test numpy functionality
    array = numpy.array([1, 2, 3])
    print("NumPy array created:", array)

def test_torch():
    # Test basic PyTorch functionality
    x = torch.rand(5, 3)
    print("Random tensor from torch:", x)

    if torch.cuda.is_available():
        x = x.cuda()
        print("CUDA is available. Tensor moved to GPU:", x)
    else:
        print("CUDA is not available.")

def test_torchvision():
    # Test torchvision by loading an example model
    model = torchvision.models.alexnet(pretrained=False)
    print(f"Torchvision model {model.__class__.__name__} created.")

def test_cv2():
    # Test basic OpenCV functionality
    sample_array = numpy.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=numpy.uint8)
    colored_image = cv2.cvtColor(sample_array, cv2.COLOR_RGB2BGR)
    print("OpenCV image with BGR color space:", colored_image)

def main():
    Idx = np.array([1,4,1,3,4,4])
    histo = np.bincount(Idx, minlength=5)
    print(histo)
if __name__ == "__main__":
    main()
