import torch  # PyTorch library for deep learning
from tqdm import tqdm  # For showing progress bars during loops
import argparse  # For parsing command-line arguments
from loader.dataloader_cifar10 import DataloaderCifar10  # Custom CIFAR-10 dataset class
from models.vgg_simplified import Vgg  # Simplified VGG model class

####### Training Settings #########
# Create an argument parser to handle command-line arguments
parser = argparse.ArgumentParser()

# Add argument for batch size (default is 128)
parser.add_argument("--batch_size", default=128, type=int)

# Add argument for the size of the fully connected layer (default is 512)
parser.add_argument("--fc_layer", default=512, type=int)

# Add argument for the path to the trained model file (default path is 'runs/66192/last_model.pkl')
parser.add_argument("--model_path", default='runs/66192/last_model.pkl', type=str, 
                    help='path to the trained model, please change the default value to the path of your trained model')

# Add argument for the dataset directory path (default is 'data/data_cnn/cifar-10-batches-py')
parser.add_argument("--root", default='data/data_cnn/cifar-10-batches-py', type=str, 
                    help='path to dataset folder')

# Parse the arguments from the command-line input
args = parser.parse_args()
###################################

# Check if CUDA is available, otherwise fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Alternatively, you could force it to run on CPU by uncommenting the following line:
# device = torch.device('cpu')

# Function to test the trained model on the CIFAR-10 test set
def test(args):
    # Load the CIFAR-10 test dataset with transformation enabled
    test_dataset = DataloaderCifar10(img_size=32, is_transform=True, split='test')
    
    # Load the data from the specified directory
    test_dataset.load_data(args.root)

    # Create a DataLoader for the test dataset
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                                  batch_size=args.batch_size, 
                                                  shuffle=False,  # No need to shuffle test data
                                                  num_workers=0)  # Set the number of workers for data loading

    # Initialize the model with the fully connected layer size and number of output classes
    model = Vgg(fc_layer=args.fc_layer, classes=10).to(device)

    # Load the saved model weights from the specified checkpoint file
    checkpoint = torch.load(args.model_path, map_location=lambda storage, loc: storage)  # Load model to device
    model.load_state_dict(checkpoint['model_state'])  # Load the saved state dictionary into the model
    model.eval()  # Set the model to evaluation mode

    # Initialize variables to keep track of total samples and correct predictions
    total, correct = 0, 0

    # Disable gradient computation for testing (saves memory and computation)
    with torch.no_grad():
        # Loop over the test dataset in batches
        for step, data in tqdm(enumerate(test_dataloader)):
            # Get the input images and labels, and move them to the specified device (CPU or GPU)
            imgs = data[0].to(device)  # [batch_size, 3, 32, 32] - RGB images
            labels = data[1].to(device)  # [batch_size] - corresponding labels

            # Perform forward pass to get model predictions
            preds = model(imgs)

            # Get the predicted class by taking the maximum value along the class dimension
            _, predicted = preds.max(1)

            # Update the total number of samples
            total += labels.size(0)

            # Update the number of correct predictions
            correct += predicted.eq(labels).sum().item()

    # Calculate the overall test accuracy
    acc = 100.0 * correct / total
    print('test accuracy:', acc)  # Output the test accuracy

# If this script is run directly, call the test function with parsed arguments
if __name__ == '__main__':
    test(args)
