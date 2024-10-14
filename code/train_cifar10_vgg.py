import torch  # PyTorch library for deep learning
import time  # For tracking time durations
from tqdm import tqdm  # For displaying progress bars during loops
import argparse  # For parsing command-line arguments
from loader.dataloader_cifar10 import DataloaderCifar10  # Custom CIFAR-10 dataset class
from models.vgg_simplified import Vgg  # Simplified VGG model class
from utils import *  # Utility functions for logging, etc.
import torch.nn.functional as F  # For common neural network functions, such as loss functions

####### Training Settings #########
# Create an argument parser to handle command-line arguments
parser = argparse.ArgumentParser()

# Add argument for batch size (default is 128)
parser.add_argument("--batch_size", default=128, type=int)

# Add argument for logging frequency (default is 100 steps)
parser.add_argument("--log_step", default=100, type=int, help='how many steps to log once')

# Add argument for validation frequency (default is 100 steps)
parser.add_argument("--val_step", default=100, type=int)

# Add argument for the number of training epochs (default is 50)
parser.add_argument("--num_epoch", default=50, type=int, help='maximum num of training epochs')

# Add argument for the size of the fully connected layer (default is 512)
parser.add_argument("--fc_layer", default=512, type=int, help='feature number the first linear layer in VGG')

# Add argument for learning rate (default is 0.0001)
parser.add_argument("--lr", default=0.0001, type=float, help='learning rate')

# Add argument for directory to save models and logs (default is 'runs')
parser.add_argument("--save_dir", default='runs', type=str, help='path to save trained models and logs')

# Add argument for the dataset root directory (default is 'data/data_cnn/cifar-10-batches-py')
parser.add_argument("--root", default='data/data_cnn/cifar-10-batches-py', type=str, help='path to dataset folder')

# Parse the arguments from the command-line input
args = parser.parse_args()
###################################

# Check if CUDA is available, otherwise fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Alternatively, you could force it to run on CPU by uncommenting the following line:
# device = torch.device('cpu')

# Function to train the model
def train(writer, logger):
    # Load the CIFAR-10 training dataset with transformation enabled
    train_dataset = DataloaderCifar10(img_size=32, is_transform=True, split='train')
    train_dataset.load_data(args.root)

    # Create a DataLoader for the training dataset
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                                   batch_size=args.batch_size, 
                                                   shuffle=True,  # Shuffle the dataset during training
                                                   num_workers=2)  # Number of workers for data loading

    # Load the CIFAR-10 validation dataset without transformation
    val_dataset = DataloaderCifar10(img_size=32, is_transform=False, split='val')
    val_dataset.load_data(args.root)

    # Create a DataLoader for the validation dataset
    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                                 batch_size=args.batch_size, 
                                                 shuffle=False,  # No need to shuffle validation data
                                                 num_workers=2)

    # Initialize the VGG model with the specified fully connected layer size and output classes
    model = Vgg(fc_layer=args.fc_layer, classes=10).to(device)

    # Define the optimizer (Adam) with the specified learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    total_steps = 0  # Track the number of steps taken in training
    best_acc = 0  # Track the best validation accuracy achieved

    # Training loop for the specified number of epochs
    for epoch in tqdm(range(args.num_epoch)):
        for step, data in enumerate(train_dataloader):
            start_ts = time.time()  # Record the start time of the step
            total_steps += 1  # Increment the total step counter

            model.train()  # Set the model to training mode
            imgs = data[0].to(device)  # Move the batch of images to the device (GPU/CPU)
            label = data[1].to(device)  # Move the corresponding labels to the device

            optimizer.zero_grad()  # Zero out the gradients for the optimizer

            # Forward pass through the model
            preds = model(imgs)

            # Compute the cross-entropy loss
            loss = F.cross_entropy(input=preds, target=label, reduction='mean')

            # Backpropagate the loss
            loss.backward()
            optimizer.step()  # Update the model's weights

            # Log training progress every log_step iterations
            if total_steps % args.log_step == 0:
                print_str = '[Step {:d}/ Epoch {:d}]  Loss: {:.4f}'.format(step, epoch, loss.item())
                logger.info(print_str)  # Log the progress
                writer.add_scalar('train/loss', loss.item(), total_steps)  # Write loss to TensorBoard
                print(print_str)

            # Perform validation every val_step iterations
            if total_steps % args.val_step == 0:
                model.eval()  # Set the model to evaluation mode
                total, correct = 0, 0  # Track the number of samples and correct predictions

                with torch.no_grad():  # Disable gradient computation for validation
                    for step_val, data in enumerate(val_dataloader):
                        imgs = data[0].to(device)  # Move images to the device
                        labels = data[1].to(device)  # Move labels to the device
                        preds = model(imgs)  # Get predictions from the model

                        # Get the predicted class with the highest score
                        _, predicted = preds.max(1)
                        total += labels.size(0)  # Update the total number of samples
                        correct += predicted.eq(labels).sum().item()  # Update the number of correct predictions

                    # Calculate the validation accuracy
                    acc = 100.0 * correct / total
                    logger.info('val acc: {}'.format(acc))  # Log the validation accuracy
                    writer.add_scalar('val/acc', acc, total_steps)  # Write accuracy to TensorBoard
                    print('Validation Accuracy:', acc)

                    # Save the model if the current validation accuracy is the best
                    if acc > best_acc:
                        best_acc = acc
                        state = {
                            "epoch": epoch,
                            "total_steps": total_steps,
                            "model_state": model.state_dict(),
                            "best_acc": acc,
                        }
                        save_path = os.path.join(writer.file_writer.get_logdir(), "last_model.pkl")
                        torch.save(state, save_path)  # Save the best model
                        print('[*] Best model saved\n')
                        logger.info('[*] Best model saved\n')

        # Break the loop if the maximum number of epochs is reached
        if epoch == args.num_epoch:
            break

# Main block for running the training script
if __name__ == '__main__':
    # Generate a random run ID and create a new directory to store logs and models
    run_id = random.randint(1, 100000)
    logdir = os.path.join(args.save_dir, str(run_id))  # Create a new directory for the current run
    writer = SummaryWriter(log_dir=logdir)  # Initialize the TensorBoard writer
    print('RUNDIR: {}'.format(logdir))  # Print the log directory path

    # Initialize the logger for logging training progress
    logger = get_logger(logdir)
    logger.info('Let the games begin')  # Log the start of training

    # Save the training configuration
    save_config(logdir, args)

    # Start the training process
    train(writer, logger)
