import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from PIL import Image
import gdown
import os
from .cnn_model import CNN_LSTM_Model


def download_model_from_drive(save_path):
    """
    Downloads a PyTorch model from Google Drive and saves it to the specified path.

    Args:
        save_path (str): The path where the model file will be saved.
    """
    download_url = "https://drive.google.com/file/d/1---jSknfU_n_069Fk3Lw-T4m7MMLciYG/uc?export=download"
    gdown.download(download_url, save_path, quiet=False)


def load_model(model_path, num_future_steps=3):
    """
    Loads a PyTorch model from a specified path and prepares it for inference.

    Args:
        model_path (str): The path to the PyTorch model file.

    Returns:
        torch.nn.Module: The loaded and initialized PyTorch model.
    """
    if num_future_steps != 3:
        model = CNN_LSTM_Model(num_future_steps=num_future_steps)
    else:
        model = CNN_LSTM_Model()

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


def preprocess_input(file_loc, length=10):
    """
    Preprocesses a sequence of images for input to the CNN-LSTM model.

    Args:
        file_loc (str): The directory containing the image files.
        length (int): The number of images to include in the sequence.

    Returns:
        torch.Tensor: The preprocessed input data as a tensor.
    """
    image_files = sorted([os.path.join(file_loc, img) for img in os.listdir(file_loc) if img.endswith('.jpg')])[251-length:251]
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    batch_size = 1
    sequence_length = len(image_files)
    num_channels = 1
    width, height = 128, 128
    tensor_batch = torch.empty((batch_size, sequence_length, num_channels, height, width))

    for i, image_file in enumerate(image_files):
        image = Image.open(image_file)
        tensor_image = transform(image)
        tensor_batch[0, i] = tensor_image

    return tensor_batch


def perform_inference(model, tensor_batch):
    """
    Performs inference using the CNN-LSTM model on the preprocessed input data.

    Args:
        model (torch.nn.Module): The loaded CNN-LSTM model.
        tensor_batch (torch.Tensor): The preprocessed input data as a tensor.

    Returns:
        torch.Tensor: The predicted images.
    """
    predicted_images = model(tensor_batch).unsqueeze(0)
    return predicted_images


def save_images(name):
    """
    Saves a sequence of images to individual files.

    Args:
        name (torch.Tensor): The sequence of images to be saved.
    """
    tensor = name.squeeze(0)
    for i in range(tensor.size(0)):
        save_image(tesnor=tensor[i], fp=f'tst_{251+i}.jpg')


def plot_images(name):
    """
    plots the sequence of iamges produced

    Args:
        name (torch.Tensor): The sequence of images to be plotted.
    """
    print(name.size())
    fig, axs = plt.subplots(1, name.shape[1], figsize=(20, 5))
    for i in range(name.shape[1]):
        img = name[0, i].cpu().detach().numpy().squeeze()
        axs[i].imshow(img, cmap='gray')
        axs[i].set_title(f'image {i + 1 }')
        axs[i].axis('off')
