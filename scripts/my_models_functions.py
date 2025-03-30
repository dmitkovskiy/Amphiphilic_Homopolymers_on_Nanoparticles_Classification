import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from PIL import Image

import prepare_data_functions as pdf

class CNN_Net(nn.Module):
        def __init__(self, image_height=18, image_width=24):
            super().__init__()

            self.block1 = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout2d(p=0.25)
            )
            
            self.block2 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout2d(p=0.25)
            )
            
            h, w = calculate_output_size(image_height, image_width)

            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * h * w, 128),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(128, 6)
            )

        def forward(self, x):
            x = self.block1(x)
            x = self.block2(x)
            x = self.classifier(x)
            return x

def calculate_output_size(image_height: int, image_width: int) -> tuple[int, int]:
    """
    Calculates the output size after two convolution and pooling layers.
    
    Parameters:
    - image_height (int): Height of the input image.
    - image_width (int): Width of the input image.
    
    Returns:
    - tuple[int, int]: Height and width of the image after the layers.
    """
    
    def conv2d_size(size: int, kernel_size: int = 3, stride: int = 1, padding: int = 1) -> int:
        return (size - kernel_size + 2 * padding) // stride + 1

    def maxpool2d_size(size: int, kernel_size: int = 2, stride: int = 2) -> int:
        return (size - kernel_size) // stride + 1
    
    h, w = conv2d_size(image_height), conv2d_size(image_width)
    h, w = maxpool2d_size(h), maxpool2d_size(w)
    
    h, w = conv2d_size(h), conv2d_size(w)
    h, w = maxpool2d_size(h), maxpool2d_size(w)
    
    return h, w

def load_my_image_pred_model(ckpt_path: str, image_height=18, image_width=24) -> CNN_Net:
    """
    Loads a trained CNN model from a file.
    
    Parameters:
    - ckpt_path (str): Path to the model file.
    - image_height (int): Height of the input images. Default is 18.
    - image_width (int): Width of the input images. Default is 24.
    
    Returns:
    - CNN_Net: The loaded CNN model.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    state_dict = {}
    for key in checkpoint["state_dict"].keys():
        key_new = key.lstrip("model.")
        state_dict[key_new] = checkpoint["state_dict"][key]

    model = CNN_Net(image_height, image_width)
    model.load_state_dict(state_dict)
    model.eval()

    return model

def move_image(image_pil: Image.Image, n: int = 10) -> Image.Image:
    """
    Shifts image horizontally, moving part of the image from left to right.
    
    Parameters:
    - image_pil (Image.Image): The input PIL image.
    - n (int): Number of parts to divide the image width. Default is 10.
    
    Returns:
    - Image.Image: The shifted PIL image.
    """

    width, height = image_pil.size
    crop_width = width // n

    left_part = image_pil.crop((0, 0, crop_width, height))
    right_part = image_pil.crop((crop_width, 0, width, height))

    new_image_pil = Image.new('RGB', (width, height))
    new_image_pil.paste(right_part, (0, 0))
    new_image_pil.paste(left_part, (width - crop_width, 0))

    return new_image_pil

def predict_image(image: Image.Image, model: torch.nn.Module) -> tuple[int, float]:
    """
    Predicts the class of an input image using a trained model.
    
    Parameters:
    - image (Image.Image): The input PIL image.
    - model (torch.nn.Module): The trained model.
    
    Returns:
    - tuple[int, float]: Predicted class label and its probability.
    """    
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)
    _, predicted = torch.max(outputs, 1)
    predicted_class = predicted.item()
    
    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    probability = probabilities[predicted_class].item()

    return predicted_class, probability

def get_class_name(class_number: int) -> str:
    """
    Maps a class index to its name.
    
    Parameters:
    - class_number (int): The class index.
    
    Returns:
    - str: The name of the class.
    """
    
    class_names = {
    0: "Sector",
    1: "Part of helicoid",
    2: "Disk",
    3: "Helicoid",
    4: "Enneper",
    5: "Complex structure"
    }
        
    return class_names.get(class_number, "Unknown")

def predict_el_class(image_pil: Image.Image, n_shift: int, model: torch.nn.Module) -> tuple[str, float]:
    """
    Predicts the class of an image with shifting.
    
    Parameters:
    - image_pil (Image.Image): The input PIL image.
    - n_shift (int): Number of shifts to perform.
    - model (torch.nn.Module): The trained model.
    
    Returns:
    - tuple[str, float]: Predicted class name and its probability.
    """
    preds = {}
    for i in range(n_shift):
        pred, prob = predict_image(image_pil, model)
        preds[i] = (pred, prob)
        image_pil = move_image(image_pil, n_shift)

    class_weights = {}
    for prediction in preds.values():
        predicted_class, probability = prediction
        class_weights[predicted_class] = class_weights.get(predicted_class, 0) + probability

    element = max(class_weights, key=class_weights.get)
    normalized_probability = class_weights[element] / sum(class_weights.values())
    el_name = get_class_name(int(element))

    return el_name, normalized_probability

def custom_sort(arr: list[str]) -> list[str]:
    """
    Sorts a list of class names in a predefined order.
    
    Parameters:
    - arr (list[str]): List of class names.
    
    Returns:
    - list[str]: Sorted list of class names.
    """
    order = {
        "Sector": 0,
        "Part of helicoid": 1,
        "Disk": 2,
        "Helicoid": 3,
        "Enneper": 4,
        "Complex structure": 5
    }
    return sorted(arr, key=lambda x: order.get(x, float('inf')))

def map_values(input_arr: list[str]) -> tuple[str, any, any]:
    """
    Maps a list of class names to a structured classification.
    
    Parameters:
    - input_arr (list[str]): List of class names.
    
    Returns:
    - tuple[str, any, any]: Mapped structure and its features.
    """
    elements = ' '.join(input_arr)
    mapping = {
        "Sector": ("Sector", np.NaN, np.NaN),


        "Sector Sector":                    ("Double helicoid", np.NaN, np.NaN),
        "Sector Part of helicoid":          ("Double helicoid", np.NaN, np.NaN),
        "Part of helicoid Part of helicoid":("Double helicoid", np.NaN, np.NaN),
        
        "Disk":                             ("Disk", np.NaN, np.NaN),
        "Sector Disk":                      ("Disk", "with sector", np.NaN),
        "Sector Sector Disk":               ("Disk", "with two sectors", np.NaN),

        "Disk Disk":                        ("Catenoid", np.NaN, np.NaN),
        "Sector Disk  Disk":                ("Catenoid", "with sector", np.NaN),
        "Sector Sector Disk  Disk":         ("Catenoid", "with two sectors", np.NaN),

        "Disk Disk Disk":                   ("Costa", np.NaN, np.NaN),
        "Sector Disk Disk Disk":            ("Costa", "with sector", np.NaN),

        "Helicoid":                         ("Helicoid", np.NaN, np.NaN),
        "Disk Helicoid":                    ("Helicoid", "with disk", np.NaN),
        "Part of helicoid Disk":            ("Helicoid", "with disk", np.NaN),
        "Sector Disk Helicoid":             ("Helicoid", "with disk", "and sector"),
        
        "Enneper":                          ("Enneper", np.NaN, np.NaN),
        "Sector Enneper":                   ("Enneper", "with sector", np.NaN),
        "Sector Sector Enneper":            ("Enneper", "with two sectors", np.NaN),
    
        "Complex structure":                ("Complex structure", np.NaN, np.NaN),
    }
    
    return mapping.get(elements, (elements, np.NaN, np.NaN))

def pred_structure(structure: np.ndarray, model: torch.nn.Module, n_shift: int = 10) -> tuple[str, any, any]:
    """
    Predicts all the elements of the structure by their points.
    
    Parameters:
    - structure (np.ndarray): Array of points representing the structure.
    - models (dict): Dictionary of models.
    - model_errors (dict): Dictionary of model errors for different classes.
    
    Returns:
    - tuple[str, any, any]: Predicted structure and its features.
    """

    clustered_dots = pdf.select_elements_sort_DBSCAN(structure)

    elements = []
    for _, el_dots in clustered_dots:

        dots_B = el_dots[el_dots[:, 3] == 2.]
        image_pil = pdf.get_phi_theta_image(dots_B)
        pred_el, _ = predict_el_class(image_pil, n_shift, model)
        elements.append(pred_el)

    sorted_elements = custom_sort(elements)
    structure, feature_1, feature_2 = map_values(sorted_elements)

    return  structure, feature_1, feature_2

def demo_predict_pipeline(structure: np.ndarray, model: torch.nn.Module, n_shift: int = 10, SP_diffuse_param = [0.65, 0.3,1]) -> None:
    """
    Demonstrates a pipeline for predicting classes of structures.
    
    Parameters:
    - structure (np.ndarray): Array of points representing the structure.
    - models (dict): Dictionary of models.
    - model_errors (dict): Dictionary of model errors for different classes.
    
    Returns:
    - None: Displays a 3D scatter plot and the 2D projections as images.
    """

    clustered_elements = pdf.select_elements_sort_DBSCAN(structure)    

    clustered_dots = [el_dots for _, el_dots in clustered_elements]
    pdf.plot_points_3D(clustered_dots)
    print('\n\n')
    labels = []
    for _, el_dots in clustered_elements:

        dots_B = el_dots[el_dots[:, 3] == 2.]
        image_pil = pdf.get_phi_theta_image(dots_B)
        pred_el, _ = predict_el_class(image_pil, n_shift, model)
        labels.append(pred_el)
    title = pred_structure(structure, model)[0]
    
    pdf.plot_points_2D(clustered_dots, labels = labels, title = title)