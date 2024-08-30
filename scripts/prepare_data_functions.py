import os
import numpy as np

from typing import List, Union, Tuple, Optional

from sklearn.cluster import DBSCAN
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

import my_geom_functions as mgf

def select_elements_sort_DBSCAN(norm_dots_with_type: np.ndarray, eps: float = 1.6, 
                                min_samples: int = 13, n_clusters: int = None) -> list:
    """
    Applies DBSCAN clustering to select and sort elements, returning a list of clustered points.

    Parameters:
    - norm_dots_with_type (np.ndarray): A 2D array of shape (n, 3) or (n, 4), where n is the number of points. 
                                        The first three columns represent the (x, y, z) coordinates. 
                                        The optional fourth column represents the type of molecule.
    - eps (float): The maximum distance between two samples for them to be considered as in the same neighborhood. Default is 1.6.
    - min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point. Default is 13.
    - n_clusters (int, optional): The number of top clusters to return. If None, all clusters are returned.

    Returns:
    - list: A list of tuples, where each tuple contains the cluster index and the points belonging to that cluster.
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(norm_dots_with_type[:, :3])

    mask = labels != -1
    dots = norm_dots_with_type[mask]
    labels = labels[mask]

    unique_labels, cluster_counts = np.unique(labels, return_counts=True)
    sorted_cluster_indices = np.argsort(cluster_counts)[::-1]

    clustered_points = []
    for i in range(len(sorted_cluster_indices)):
        mask = labels == unique_labels[sorted_cluster_indices[i]]
        if len(dots[mask])>25:
            clustered_points.append((i, dots[mask]))

    if n_clusters is not None:
        clustered_points = clustered_points[:n_clusters]

    return clustered_points

def plot_image(x: np.ndarray, y: np.ndarray, xlim: list = [-np.pi, np.pi], ylim: list = [0, np.pi], 
               s: int = 15, color: tuple = (1.000, 0.350, 0.250), marker: str = 'o', 
               PIL_width: int = 24, PIL_height: int = 18, dpi: int = 10) -> Image.Image:
    """
    Plots an image based on x and y coordinates and returns the image as a PIL Image object.

    Parameters:
    - x (np.ndarray): Array of x-coordinates.
    - y (np.ndarray): Array of y-coordinates.
    - xlim (list): The x-axis limits. Default is [-np.pi, np.pi].
    - ylim (list): The y-axis limits. Default is [0, np.pi].
    - s (int): Marker size. Default is 15.
    - color (Union[tuple, str]): Marker color. Can be a tuple (RGB) or a string (e.g., 'b'). Default is (1.000, 0.350, 0.250)
    - marker (str): Marker type. Default is 'o'.
    - PIL_width (int): Width of the output PIL image. Default is 24.
    - PIL_height (int): Height of the output PIL image. Default is 18.
    - dpi (int): Dots per inch for the output image. Default is 10.

    Returns:
    - Image.Image: A PIL Image object of the plotted image.
    """
    fig, ax = plt.subplots(figsize=(4, 3))

    ax.scatter(x, y, marker=marker, color=color, s=s)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.tick_params(axis='both', which='both', length=0)
    
    ax.set_xticks([])
    ax.set_yticks([])    

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0, dpi=dpi)
    buffer.seek(0)
    plt.close()

    image_pil = Image.open(buffer)
    resized_image = image_pil.resize((PIL_width, PIL_height))
    
    resized_buffer = BytesIO()
    resized_image.save(resized_buffer, format='png')
    resized_buffer.seek(0)
  
    return Image.open(resized_buffer)

def plot_points_3D(points: Union[np.ndarray, list], color: list = ['r', 'g', 'b'], minmax: list = [-20., 20.]) -> None:
    """
    Plots 3D points in a scatter plot.

    Parameters:
    - points (np.ndarray or list):  A 2D array of shape (n, 3) or (n, 4) or a list of 2D arrays of points, 
                                    The first three columns represent the (x, y, z) coordinates. 
                                    The optional fourth column represents the type of molecule.
    - color (list): A list of colors for different sets of points. Default is ['r', 'g', 'b'].
    - minmax (list): The limits for the axes. Default is [-20., 20.].

    Returns:
    - None: Displays a 3D scatter plot.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if isinstance(points, list):
        for i in range(len(points)):
            ax.scatter(points[i][:, 0], points[i][:, 1], points[i][:, 2], c=color[i])
    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color[0])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim(minmax)
    ax.set_ylim(minmax)
    ax.set_zlim(minmax)

    plt.show()

def plot_points_2D(points: list, color: list = ['r', 'g', 'b'], labels: list = None, title: str = None) -> None:
    """
    Plots 2D projections of points as Phi-Theta images.

    Parameters:
    - points (list): A 2D array of shape (n, 3) or (n, 4), where n is the number of points. 
                     The first three columns represent the (x, y, z) coordinates. 
                     The optional fourth column represents the type of molecule.
    - color (list): A list of colors for different sets of points. Default is ['r', 'g', 'b'].
    - labels (list): A list of labels for each plot. Default is None.
    - title (str): A title for the entire plot. Default is None.

    Returns:
    - None: Displays the 2D projections as images.
    """
    fig, axs = plt.subplots(1, len(points), figsize=(4 * len(points), 4))

    if title:
        fig.text(0.5, 0.02, f'Predicted structure class: {title}', ha='center', va='center', fontsize=18)

    if len(points) == 1:
        phi_theta_image = get_phi_theta_image(points[0], color=color[0])
        axs.imshow(phi_theta_image)
        axs.set_title( f'Predicted label: {labels[0]}' if labels else 'Phi Theta Image', fontsize=16)
        axs.axis('off')
    else:
        for idx in range(len(points)):
            phi_theta_image = get_phi_theta_image(points[idx], color=color[idx])
            axs[idx].imshow(phi_theta_image)
            axs[idx].set_title( f'Predicted label: {labels[idx]}' if labels else 'Phi Theta Image', fontsize=16)
            axs[idx].axis('off')

    plt.tight_layout()
    plt.show()

def show_one_example_per_label(dataset: datasets.ImageFolder) -> None:
    """
    Displays one example image per label from the dataset.

    Parameters:
    - dataset (datasets.ImageFolder): The dataset containing labeled images.

    Returns:
    - None: Displays one example image per label.
    """
    label_to_image = {}
    for image, label in dataset:
        if label not in label_to_image:
            label_to_image[label] = image
        if len(label_to_image) == len(dataset.classes):
            break
    
    fig, axes = plt.subplots(1, len(label_to_image), figsize=(15, 5))
    for idx, (label, image) in enumerate(label_to_image.items()):
        ax = axes[idx]
        ax.imshow(image.squeeze(), cmap='gray')
        ax.set_title(dataset.classes[label])
        ax.axis('off')
    
    plt.show()

def get_phi_theta_image(dots: np.ndarray, a: float = 4, da: float = 6, color: Union[tuple, str] = (1.000, 0.350, 0.250)) -> Image.Image:
    """
    Generates an image of points in phi-theta coordinates.

    Parameters:
    - dots (np.ndarray):A 2D array of shape (n, 3) or (n, 4), where n is the number of points. 
                        The first three columns represent the (x, y, z) coordinates. 
                        The optional fourth column represents the type of molecule.
    - a (float): The minimum distance from the origin. Default is 4.
    - da (float): The range of distances to include. Default is 6.
    - color (Union[tuple, str]): Marker color. Can be a tuple (RGB) or a string (e.g., 'b').

    Returns:
    - Image.Image: A PIL Image object of the phi-theta image.
    """    
    f_d_r_dots = mgf.rotate_drop_flatten(dots, a=a, da=da)
    sp_dots = mgf.cartesian_to_spherical(f_d_r_dots)
    phi = sp_dots[:, 2]
    theta = sp_dots[:, 1]
    image_pil = plot_image(phi, theta, xlim=[-np.pi, np.pi], ylim=[0, np.pi], color = color)
    return image_pil

def create_folders(main_folder: str) -> None:
    """
    Creates a set of folders for data organization.

    Parameters:
    - main_folder (str): The path to the main folder.

    Returns:
    - None: Creates the folder structure.
    """
    os.makedirs(main_folder, exist_ok=True)
    for folder in ['train', 'val', 'test']:
        for label in ['0_Sector', '1_Part of helicoid', '2_Disk', 
                      '3_Helicoid', '4_Enneper', '5_Complex structure']:
            os.makedirs(f'{main_folder}/{folder}/{label}', exist_ok=True)

def create_phi_theta_dataset(source_folder: str, destination_folder: str) -> None:
    """
    Converts point cloud data to phi-theta images and saves them.

    Parameters:
    - source_folder (str): The path to the source folder containing point cloud data.
    - destination_folder (str): The path to the destination folder to save the images.

    Returns:
    - None: Converts and saves the images.
    """
    for folder in ['train', 'val', 'test']:
        for label in ['0_Sector', '1_Part of helicoid', '2_Disk', '3_Helicoid', '4_Enneper', '5_Complex structure']:
            original_path = f'{source_folder}/{folder}/{label}'
            new_path = f'{destination_folder}/{folder}/{label}'

            for file_name in os.listdir(original_path):
                if file_name.endswith('.npy'):
                    file_path = os.path.join(original_path, file_name)
                    dots = np.load(file_path)
                    image_pil = get_phi_theta_image(dots)
                    
                    image_name = file_name.replace('.npy', '.png')
                    image_pil.save(os.path.join(new_path, image_name))

def move_image(input_path: str, output_path: str, n: int) -> None:
    """
    Shifts image horizontally, moving part of the image from left to right and saves them.

    Parameters:
    - input_path (str): The path to the input image.
    - output_path (str): The path to save the rearranged image.
    - n (int): The number of sections to split the image into.

    Returns:
    - None: Saves the rearranged image.
    """
    image = Image.open(input_path)
    width, height = image.size
    crop_width = width // n

    left_part = image.crop((0, 0, crop_width, height))
    right_part = image.crop((crop_width, 0, width, height))

    new_image = Image.new('RGB', (width, height))
    new_image.paste(right_part, (0, 0))
    new_image.paste(left_part, (width - crop_width, 0))
    new_image.save(output_path)

    image.close()
    new_image.close()

def copy_images(main_folder_path: str, n_shift: int = 10) -> None:
    """
    Copies and shifts images within folders.

    Parameters:
    - main_folder_path (str): The path to the main folder containing images.
    - n_shift (int): The number of shifts to perform. Default is N_SHIFT.

    Returns:
    - None: Copies and shifts the images.
    """
    folders = [os.path.join(main_folder_path, f) for f in os.listdir(main_folder_path) if os.path.isdir(os.path.join(main_folder_path, f))]
    for folder in folders:
        files = [os.path.join(folder, file) for file in os.listdir(folder)]
        for image_path in files:
          name = os.path.splitext(os.path.basename(image_path))[0]
          for i in range(n_shift-1):
            new_file_name = name + f"_shift_{i+1}.jpg"
            new_image_path = os.path.join(os.path.dirname(image_path), new_file_name)
            move_image(image_path, new_image_path, n_shift)
            image_path = new_image_path

def load_dataset(dataset_path: str) -> datasets.ImageFolder:
    """
    Loads an image dataset with transformations.

    Parameters:
    - dataset_path (str): The path to the dataset.

    Returns:
    - datasets.ImageFolder: The loaded image dataset.
    """
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

    return dataset
