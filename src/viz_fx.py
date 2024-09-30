import random
import matplotlib.pyplot as plt
from typing import List, Tuple
from PIL import Image

def display_random_images(image_paths: List[str], n: int = 25) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a grid of random images from the dataset using matplotlib.

    Args:
        image_paths (List[str]): List of paths to all images in the dataset.
        n (int): Number of images to display. Default is 25.

    Returns:
        Tuple[plt.Figure, plt.Axes]: A tuple containing the Figure and Axes objects.

    Example:
        fig, axes = display_random_images(filepaths, n=25)
        plt.show()  # To display the plot
        # or
        fig.savefig('random_images.png')  # To save the plot
    """
    random_images = random.sample(image_paths, n)
    grid_size = int(n**0.5)
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.flatten()
    for i, img_path in enumerate(random_images):
        img = Image.open(img_path)
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f"Image {i+1} from {img_path.split('/')[1]}")
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    return fig, axes