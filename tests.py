import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def evaluate_images(ground_truth_path, generated_image_path):
    """
    Compare a ground truth color image with a generated colorized image.

    Args:
        ground_truth_path (str): Path to the original full-color image.
        generated_image_path (str): Path to the colorized image.

    Returns:
        tuple: (PSNR value, SSIM value)
    """

    # Load images
    ground_truth = cv2.imread(ground_truth_path)
    generated_image = cv2.imread(generated_image_path)

    if ground_truth is None:
        raise ValueError(f"Failed to load ground truth image: {ground_truth_path}")
    if generated_image is None:
        raise ValueError(f"Failed to load generated image: {generated_image_path}")

    # Resize generated image to match ground truth
    if ground_truth.shape != generated_image.shape:
        generated_image = cv2.resize(
            generated_image, (ground_truth.shape[1], ground_truth.shape[0])
        )

    # Convert BGR to RGB
    ground_truth_rgb = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2RGB)
    generated_rgb = cv2.cvtColor(generated_image, cv2.COLOR_BGR2RGB)

    # Calculate PSNR
    psnr_value = psnr(ground_truth_rgb, generated_rgb, data_range=255)

    # Calculate SSIM (channel_axis is used instead of multichannel for latest skimage)
    ssim_value = ssim(ground_truth_rgb, generated_rgb, channel_axis=2, data_range=255)

    return psnr_value, ssim_value


if __name__ == "__main__":
    # Example usage

    ground_truth_path = r"Scarlett_Johannson.jpg"
    generated_image_path = r"Scarlett_JohannsonColorized.jpg"

    psnr_val, ssim_val = evaluate_images(ground_truth_path, generated_image_path)

    print(f"PSNR: {psnr_val:.2f} dB")
    print(f"SSIM: {ssim_val:.4f}")

    ground_truth_path = r"Will_Smith.jpg"
    generated_image_path = r"Will_Smith_Colorized.jpg"

    psnr_val, ssim_val = evaluate_images(ground_truth_path, generated_image_path)

    print(f"PSNR: {psnr_val:.2f} dB")
    print(f"SSIM: {ssim_val:.4f}")
