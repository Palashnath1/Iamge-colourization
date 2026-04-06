import numpy as np
import cv2
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import threading
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Paths to load the model
DIR = r"F:\IMAGE_COLORIZATION_2"
PROTOTXT = os.path.join(DIR, r"Colorization_Models\colorization_deploy_v2.prototxt")
POINTS = os.path.join(DIR, r"Colorization_Models\pts_in_hull.npy")
MODEL = os.path.join(DIR, r"Colorization_Models\colorization_release_v2.caffemodel")

# Load the Caffe model
print("Loading colorization model...")
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
pts = np.load(POINTS).transpose().reshape(2, 313, 1, 1)

# Load centers for ab channel quantization used for rebalancing
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]


# Function to colorize the image
def colorize_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        messagebox.showerror("Error", f"Failed to load image from path: {image_path}")
        return None, None

    # Preprocess the image for the neural network
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    # Run the model to predict the ab channels
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    # Resize the predicted ab channels back to the size of the original image
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

    # Combine the original L channel with the predicted ab channels
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

    # Convert LAB back to BGR
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")

    return image, colorized


# Function to evaluate PSNR & SSIM
def evaluate_images(ground_truth_path, generated_image):
    ground_truth = cv2.imread(ground_truth_path)
    if ground_truth is None:
        messagebox.showerror(
            "Error", f"Failed to load ground truth image:\n{ground_truth_path}"
        )
        return None, None

    if ground_truth.shape != generated_image.shape:
        generated_image = cv2.resize(
            generated_image, (ground_truth.shape[1], ground_truth.shape[0])
        )

    gt_rgb = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2RGB)
    gen_rgb = cv2.cvtColor(generated_image, cv2.COLOR_BGR2RGB)

    psnr_value = psnr(gt_rgb, gen_rgb, data_range=255)
    ssim_value, _ = ssim(gt_rgb, gen_rgb, full=True, multichannel=True)

    return psnr_value, ssim_value


# Function to select an image
def select_image():
    global original_image_path
    original_image_path = filedialog.askopenfilename(
        title="Select a Black and White Image",
        filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")],
    )
    if original_image_path:
        preview_image(original_image_path)
        messagebox.showinfo(
            "Image Selected",
            "Image loaded successfully! Click on 'Colorize Image' to continue.",
        )
    else:
        messagebox.showerror("Error", "No image selected.")


# Function to preview selected image
def preview_image(image_path):
    img = Image.open(image_path)
    img.thumbnail((200, 200))
    img = ImageTk.PhotoImage(img)
    preview_label.configure(image=img)
    preview_label.image = img


# Function to handle colorization in a background thread
def start_colorization():
    if not original_image_path:
        messagebox.showerror("Error", "Please select an image first.")
        return
    progress_bar.start()
    thread = threading.Thread(target=colorize_in_background)
    thread.start()


# Function to run colorization and then call the display function
def colorize_in_background():
    original_image, colorized_image = colorize_image(original_image_path)
    if original_image is None or colorized_image is None:
        progress_bar.stop()
        return
    root.after(0, lambda: display_images(original_image, colorized_image))


# Function to display the images and save
def display_images(original_image, colorized_image):
    original_image_pil = Image.fromarray(
        cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    )
    colorized_image_pil = Image.fromarray(
        cv2.cvtColor(colorized_image, cv2.COLOR_BGR2RGB)
    )

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_image_pil)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Colorized Image")
    plt.imshow(colorized_image_pil)
    plt.axis("off")
    plt.show()

    # Ask user for ground truth image to evaluate
    gt_path = filedialog.askopenfilename(
        title="Select Ground Truth Color Image",
        filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")],
    )
    if gt_path:
        psnr_val, ssim_val = evaluate_images(gt_path, colorized_image)
        if psnr_val is not None:
            messagebox.showinfo(
                "Evaluation Results", f"PSNR: {psnr_val:.2f} dB\nSSIM: {ssim_val:.4f}"
            )
            print(f"PSNR: {psnr_val:.2f} dB")
            print(f"SSIM: {ssim_val:.4f}")

    progress_bar.stop()

    # Save button
    def save_image():
        save_path = filedialog.asksaveasfilename(
            defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg")]
        )
        if save_path:
            colorized_image_pil.save(save_path)
            messagebox.showinfo("Success", f"Image saved to {save_path}")

    download_button.config(state=tk.NORMAL, command=save_image)


# Create tkinter window
root = tk.Tk()
root.title("Enhanced Image Colorizer")
root.geometry("500x400")
root.configure(bg="#f0f0f0")

# Styles
button_style = {
    "font": ("Arial", 14),
    "bg": "#4CAF50",
    "fg": "white",
    "relief": "raised",
}
label_style = {"font": ("Arial", 12), "bg": "#f0f0f0", "fg": "#333"}

# Heading
heading = tk.Label(
    root,
    text="Welcome to the Image Colorizer",
    font=("Arial", 16),
    bg="#f0f0f0",
    fg="#333",
)
heading.pack(pady=10)

# Preview label
preview_label = tk.Label(root, text="Image Preview", **label_style)
preview_label.pack(pady=10)

# Buttons
select_image_button = tk.Button(
    root, text="Select Image", **button_style, command=select_image
)
select_image_button.pack(pady=10)

colorize_image_button = tk.Button(
    root, text="Colorize Image", **button_style, command=start_colorization
)
colorize_image_button.pack(pady=10)

# Progress bar
progress_bar = ttk.Progressbar(root, mode="indeterminate")
progress_bar.pack(pady=10, fill=tk.X)

# Download button
download_button = tk.Button(
    root, text="Download Colorized Image", state=tk.DISABLED, **button_style
)
download_button.pack(pady=10)

# Start loop
root.mainloop()
