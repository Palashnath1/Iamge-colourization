# 🎨 Image Colorization Project

This project uses Deep Learning and OpenCV to convert black & white images into realistic colorized images using a pre-trained model.

---

## 🚀 Features

* 🖼️ Convert grayscale images to color
* 🧠 Uses pre-trained Caffe deep learning model
* 🖥️ Simple GUI built with Tkinter
* 📂 Supports JPG, PNG, JPEG formats
* 💾 Download/save colorized images

---

## 📁 Project Structure

```
IMAGE_COLORIZATION_2/
│
├── colorize5.py
├── Colorization_Models/
│   ├── colorization_deploy_v2.prototxt
│   ├── colorization_release_v2.caffemodel
│   ├── pts_in_hull.npy
│
├── images/
├── output/
├── README.md
```

---

## ⚙️ Installation

1. Clone the repository:

```
git clone https://github.com/YOUR-USERNAME/Iamge-colourization.git
cd Iamge-colourization
```

2. Install dependencies:

```
pip install opencv-python numpy pillow matplotlib
```

---

## 📥 Download Model Files (Important)

Due to GitHub size limits, model files are not included.

Download the following files manually:

* `colorization_deploy_v2.prototxt`
* `colorization_release_v2.caffemodel`
* `pts_in_hull.npy`

👉 Place them inside:

```
Colorization_Models/
```

---

## ▶️ How to Run

```
python colorize5.py
```

---

## 🖱️ How to Use

1. Click **Select Image**
2. Choose a black & white image
3. Click **Colorize Image**
4. View results
5. Save the colorized image

---

## 🧠 Technology Used

* Python
* OpenCV (DNN Module)
* NumPy
* Tkinter (GUI)
* Matplotlib

---

## ⚠️ Notes

* Ensure model files are placed correctly
* Large model files are excluded using `.gitignore`
* Works best with clear grayscale images

---

## 📸 Sample Output

| Original  | Colorized     |
| --------- | ------------- |
| B&W Image | Colored Image |

---

## 🙌 Acknowledgements

* OpenCV Team
* Deep Learning Caffe Model Contributors

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!

---
