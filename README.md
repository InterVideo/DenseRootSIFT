# DenseRootSIFT
Implementation of Dense Root SIFT wrapper using OpenCV

## Usage

```python
import cv2

image = cv2.imread('image.jpg')
dense_root_sift = DenseRootSIFT()
descriptors = dense_root_sift.detectAndCompute(image, window_size=None)
```
