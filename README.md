# Aesthetic Filter using Computer Vision (3D Effect with Neon Lights) 

## Project Overview:
This project applies a custom aesthetic filter to input images using a combination of warped line masks, motion blur, clustering, and edge enhancement. The final effect is a stylized transformation with a cyberpunk flair, emphasizing vivid colors, motion distortion, and glowing outlines. The process involves several advanced image processing techniques implemented in MATLAB.  

## 🖼️ Input Images:  

The project processes two input images:  

<p align="center">
  <img src="https://github.com/user-attachments/assets/6241e72d-44ab-4781-b5c6-ddc4ecfe1585" width="45%" />
  <img src="https://github.com/user-attachments/assets/17af6446-8566-4922-b77f-73e316cf5cf4" width="28.1%" />
</p>

These images are transformed through a pipeline of custom effects to generate two final stylized outputs. 

## 🧾 Pseudo Code:

![White Minimalist Modern Recruitment Process Flowchart (1)](https://github.com/user-attachments/assets/e52490c0-af4e-4a65-9ac7-772da5ae3065)

The pseudocode diagram provides an overview of the full pipeline: from loading input images, generating warped line masks, applying cyberpunk-style color distortion, motion blur, k-means clustering for object segmentation, and finally, applying an edge-based glow effect.  

## ⚙️ Implementation Steps:
### 🔶 3D Warped Line Mask Application  

A sinusoidal wave pattern is generated and overlaid on the image to simulate a 3D warped effect. The mask consists of alternating gray and black stripes, distorted by sine functions in both X and Y directions. 

<p align="center">
  <img src="https://github.com/user-attachments/assets/ace8044a-1b4a-4bd3-9a38-6214ac852008" width="45%" />
  <img src="https://github.com/user-attachments/assets/063e7b50-8a72-4308-81fc-3752b77302a1" width="28%" />
</p>

### 🔶 3D Effect

Applies RGB channel shifting for a futuristic color offset. A motion blur filter is applied to the right half of the image. Final image is created by blending the blurred and non-blurred halves using a horizontal mask. the line
masks are applied to the input images. The red and blue channels are shifted by 200 pixels. The red channel is shifted to the right and the blue channel is shifted to the left side. The green channel is not shifted.

<p align="center">
  <img src="https://github.com/user-attachments/assets/9f5a4b4b-be13-47ed-b38e-7eaeee38c85f" width="45%" />
  <img src="https://github.com/user-attachments/assets/f039d7a3-8a95-46b9-84cb-bbc90db4f7b4" width="28%" />
</p>


### 🔶 K-Means Clustering

Initially, the function converts the input image from the RGB color space to HSV color
space using the rgb2hsv function. In the HSV image, each channel is combined into
a single matrix to start clustering. The function performs K-means clustering with the
number of clusters (numClusters = 5) using the kmeans function. This function
assigns each pixel to one of the clusters based on color similarity. After clustering,
the cluster labels are reshaped back to the original image dimensions and assigned
to a variable. Then each cluster is turned into a mask.
After obtaining 5 different clusters, I examined the results. The cluster that gives the
background in both images has a non-zero pixel as its first pixel. So, to only find the
cluster that corresponds to the background image the first pixel of each cluster is
checked. Upon finding the background cluster, the masked cluster image is assigned
to cluster_1 and cluster_2 variables. They are displayed below.

<p align="center">
  <img src="https://github.com/user-attachments/assets/521e6525-a007-4475-9f9d-94e686341224" width="45%" />
  <img src="https://github.com/user-attachments/assets/61a95e56-f1eb-4552-b2b5-f66bff4c33a3" width="28%" />
</p>

### 🔶 Borderline Edge Enhancement

The cluster masks cluster_1 and cluster_2 are converted to grayscale, then using the
imbinarize function they are converted to binary format. To get rid of the isolated
regions in the masks I performed a thresholding using the bwareaopen function with
15000 threshold size. The results after the isolated regions thresholding is given in
the pictures below.

<p align="center">
  <img src="https://github.com/user-attachments/assets/1c28aa22-fffc-4c91-94f6-3929bc6f755c" width="45%" />
  <img src="https://github.com/user-attachments/assets/697dd1d1-4b82-4882-8323-fac809b85b53" width="28%" />
</p>


Then I performed morphological dilation structuring element strel('disk', 15) with
imdilate function to expand the mask areas, so I can create edges for the images.

<p align="center">
  <img src="https://github.com/user-attachments/assets/2b067809-cebe-4bae-8ba2-c9de08e8cb91" width="45%" />
  <img src="https://github.com/user-attachments/assets/bdcda54f-1d03-4ea5-966b-22b7312ee8d6" width="28%" />
</p>

Then I performed a morphological erosion using the imerode function with another
structuring element strel('disk', 20) on the dilated image and extracted edges by
combining dilated masks with their eroded versions, resulting in isolated boundary
regions. The obtained edges can be seen below:

<p align="center">
  <img src="https://github.com/user-attachments/assets/6aab9759-1664-44c9-a05e-30acd7a7ae71" width="45%" />
  <img src="https://github.com/user-attachments/assets/978b836a-22c6-48d8-8005-b30a360f6ee6" width="28%" />
</p>

## 🎨 Outputs:

Finally, I colored the edges cyan and to get a neon glow effect I multiplied the colored
edges by 8 and blurred using a Gaussian blur. Then I overlayed 5 shifted and faded
layers of the blurred edges onto the original images to create a moving effect. Each
layer was shifted by 100 pixels diagonally and its intensity was reduced by a factor of
0.3. This was the last step of cyberpunk filter creation. The final results of the filter
are displayed below.

<p align="center">
  <img src="https://github.com/user-attachments/assets/76b7bfe8-1d7c-442c-bcf3-a488655a6e33" width="45%" />
  <img src="https://github.com/user-attachments/assets/6fc9d7d2-e85c-485d-9cbc-c9d2140857ab" width="28.5%" />
</p>

## 💻 Technologies Used:
🔹MATLAB: Image processing and visualization  
🔹Image Processing Toolbox  
🔹K-Means Clustering  
🔹Gaussian Blur, Dilation, Erosion, RGB Manipulation  
🔹Custom Sine Wave Warping and Blending Techniques  

