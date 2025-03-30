clc; clear; close all;

% ================================
% 3D Warped Line Mask Application
% ================================

% Input image filenames
inputImages = {'input (1).jpg', 'input (2).jpg'};

% Parameters
lineThickness = 25;    % Line thickness
lightGray = 200;       % Light gray intensity
black = 255;           % Black intensity
freq_x = 8;            % Frequency of wave in x direction
freq_y = 3;            % Frequency of wave in y direction
amp_x = 20;            % Amplitude of distortion in x direction
amp_y = 15;            % Amplitude of distortion in y direction
phase_shift = pi/4;    % Phase shift
amplitude = 20;        % Amplitude of sinusoidal wave
frequency = 0.02;      % Frequency of sinusoidal wave

% Initialize variables for the warped line masks
warpedLineMask_1 = [];
warpedLineMask_2 = [];

% Loop through both input images
for i = 1:length(inputImages)
    % Read the input image
    inputImage = imread(inputImages{i});
    [rows, cols, ~] = size(inputImage);

    % Step 1: Create a warped line mask
    lineMask = zeros(rows, cols, 'uint8');
    for y = 1:rows
        % Calculate the horizontal wave displacement
        warpShift = round(amplitude * sin(2 * pi * frequency * y));
        % Determine the line color (light gray or black)
        if mod(floor(y / lineThickness), 2) == 0
            lineColor = lightGray;
        else
            lineColor = black;
        end
        % Apply the warped lines
        for x = 1:cols
            warpedX = mod(x + warpShift, cols); % Ensure it wraps around
            lineMask(y, warpedX + 1) = lineColor;
        end
    end

    % Step 2: Apply 3D wave warping to the mask
    lineMask = im2double(lineMask);
    [X, Y] = meshgrid(1:cols, 1:rows);
    X_new = X + amp_x * sin(2 * pi * freq_x * Y / rows + phase_shift);
    Y_new = Y + amp_y * sin(2 * pi * freq_y * X / cols);
    X_new = min(max(X_new, 1), cols);
    Y_new = min(max(Y_new, 1), rows);
    warpedLineMask = interp2(X, Y, lineMask, X_new, Y_new, 'cubic', 0);
    warpedLineMask = uint8(warpedLineMask * 255);

    % Assign the warped mask to the corresponding variable
    if i == 1
        warpedLineMask_1 = warpedLineMask;
    elseif i == 2
        warpedLineMask_2 = warpedLineMask;
    end
end

disp('Warped lines achieved.');


% ================================
% Cyberpunk Effect
% ================================

% Define motion blur kernel size and angle
kernel_size = 300;  % Motion blur intensity
motion_angle = 25;  % Motion blur direction

% Assign line masks
lineMasks = {warpedLineMask_1, warpedLineMask_2};

% Initialize output variables
finalimage_1 = [];
finalimage_2 = [];

% Loop over both input images
for i = 1:length(inputImages)
    % Read the input image
    img = imread(inputImages{i});

    % Get the respective line mask
    lineMask = lineMasks{i};
    
    % Ensure the mask matches the size of the input image
    lineMaskResized = imresize(im2double(repmat(lineMask, [1, 1, 3])), [size(img, 1), size(img, 2)]);
    
    % Apply the line mask to the input image
    masked_img = im2double(img) .* lineMaskResized;

    % Create shifted color effects
    redShifted = circshift(masked_img, [0, 300, 0]);  % Shift red channel
    greenShifted = circshift(masked_img, [0, 0, 0]);  % No shift for green
    blueShifted = circshift(masked_img, [0, -300, 0]); % Shift blue channel

    img_shifted = cat(3, redShifted(:,:,1), greenShifted(:,:,2), blueShifted(:,:,3)); % Merge RGB channels

    % Generate a motion blur filter
    motion_blur_filter = fspecial('motion', kernel_size, motion_angle);

    % Apply motion blur to each channel (RGB)
    blurred_img = img_shifted; % Initialize output image
    for c = 1:3
        blurred_img(:,:,c) = imfilter(img_shifted(:,:,c), motion_blur_filter, 'replicate');
    end

    % Create a mask for blending (0 on the left, 1 on the right)
    [rows, cols, ~] = size(img_shifted);
    blend_mask = repmat(linspace(0, 1, cols), rows, 1);

    % Blend sharp left half with blurred right half
    final_img = img_shifted .* (1 - blend_mask) + blurred_img .* blend_mask;

    % Assign the result to the respective variable
    if i == 1
        finalimage_1 = final_img;
    elseif i == 2
        finalimage_2 = final_img;
    end
end

disp('Right-side motion blur processing complete. Results stored in finalimage_1 and finalimage_2.');


% ================================
% K-Means Clustering
% ================================

% Function to process clusters and return selected cluster images
function finalClusterImage = processImage(img)
    % Convert the image to HSV color space
    img_hsv = rgb2hsv(img);

    % Extract HSV channels
    H = img_hsv(:, :, 1); % Hue
    S = img_hsv(:, :, 2); % Saturation
    V = img_hsv(:, :, 3); % Value

    % Get the image dimensions
    [height, width, ~] = size(img_hsv);

    % Normalize the features (HSV values and distance)
    featureH = H(:); % Hue
    featureS = S(:); % Saturation
    featureV = V(:); % Value

    % Combine features into a single matrix for clustering
    features = [featureH, featureS, featureV];

    % Perform K-means clustering
    numClusters = 5;
    [clusterLabels, ~] = kmeans(features, numClusters);

    % Reshape cluster labels to image dimensions
    clusteredImage = reshape(clusterLabels, [height, width]);

    % Initialize variable for the selected cluster image
    finalClusterImage = []; % Initialize variable for storing the cluster image

    for i = 1:numClusters
        % Create a binary mask for the current cluster
        clusterMask = clusteredImage == i; 
        
        % Apply the mask to the input image to extract the cluster
        clusterImage = bsxfun(@times, img, uint8(clusterMask)); 
    
        % Check if the first pixel in the cluster is non-black
        if any(clusterImage(1, 1, :) > 0)
            % Assign the cluster image to the final variable
            finalClusterImage = clusterImage;
    
            % Display a message indicating the selected cluster
            disp(['Selected cluster with non-black first pixel: Cluster ', num2str(i)]);
            
            % Exit the loop after finding the first valid cluster
            break;
        end
    end
end

% Process both images and get the final cluster images
img1 = imread(inputImages{1});
img2 = imread(inputImages{2});
cluster_1 = processImage(img1);
cluster_2 = processImage(img2);

disp('Final images created and displayed.');

% ===============================
% Borderline
% ===============================

% Load input images
img1 = finalimage_1; % Replace with your image file
img2 = finalimage_2; % Replace with your image file

% Load cluster images (masks)
img1_cluster = cluster_1; % Replace with your cluster file
img2_cluster = cluster_2; % Replace with your cluster file

% Convert cluster images to grayscale if needed
if size(img1_cluster, 3) == 3
    img1_cluster = rgb2gray(img1_cluster);
end
if size(img2_cluster, 3) == 3
    img2_cluster = rgb2gray(img2_cluster);
end

% Create binary masks
mask1 = imbinarize(img1_cluster); % Binary mask from input1_cluster_1
mask2 = imbinarize(img2_cluster); % Binary mask from input2_cluster_5

% Remove small isolated regions using a size threshold
sizeThreshold = 15000; % Minimum size of regions to keep (adjust as needed)
mask1 = bwareaopen(mask1, sizeThreshold);
mask2 = bwareaopen(mask2, sizeThreshold);

% Perform dilation on the masks
se = strel('disk', 15); % Structuring element (disk-shaped with radius 15)
mask1_dilated = imdilate(mask1, se); % Dilate mask1
mask2_dilated = imdilate(mask2, se); % Dilate mask2

% Extract edges of the masks
seEdge = strel('disk', 30); % Structuring element for edge thickness
mask1_edge = mask1_dilated & ~imerode(mask1_dilated, seEdge); % Extract thick edges from mask1
mask2_edge = mask2_dilated & ~imerode(mask2_dilated, seEdge); % Extract thick edges from mask2

% Create RGB images to paint edges cyan and add red
edge1_effect = zeros(size(img1), 'uint8'); % Initialize RGB canvas for edges of mask1
edge2_effect = zeros(size(img2), 'uint8'); % Initialize RGB canvas for edges of mask2

% Paint edges with cyan (cyan = [0, 255, 255]) and red ([255, 0, 0])
edge1_effect(:, :, 1) = uint8(mask1_edge) * 0; % Red channel
edge1_effect(:, :, 2) = uint8(mask1_edge) * 100; % Green channel
edge1_effect(:, :, 3) = uint8(mask1_edge) * 255; % Blue channel

edge2_effect(:, :, 1) = uint8(mask2_edge) * 0; % Red channel
edge2_effect(:, :, 2) = uint8(mask2_edge) * 100; % Green channel
edge2_effect(:, :, 3) = uint8(mask2_edge) * 255; % Blue channel

% Apply blur to the edges
edge1_effect = edge1_effect * 8;
edge2_effect = edge2_effect * 8;

blurKernel = fspecial('gaussian', [15, 15], 100); % Gaussian blur kernel
edge1_effect_blurred = imfilter(im2double(edge1_effect), blurKernel, 'symmetric'); % Convert to double for consistency
edge2_effect_blurred = imfilter(im2double(edge2_effect), blurKernel, 'symmetric'); % Convert to double for consistency

% Initialize output images with original
img1_with_effect = im2double(img1); % Convert to double for blending
img2_with_effect = im2double(img2);

% Add multiple shifted layers for "largening effect"
numLayers = 5; % Number of additional edge layers
shiftStep = 100; % Pixels to shift for each layer
intensityFactor = 0.5; % Intensity reduction per layer

for i = 1:numLayers
    % Shift the blurred edges
    shifted_edge1 = circshift(edge1_effect_blurred, [i * shiftStep, -i * shiftStep]);
    shifted_edge2 = circshift(edge2_effect_blurred, [i * shiftStep, -i * shiftStep]);
    
    % Reduce intensity for each layer
    shifted_edge1 = shifted_edge1 * (intensityFactor^(i-1));
    shifted_edge2 = shifted_edge2 * (intensityFactor^(i-1));
    
    % Blend with the images
    img1_with_effect = img1_with_effect + shifted_edge1; % Element-wise addition
    img2_with_effect = img2_with_effect + shifted_edge2; % Element-wise addition
end

% Clamp output to [0, 1] to avoid intensity overflow
img1_with_effect = min(img1_with_effect, 1);
img2_with_effect = min(img2_with_effect, 1);


% Display the resized image
figure;
imshow(img1_with_effect);
title('Final Image 1');

figure;
imshow(img2_with_effect);
title('Final Image 2');

% Save the line mask for reuse
imwrite(img1_with_effect, 'Esra_input (1).png');
imwrite(img2_with_effect, 'Esra_input (2).png');