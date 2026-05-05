# Module5-Part2-bitsom_ba_2511666
# Module5-Part2-bitsom_ba_2511666
Computer Vision Problem Formulation and CNN Prototype
Assignment Summary & Notes
WHAT THIS ASSIGNMENT IS ABOUT
We are given a Synthetic Manufacturing Defect Image Dataset and our job is to:
Identify what type of computer vision problem this is
Explore the dataset
Preprocess images for a neural network
Build a CNN (Convolutional Neural Network) model
Train it and check how well it performs
Explain CNN concepts in our own words
Connect it to a real-world business use case
THE DATASET
Name: Synthetic Manufacturing Defect Image Dataset
Total Images: 480 PNG files
Classes: 4 (equally distributed)
normal (120 images) - clean product surface, no defects
scratch (120 images) - surface with scratch-like marks
dent (120 images) - surface with circular dent-like marks
stain (120 images) - surface with colored stain-like marks
Files provided:
images/ folder with 4 subfolders (normal, scratch, dent, stain)
labels.csv (maps each filename to its class)
data_dictionary.md (describes the dataset)
Balance: Perfectly balanced (120 images per class)
TASK-BY-TASK BREAKDOWN
Task 1: Problem Identification
ANSWER: This is a Multi-class Image Classification problem.
WHY:
Dataset has 4 folders, each folder = one class
Each image belongs to exactly ONE class
No bounding boxes given (so it's NOT object detection)
No pixel-level masks given (so it's NOT segmentation)
We just need to predict WHICH class an image belongs to
The labels.csv confirms: one image -> one label
Task 2: Dataset Exploration
What we found:
4 classes: normal, scratch, dent, stain
120 images in each class (480 total)
Dataset is perfectly balanced (ratio = 1.0)
Images are RGB (3 channels) PNG files
All synthetic (computer-generated) manufacturing surface images
What we did:
Loaded labels.csv and counted images per class
Checked image dimensions (width, height, channels)
Displayed sample images from each class
Plotted a bar chart showing class distribution
Task 3: Image Preprocessing
Steps we applied:
RESIZE: All images resized to 128x128 pixels
Why: CNN needs fixed-size input, images may vary in size
NORMALIZE: Pixel values divided by 255 (scaled to 0-1 range)
Why: Smaller values help the model learn faster and more stable
SPLIT: 80% training, 20% validation
Why: Need separate data to test if model actually learned (not just memorized)
AUGMENT (training only): Random rotation, flip, shift, zoom
Why: Creates variety so model doesn't overfit to exact positions
Tools used: ImageDataGenerator from Keras
Task 4: CNN Model Architecture
Our model structure:
Input: 128x128x3 (RGB image)
Conv2D(32 filters, 3x3) + ReLU + MaxPooling(2x2)
Conv2D(64 filters, 3x3) + ReLU + MaxPooling(2x2)
Conv2D(128 filters, 3x3) + ReLU + MaxPooling(2x2)
Flatten
Dense(128, ReLU)
Dropout(0.5)
Dense(4, Softmax) -> Output
Compiled with:
Optimizer: Adam
Loss: Categorical Crossentropy
Metric: Accuracy
Task 5: Model Training & Evaluation
What we did:
Trained for 25 epochs
Plotted accuracy and loss curves (training vs validation)
Evaluated on validation set
Generated confusion matrix
Printed classification report (precision, recall, f1-score)
Showed sample predictions with confidence percentages
Output files saved:
results/accuracy_loss_curves.png
results/confusion_matrix.png
sample_predictions/prediction_outputs.png
Task 6: CNN Concept Explanation
WHAT IS CONVOLUTION?
A small filter (3x3) slides across the image, multiplying and summing
values at each position. This creates a "feature map" that highlights
patterns like edges, lines, or textures. Different filters detect
different patterns (one for scratches, one for dent edges, etc.)
WHY IS POOLING USED?
MaxPooling takes the maximum value from small blocks (2x2). This:
Reduces data size (faster computation)
Keeps the most important features
Makes model tolerant to small position changes
WHY IS ReLU USED?
ReLU: if positive keep it, if negative make it 0.
Very fast to compute
Helps gradients flow during training (no vanishing gradient)
Creates sparse activations (efficient)
WHY CNNs > REGULAR NETWORKS FOR IMAGES?
Local connectivity: looks at nearby pixels together (spatial awareness)
Weight sharing: same filter used everywhere (fewer parameters)
Translation invariance: detects patterns regardless of position
Hierarchical: builds from edges -> textures -> complex shapes
Task 7: Business Use Case
DOMAIN: Manufacturing - Automated Quality Inspection
Problem: Human inspectors on factory assembly lines are slow, get
tired, and miss subtle defects after long shifts.
Solution: Mount a camera on the conveyor belt, feed images to our
CNN model. It classifies each product as normal/scratch/dent/stain
in milliseconds. Defective items get automatically diverted for repair.
Benefits:
100+ inspections per minute (vs 10-15 by humans)
Consistent 24/7 quality standard
Reduces customer returns by 40-50%
Tracks defect trends to fix root causes in production
Companies like Tesla, Samsung, Foxconn already use this approach
