# Deep Face Mask Detector
A deep learning model created to detect face masks during COVID achieving 98.5% accuracy.

The deep model was trained on a labelled dataset of ~1.7k face mask wearing images and ~1.9k non-face mask images, i.e. just a persons face.
The images were also horizontally flipped and subject to width and height shifts to increase the dataset.
The images were then pre-processed to a 3 channel 128x128 array prior to input into the neural network.

The trained model was then used in combination with a Haar cascade classifier to first detect and take an image of a face to then pass to the model.
The model could then make a prediciton of whether the person in the frame was wearing a mask or not as displayed below.

# Requirements
TensorFlow

# Try It Out
1. Clone repo
2. Run mask_detector.py
3. See results!
