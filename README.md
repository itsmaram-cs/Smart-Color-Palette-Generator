# Smart Color Palette Generator

Smart Color Palette Generator is a real-time computer vision project developed using Python, OpenCV, and NumPy.

The application captures a selected region from the webcam and analyzes its dominant colors using image processing and K-Means clustering. It extracts the most representative colors in the scene, calculates their coverage percentage, and displays their RGB, HSV, and HEX values. The generated palette can also be saved as an image for future use.

## Features

- Real-time webcam analysis
- Detect dominant colors
- Display RGB, HSV, and HEX values
- Calculate color coverage percentage
- Save the generated color palette as an image

## Requirements

- Python
- OpenCV
- NumPy
pip install opencv-python numpy

## Run
python main.py

## Controls

| Key | Function |
|------|----------|
| S | Analyze the selected area |
| P | Save the generated palette |
| R | Reset the analysis |
| Q | Exit the application |

## Example

Example output image: [project-output.png](project-output.png)
