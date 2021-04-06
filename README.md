# The Komatsuna Dataset with OpenCV and python

## In a nutshell

Python-based OpenCV program for detecting leaves and creating segmentation masks based on images in the [Komatsuna dataset](http://limu.ait.kyushu-u.ac.jp/~agri/komatsuna/).

### Using the script

Set up the dataset from the above link, install the pip requirements and run `python3 PlantDetector.py`.

### Controls

Key | Function
--- | ---
m   | next image
n   | prev image
s   | save selected image in the selected mode
z   | save all images in selected mode
esc | exit the program
d   | dynamically calculate dice
f   | show dice data based on saved images
1-5 | select the respective mode

### Modes

Mode | Window "Altered" | Window "Original"
--- | --- | ---
0 | original (fallback)   | original
1 | HSV filter range      | original
2 | bare watershed masks  | labels
3 | watershed masks w/ bg | original
4 | sequential bg sub     | original
5 | seq bg sub w/ watersh | original

## The project report

View the project report below or on [Overleaf](https://www.overleaf.com/read/thzjxjdqnczs).

![01](https://raw.githubusercontent.com/julzerinos/python-opencv-leaf-detection/assets/python_opencv_plant_detection-01.jpg)
![02](https://raw.githubusercontent.com/julzerinos/python-opencv-leaf-detection/assets/python_opencv_plant_detection-02.jpg)
![03](https://raw.githubusercontent.com/julzerinos/python-opencv-leaf-detection/assets/python_opencv_plant_detection-03.jpg)
![04](https://raw.githubusercontent.com/julzerinos/python-opencv-leaf-detection/assets/python_opencv_plant_detection-04.jpg)
![05](https://raw.githubusercontent.com/julzerinos/python-opencv-leaf-detection/assets/python_opencv_plant_detection-05.jpg)
![06](https://raw.githubusercontent.com/julzerinos/python-opencv-leaf-detection/assets/python_opencv_plant_detection-06.jpg)
![07](https://raw.githubusercontent.com/julzerinos/python-opencv-leaf-detection/assets/python_opencv_plant_detection-07.jpg)
![08](https://raw.githubusercontent.com/julzerinos/python-opencv-leaf-detection/assets/python_opencv_plant_detection-08.jpg)
![09](https://raw.githubusercontent.com/julzerinos/python-opencv-leaf-detection/assets/python_opencv_plant_detection-09.jpg)
![10](https://raw.githubusercontent.com/julzerinos/python-opencv-leaf-detection/assets/python_opencv_plant_detection-10.jpg)
![11](https://raw.githubusercontent.com/julzerinos/python-opencv-leaf-detection/assets/python_opencv_plant_detection-11.jpg)
![12](https://raw.githubusercontent.com/julzerinos/python-opencv-leaf-detection/assets/python_opencv_plant_detection-12.jpg)
![13](https://raw.githubusercontent.com/julzerinos/python-opencv-leaf-detection/assets/python_opencv_plant_detection-13.jpg)
![14](https://raw.githubusercontent.com/julzerinos/python-opencv-leaf-detection/assets/python_opencv_plant_detection-14.jpg)
![15](https://raw.githubusercontent.com/julzerinos/python-opencv-leaf-detection/assets/python_opencv_plant_detection-15.jpg)
![16](https://raw.githubusercontent.com/julzerinos/python-opencv-leaf-detection/assets/python_opencv_plant_detection-16.jpg)
![17](https://raw.githubusercontent.com/julzerinos/python-opencv-leaf-detection/assets/python_opencv_plant_detection-17.jpg)
