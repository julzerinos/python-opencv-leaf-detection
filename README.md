# The Komatsuna Dataset with OpenCV and python

## TL;DR

Python-based OpenCV program for detecting leaves and creating segmentation masks based on images in the [Komatsuna dataset](http://limu.ait.kyushu-u.ac.jp/~agri/komatsuna/).

## More information

View the project report on [overleaf](https://www.overleaf.com/read/thzjxjdqnczs).

## The program

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
