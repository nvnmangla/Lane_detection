# Histogram Equalization and Lane Detection
This Repository consists of two Projects 
## Histogram Equalization
#### Introduction

Histogram is a graphical representation of the intensity distribution of
an image. In simple terms, it represents the number of pixels for each
intensity value considered. Sometimes, images are taken in very dark or
poor lightning conditions. An example of poor exposure image is given
below in Fig 1

![Poor Exposure Image](https://github.com/nvnmangla/Lane_detection/blob/9961ec55cd0a6bf3d274ec43a5f537895c4e526b/adaptive_hist_data/0000000000.png)*Poor Exposure Image*

Histogram equalisation helps to enhance the intensity level of pixels with poor exposure.It also used to enhance the contrast in the image.
The Histogram for above image is shown in Fig 2 and 3

![Histogram Image](https://github.com/nvnmangla/Lane_detection/blob/729b4dba8c5547eb3486fa24cbde8e61903ad2a1/histogram_results/result_hist.png)*Image after Histogram equalization*



![Adaptive Histogram Image](https://github.com/nvnmangla/Lane_detection/blob/4c706aea1bdcc41b9dd47496411d2abc8a0d73b7/histogram_results/adaptive.png)*Image after Adaptive Histogram equalization*

## Lane Detection
### Bird Eye View
A bird’s-eye view is an elevated view of an object from above, with a perspective as though the observer were a bird. For this I used a constant size "Parking frame" as shown in the Fig. 4 as target area.

![Birds Eye View](https://github.com/nvnmangla/Lane_detection/blob/68e2af93689eaaecfb461f7350eb54c6883f94ed/lane_results/frame.png)*Building Frame for Birds Eye View*

Then Homography was performed using corners of the above frame and a portrait rectangle.
Birds Eye             |   Sliding Window
<!-- :-------------------------:|:-------------------------: -->

![Birds Eye](https://github.com/nvnmangla/Lane_detection/blob/68e2af93689eaaecfb461f7350eb54c6883f94ed/lane_results/out.png) | 
![Sliding Window](https://github.com/nvnmangla/Lane_detection/blob/68e2af93689eaaecfb461f7350eb54c6883f94ed/lane_results/out.png)



#### Running Instructions
```
git clone git@github.com:nvnmangla/Lane_detection.git
cd Lane_detection/
```
To run Histogram
```
python3 solution.py --isHist=True

```
Else
```
python3 solution.py
```