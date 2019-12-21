## LFFD: A Light and Fast Face Detector for Edge Devices
[paper link](https://arxiv.org/pdf/1904.10633.pdf)  
## Requirements:  
* pytorch>=1.2  
* opencv-python  
* torchvision>=0.4  
## Label Assignment  
Boxes are in different scale. Anchor points with more than one matched gt box are ignored.  
`green box --> gt box`  
`blue/red point --> positive sample`  
The positive sample sampling of the fifth branch is shown below.  
<p align=center>
<img src="assets/target_test.jpg" height="320" width="320">
</p>  
## Different settings from the paper  
* regression target
* add BN
* more classes  
## Test  
I have trained the model on a safety helmet dataset without any augmentation. It seems to be working well.  
<p align=center>
    <img src="assets/lffd_tb.jpg">
    <p align=center>
        <em>tensorboard</em>
    </p>
</p>
<p align=center>
    <img src="assets/helmet1.jpg" height=70% width=70%>  
    <img src="assets/helmet2.jpg" height=70% width=70%>  
    <p align=center>
        <em>test_result</em>
    </p>
</p>
