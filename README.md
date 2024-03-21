# Image-processing-based-on-OpenCV

## Project Remembers
ZiHao Zhou:  School of Information Engineering, South China University of Technology, Guangzhou, Guangdong, China 
             
JunYe Chen, Chen Yang, ZiQiang Qin

## Research Focus
This code repository contains image processing related codes based on OpenCV (C++)

### Direction 1: underwater image enhancement based on traditional digital image processing algorithm
We focus on image enhancement and reconstruction technology based on OpenCV, and will finally apply our algorithm to underwater image enhancement and reconstruction. We will first study the traditional image enhancement methods, such as Gamma transform, histogram equalization, white balance, dark channel priori, image fusion based on Laplacian pyramid, etc. 

#### Traditional algorithms implemented
1. Laplacian pyramid image fusion algorithm  
   Advantages: Some othor algorithm fuses advantage maps(such as illuminance diagram, chromaticity diagram and saliency diagram) based on laplacian pyramid.
   Limitations: The design of the mask is flexible and changeable. It requires some skills.

2. Dark Channel Prior  
   Adavantages: It made a great breakthrough in the task of dealing with fog pictures on land.
   Limitations: There are still some changes to be made when the land fog imaging model is moved to underwater.

3. Multi-scale detail enhancement  
   Advantages: This algorithm is quite simple but has achieved good result. It filters the original image with gaussian kernal under different scale.Using the original image to subtract from these images blurred by different scales of Gauss, so that we can get the detail of origin image.
   Limitations:There will be some distortion in the image
   
4. MSR and MSRCR  
   Generally, the result of MSR is just so so, and it will bring Color distortion. However, when the Color recovery factor is introduced(MSRCR), the effect of algorithm has become greater
   Limitations: The parameters : dynamic, is not so flexible.(the smaller the dynamic is, the stronger image contrast)

### Direction 2: underwater image enhancement based on deep learning 
At the same time, we will also study the underwater image processing based on deep learning, such as CNN image reconstruction (SRCNN, FSRCNN, SRGAN, etc.), but the difficulty of using deep learning is how to obtain pairs of data and labels.It's mainly because the underwater high-definition images are difficult to obtain. However, the advantages of deep learning is when the networks finish the training, it will be quite easy for us to use. 

![](https://github.com/ZZH0/Image-processing-based-on-OpenCV/blob/master/Image%20for%20readme/display_dl.png)

How to get higher quality underwater image in the improved algorithm is our ultimate research goal.




