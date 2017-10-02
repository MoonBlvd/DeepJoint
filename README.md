## Human Structural CNN for joint detection

#### Code guideline
* ```base_vgg.py```:  the base VGG16 (modified) layers.
* ```structural_layers.py```:  the structural CNN layers.
* ```loss```.py: loss functions. 
* ```train.py```: the script of generating, compiling, training and saving the model. 


#### Body structure
* 1 - right shoulder, 2 - right elbow, 3 - right wrist, 
* 4 - left shoulder, 5 - left elbow, 6 - left wrist, 
* 7 - right hip, 8 - right knee, 9 - right ankle, 
* 10 - left hip, 11 - left knee, 12 - left ankle, 
* 13 - head, 14 - neck

---
![123](/home/brianyao/Documents/AI_challenger/readme_fig/human_body_structure.png  "Human body structure")

####Reference
* Chu X, Ouyang W, Li H, Wang X. Structured feature learning for pose estimation. InProceedings of the IEEE Conference on Computer Vision and Pattern Recognition 2016 (pp. 4715-4723).
