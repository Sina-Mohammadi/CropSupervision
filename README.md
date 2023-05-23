# Improvement in crop mapping from satellite image time series by effectively supervising deep neural networks
The official implementation code for our paper "Improvement in crop mapping from satellite image time series by effectively supervising deep neural networks".


Our paper has been accepted to ISPRS Journal of Photogrammetry and Remote Sensing and is publicly available at: [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0924271623000679
).

## Our Method
To learn more discriminative feature representations to detect crop types and reveal the importance of proper supervision of deep neural networks in improving performance, we propose to supervise intermediate layers of a designed 3D Fully Convolutional Neural Network (FCN) by employing two middle supervision methods: Cross-entropy loss Middle Supervision (CE-MidS) and a novel middle supervision method, namely Supervised Contrastive loss Middle Supervision (SupCon-MidS). SupCon-MidS pulls together features belonging to the same class in embedding space, while pushing apart features from different classes. We demonstrate that SupCon-MidS enhances feature discrimination and clustering throughout the network, thereby improving the network performance. In addition, we employ two output supervision methods, namely F1 loss and Intersection Over Union (IOU) loss, which outperfrom the widely used cross-entropy loss.

<p align="center"><img src="https://github.com/Sina-Mohammadi/CropSupervision/blob/main/fig/framework.jpg" width="750" height="650"></p>


### Requirements
- [numpy 1.23.5](https://numpy.org/)
- [tensorflow 2.9.2](https://www.tensorflow.org/)
- [tensorflow-addons 0.16.1](https://www.tensorflow.org/addons)
- [tables 3.6.1](https://www.pytables.org/)

### Usage

1- Download the preprocessed training data for the four sites from google drive using the following links: [Site_A](https://drive.google.com/file/d/1fhoFewOoLPSWWmX5dOeme2rlZJXyyC7A/view?usp=sharing) , [Site_B](https://drive.google.com/file/d/1fHerhZHxV0w1cTU6PO37Q2E_RITV6Zwc/view?usp=sharing)  , [Site_C](https://drive.google.com/file/d/1Cc71iW4te0pMjAmMO2um2iSoQUOtrzs6/view?usp=sharing)  , [Site_D](https://drive.google.com/file/d/14WStPwEAuea9X-WnjHIq51L8iyc41Bfu/view?usp=sharing) - Then put them in the data folder.

2- Train the model with the loss function(s) of your choice using the follwing code:

```
python main.py --out_supervision 'IOU' --mid_supervision 'SupCon' --site 'A' --validation_fold 5 --data_dir 'data' --save_dir 'save'
```

- *out_supervision*: You can select either of 'IOU', 'F1', or 'Cross-entropy'.

- *mid_supervision*: You can select either of 'SupCon','Cross-entropy','off'.

- *site*: You can select either of 'A','B','C','D'.

- *validation_fold*: An integer between 1 and 5 used to select the validation fold. Train the model using all five data splits to obtain five sets of weights for the model (obtained by perfroming early stopping). 

- *save_dir*: Set the path for saving the weights.

- *data_dir*: Path where the data of the four sites are located.

3- Take the average of the softmax outputs of the five obtained weights for the model to obtain the final prediction probabilities. The test data will be uploaded soon. 

## Citation
```
@article{mohammadi2023improvement,
  title={Improvement in crop mapping from satellite image time series by effectively supervising deep neural networks},
  author={Mohammadi, Sina and Belgiu, Mariana and Stein, Alfred},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={198},
  pages={272--283},
  year={2023},
  publisher={Elsevier}
}
```
