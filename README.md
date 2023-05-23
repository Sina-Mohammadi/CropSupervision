# Improvement in crop mapping from satellite image time series by effectively supervising deep neural networks
Implementation code for our paper "Improvement in crop mapping from satellite image time series by effectively supervising deep neural networks"


Our paper has been accepted to ISPRS Journal of Photogrammetry and Remote Sensing and is publicly available at: [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0924271623000679
).

### Requirements
- [numpy 1.18.5](https://numpy.org/)
- [tensorflow 2.3.1](https://www.tensorflow.org/)
- [tables 3.6.1](https://www.pytables.org/)


If you want to train the model using the first four folds, first download the preprocessed data from [GoogleDrive](https://drive.google.com/file/d/1eql-2OsG9mr8fOUi3SMi19HELzzVbbCj/view?usp=sharing) and put it in data folder and then run:

```
python train.py --data_dir 'data' --save_dir 'save' --loss_function 'IOU' --validation_fold 5
```

In addition to *data_dir*, *save_dir*, loss_function, and validation_fold, you can set these training configurations: *batch_size, learning_rate, epochs.*

## Citation
```
@article{mohammadi20213d,
  title={3D Fully Convolutional Neural Networks with Intersection Over Union Loss for Crop Mapping from Multi-Temporal Satellite Images},
  author={Mohammadi, Sina and Belgiu, Mariana and Stein, Alfred},
  journal={arXiv preprint arXiv:2102.07280},
  year={2021}
}
```
