# LBAM_inpainting
Code will be released soon.
## Prerequisites
- Python 3.6
- Pytorch 1.0
- CPU or NVIDIA GPU + Cuda + Cudnn

### Training
To train the LBAM model:
```bash
python train.py --batchSize numOf_batch_size --dataRoot your_image_path \
--maskRoot your_mask_root --modelsSavePath path_to_save_your_model \
--logPath path_to_save_tensorboard_log --pretrain(optional) pretrained_model
```

### Testing
To test the model:
```bash
python test.py --input input_image --mask your_mask --output output_file_prefix --pretrain your_pretrained_model
```


### If you find this code would be useful
Please cite our paper

```
@article{chaohaoLBAM2019,
	title={Image Inpainting with Learnable Bidirectional Attention Maps},
	author={Xie, Chaohao and Liu, Shaohui and Li, Chao and Cheng, Ming-Ming and Zuo, Wangmeng and Liu, Xiao and Wen, Shilei and Ding, Errui},
	journal={arXiv preprint arXiv:1909.00968},
	year={2019},
}
```
