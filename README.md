This is the official implementation of paper *TEVAD: Improved video anomaly detection with captions*. 

The paper is accepted by O-DRUM workshop @ CVPR 2023.

# Preparations
## File structure
```
.
|-- README.md
|-- ckpt  # save checkpoints
|   |-- my_best
|   |   |-- ped2-both-text_agg-add-1-1-extra_loss-755-4869-i3d.pkl
|   |   |-- shanghai_v2-both-text_agg-add-1-1-extra_loss-595-i3d-best.pkl
|   |   |-- ucf-both-text_agg-concat-0.0001-extra_loss-620-4869-.pkl
|   |   `-- violence-both-text_agg-add-1-1-extra_loss-445-4869-BEST.pkl
|-- config.py
|-- dataset.py
|-- list  # ground truth and list files for training and testing
|    |-- gt-ped2.npy
|    |-- gt-sh2.npy
|    |-- gt-ucf.npy
|    |-- gt-violence.npy
|    |-- ped2-i3d-test.list
|    |-- ped2-i3d.list
|    |-- shanghai-i3d-test-10crop.list
|    |-- shanghai-i3d-train-10crop.list
|    |-- ucf-i3d-test.list
|    |-- ucf-i3d.list
|    |-- violence-i3d-test.list
|    `-- violence-i3d.list
|-- main.py  # main file, train and test
|-- main_test.py 
|-- model.py  
|-- option.py  
|-- requirement.txt
|-- results
|-- save  # save features
|   |-- Crime
|   |   |-- UCF_ten_crop_i3d_v1  # I3D features
|   |   `-- sent_emb_n  # sentence embedding features
|   |-- Shanghai
|   |   |-- SH_ten_crop_i3d_v2
|   |   `-- sent_emb_n
|   |-- UCSDped2
|   |   |-- ped2_ten_crop_i3d
|   |   `-- sent_emb_n
|   `-- Violence
|       |-- Violence_five_crop_i3d_v1
|       `-- sent_emb_n
|-- test_10crop.py
|-- train.py
`-- utils.py
```

**To run the code, take UCF-Crime dataset as an example.**

## Text features
Download from [LINK](https://1drv.ms/u/s!AlbDzA9D8VkhoO8dcvJNaAMkk5bbgA?e=Eh2LCB) (the file structure is the same as the tree map shown above) and put under `/save/Crime/snet_emb_n/` folder or generate the text features using this [repo](https://github.com/coranholmes/SwinBERT)

## Visual features
1. You can download from [here](https://1drv.ms/u/s!AlbDzA9D8VkhoO8dcvJNaAMkk5bbgA?e=Eh2LCB) or generate the visual features using this [repo](https://github.com/GowthamGottimukkala/I3D_Feature_Extraction_resnet).
2. For UCF-Crime dataset, put the generated/downloaded features under `./save/Crime/UCF_ten_crop_i3d_v1` folder. Other datasets follow the same structure.
3. For UCF-Crime dataset, change the path of visual features in `./list/ucf-i3d-test.list` and `list/ucf-i3d.list`. Other datasets follow the same structure.

## Install requirements
Run `pip install -r requirement.txt` to install the requirements.

## Run visdom
**!!!VERY IMPORTANT!!!**

Open a separate terminal and run `visdom` after installing the requirements before running the following commands.

# Training + Testing
Meanings of the arguments can be seen in `option.py`. To train the best model presented in the paper, use the following settings:

UCF-Crime dataset
```bash
python main.py --dataset ucf --feature-group both --fusion concat --aggregate_text --extra_loss
```
ShanghaiTech dataset
```bash
python main.py --dataset shanghai_v2 --feature-group both --fusion add --aggregate_text --extra_loss
```
XD-Violence dataset
```bash
python main.py --dataset violence --feature-group both --fusion add --aggregate_text --extra_loss --feature-size 1024
```
UCSD-Ped2 dataset
```bash
python main.py --dataset ped2 --feature-group both --fusion add --aggregate_text --max-epoch 10000 --extra_loss --batch-size 2
```

# Testing only (optional)
UCF-Crime dataset
```bash
python main_test.py --dataset ucf --pretrained-ckpt ./ckpt/my_best/ucf-both-text_agg-concat-0.0001-extra_loss-620-4869-.pkl --feature-group both --fusion concat --aggregate_text --save_test_results
```
ShanghaiTech dataset
```bash
python main_test.py --dataset shanghai_v2 --feature-group both --fusion add --aggregate_text --pretrained-ckpt ./ckpt/my_best/shanghai_v2-both-text_agg-add-1-1-extra_loss-595-i3d-best.pkl --save_test_results
```
XD-Violence dataset
```bash
python main_test.py --dataset violence --feature-group both --fusion add --aggregate_text --feature-size 1024 --pretrained-ckpt ./ckpt/my_best/violence-both-text_agg-add-1-1-extra_loss-445-4869-BEST.pkl --save_test_results
```
UCSDped2 dataset
```bash
python main_test.py --dataset ped2 --feature-group both --fusion add --aggregate_text --pretrained-ckpt ./ckpt/my_best/ped2-both-text_agg-add-1-1-extra_loss-755-4869-i3d.pkl --save_test_results
```

# Citation
If you find this code useful for your research, please cite our paper:
```
@inproceedings{chen2023TEVAD,
  title={TEVAD: Improved video anomaly detection with captions},
  author={Chen, Weiling and Ma, Keng Teck and Yew, Zi Jian and Hur, Minhoe and Khoo, David},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}
```

# Acknowledgements
This code is based on [RTFM](https://github.com/tianyu0207/RTFM/). We thank the authors for their great work.

