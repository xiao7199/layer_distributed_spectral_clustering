## Layer Distributed Spectral Clustering
This repo includes the official implementation of the paper [Deciphering 'What' and 'Where' Visual Pathways from Spectral Clustering of Layer-Distributed Neural Representations](https://sites.google.com/uchicago.edu/distributed-spectral-cluster/home) CVPR 2024 (Highlight)  
Authors: [Xiao Zhang*](https://xiao7199.github.io/), [David Yunis*](https://dyunis.github.io/),  [Michael Maire](https://people.cs.uchicago.edu/~mmaire/)


### Environment
This code is developed with the following major packages
```
pytorch=2.2.1
diffusers=0.14.0
transformers=4.29.2
```


 ### Running Code
We provide the code to extract low-dimension dense features from deep models (Diffusion Model as default) for a single image and multiple images. Before running the code, please replace the `HUGGIN_TOKEN` in `srun.sh` with your [hugging face token](https://huggingface.co/docs/hub/en/security-tokens) to access the pre-trained diffusion model
 #### Single Image Analysis
 To run the spectral clustering for a single image, please use the following command
```
cd single_img
bash srun.sh
```
It will load the image from `img_path` and visualize the rendered PCA, instance segmentation, and image segmentation with K-Means clustering (with K automatically decided by [silhouette scores](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)) as:
![single_img](single_output.png)

Eigenvectors are visualized as:
![eigenvector](eigenvector.png)
 #### Multiple Images Analysis
To run the spectral clustering for multi-image analysis, run: `bash srun.sh`. It will compute eigenvectors for 100 images saved under `./dataset` (the first 100 images of the COCO2017 validation split) of VV-Graph ('what' visual pathway). At the end, the script will render 15 leading eigenvectors.
 ![VV_result](VV_result.png)
For experiment with QK-Graph ('where' visual pathway), please run `bash srun.sh` by commenting out the `--vv_graph` from `srun.sh`. The 15 leading eigenvectors are visualized as:
 ![QK_result](QK_result.png)

### Citation
If you find our paper or code useful, please cite our work:
```
@inproceedings{zhang2024deciphering,
  title={Deciphering'What'and'Where'Visual Pathways from Spectral Clustering of Layer-Distributed Neural Representations},
  author={Zhang, Xiao and Yunis, David and Maire, Michael},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4165--4175},
  year={2024}
}
```
