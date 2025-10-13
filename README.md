# RadioDiff: An Effective Generative Diffusion Model for Sampling-Free Dynamic Radio Map Construction

---
### Welcome to the RadioDiff family

Base BackBone, Paper Link: [RadioDiff](https://ieeexplore.ieee.org/document/10764739), Code Link: [GitHub](https://github.com/UNIC-Lab/RadioDiff)

PINN Enhanced with Helmholtz Equation, Paper Link: [RadioDiff-$k^2$](https://arxiv.org/pdf/2504.15623)

Efficiency Enhanced RadioDiff, Paper Link: [RadioDiff-Turbo](https://ieeexplore.ieee.org/abstract/document/11152929/)

Indoor RM Construction with Physical Information, Code Link: [GitHub](https://github.com/UNIC-Lab/iRadioDiff)

3D RM with DataSet, Paper Link: [RadioDiff-3D](https://ieeexplore.ieee.org/document/11083758), Code Link: [GitHub](https://github.com/UNIC-Lab/UrbanRadio3D)

Sparse Measurement for RM ISAC, Paper Link: [RadioDiff-Inverse](https://arxiv.org/abs/2504.14298)

Sparse Measurement for NLoS Localization, Paper Link: [RadioDiff-Loc](https://www.arxiv.org/abs/2509.01875)

For more RM information, please visit the repo of [Awesome-Radio-Map-Categorized](https://github.com/UNIC-Lab/Awesome-Radio-Map-Categorized)

---

This is the code for the paper "RadioDiff: An Effective Generative Diffusion Model for Sampling-Free Dynamic Radio Map Construction", IEEE TCCN.

Pre-trained Weight: Please contact me at xcwang_1@stu.xidian.edu.cn

[Paper](https://ieeexplore.ieee.org/document/10764739)

#### 🔥🔥🔥 News

- **2024-7:** This repo is constructed.
- **2024-11:** Our paper has been accepted by IEEE TCCN.
- **2024-12:** The code has been released! 🎉🎉🎉

---

> **Abstract:** Radio map (RM) is a promising technology that can obtain pathloss based on only location, which is significant for 6G network applications to reduce the communication costs for pathloss estimation. However, the construction of RM in traditional is either computationally intensive or depends on costly sampling-based pathloss measurements. Although the neural network (NN)-based method can efficiently construct the RM without sampling, its performance is still suboptimal. This is primarily due to the misalignment between the generative characteristics of the RM construction problem and the discrimination modeling exploited by existing NN-based methods. Thus, to enhance RM construction performance, in this paper, the sampling-free RM construction is modeled as a conditional generative problem, where a denoised diffusion-based method, named RadioDiff, is proposed to achieve high-quality RM construction. In addition, to enhance the diffusion model's capability of extracting features from dynamic environments, an attention U-Net with an adaptive fast Fourier transform module is employed as the backbone network to improve the dynamic environmental features extracting capability. Meanwhile, the decoupled diffusion model is utilized to further enhance the construction performance of RMs. Moreover, a comprehensive theoretical analysis of why the RM construction is a generative problem is provided for the first time, from both perspectives of data features and NN training methods. Experimental results show that the proposed RadioDiff achieves state-of-the-art performance in all three metrics of accuracy, structural similarity, and peak signal-to-noise ratio.

## :sunny: Before Starting

1. install torch
~~~
conda create -n radiodiff python=3.9
conda avtivate radiodiff
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
~~~
2. install other packages.
~~~
pip install -r requirement.txt
~~~
3. prepare accelerate config.
~~~
accelerate config # HOW MANY GPUs YOU WANG TO USE.
~~~

## :sparkler: Prepare Data

##### We used the [RadioMapSeer](https://radiomapseer.github.io/) dataset for model training and testing.

- The data structure should look like:

```commandline
|-- $RadioMapSeer
|   |-- gain
|   |-- |-- carsDPM
|   |-- |-- |-- XXX_XX.PNG
|   |-- |-- |-- XXX_XX.PNG
|   ...
|   |-- png
|   |-- |-- buildings_complete
|   |-- |-- |-- XXX_XX.PNG
|   |-- |-- |-- XXX_XX.PNG
|	...
```
## :tada: Training
1. train the first stage model (AutoEncoder):
~~~[inference_numpy_for_slide.py](..%2F..%2F..%2F..%2Fmedia%2Fhuang%2F2da18d46-7cba-4259-9abd-0df819bb104c%2Finference_numpy_for_slide.py)
accelerate launch train_vae.py --cfg ./configs/first_radio.yaml
~~~
2. you should add the final model weight of the first stage to the config file `./configs/BSDS_train.yaml` (**line 42**), then train latent diffusion-edge model:
~~~
accelerate launch train_cond_ldm.py --cfg ./configs/radio_train.yaml
~~~

## V. Inference.
make sure your model weight path is added in the config file `./configs/radio_sample.yaml` (**line 73**), and run:
~~~
python sample_cond_ldm.py --cfg ./configs/radio_sample.yaml
~~~
Note that you can modify the `sampling_timesteps` (**line 11**) to control the inference speed.

## :green_book: Contact
If you have some questions, please contact with KD.TAO@outlook.com.
## Thanks
Thanks to the base code [DDM-Public](https://github.com/GuHuangAI/DDM-Public).
## Citation
~~~
@ARTICLE{10764739,
  author={Wang, Xiucheng and Tao, Keda and Cheng, Nan and Yin, Zhisheng and Li, Zan and Zhang, Yuan and Shen, Xuemin},
  journal={IEEE Transactions on Cognitive Communications and Networking}, 
  title={RadioDiff: An Effective Generative Diffusion Model for Sampling-Free Dynamic Radio Map Construction}, 
  year={2025},
  volume={11},
  number={2},
  pages={738-750},
  keywords={Artificial neural networks;Training;Feature extraction;Diffusion models;Electromagnetics;Vehicle dynamics;Finite element analysis;Buildings;Noise;Costs;Radio map;denoise diffusion model;generative problem;wireless network},
  doi={10.1109/TCCN.2024.3504489}}
~~~
