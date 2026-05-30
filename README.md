# RadioDiff: An Effective Generative Diffusion Model for Sampling-Free Dynamic Radio Map Construction

---
## 📡 Welcome to the RadioDiff Family

> Radio map construction via generative diffusion models — UNIC Lab, Xidian University

---

### 🔷 Base Backbone

**RadioDiff** — *The foundational diffusion model for radio map construction.*
&nbsp;&nbsp;📄 [Paper](https://ieeexplore.ieee.org/document/10764739) &nbsp;|&nbsp; 💻 [Code](https://github.com/UNIC-Lab/RadioDiff) &nbsp;|&nbsp; ![IEEE TCCN](https://img.shields.io/badge/IEEE-TCCN%202025-blue)

---

### 🔬 Physics-Informed Extensions

**RadioDiff-k²** — *PINN-enhanced diffusion guided by the Helmholtz equation.*
&nbsp;&nbsp;📄 [Paper](https://ieeexplore.ieee.org/document/11278649) &nbsp;|&nbsp; 💻 [Code](https://github.com/UNIC-Lab/RadioDiff-k) &nbsp;|&nbsp; ![IEEE JSAC](https://img.shields.io/badge/IEEE-JSAC%202026-blue)

**iRadioDiff** — *Indoor radio map construction with physical information integration.*
&nbsp;&nbsp;📄 [Paper](https://arxiv.org/abs/2511.20015) &nbsp;|&nbsp; 💻 [Code](https://github.com/UNIC-Lab/iRadioDiff) &nbsp;|&nbsp; ![IEEE ICC](https://img.shields.io/badge/IEEE-ICC%202026-blue) &nbsp;![Best Paper](https://img.shields.io/badge/🏆-Best%20Paper%20Award-orange)

---

### ⚡ Efficiency & Dynamics

**RadioDiff-Turbo** — *Efficiency-enhanced RadioDiff for accelerated inference.*
&nbsp;&nbsp;📄 [Paper](https://ieeexplore.ieee.org/abstract/document/11152929/) &nbsp;|&nbsp; ![INFOCOM Workshop](https://img.shields.io/badge/IEEE-INFOCOM%20Wksp%202025-lightgrey)

**RadioDiff-Flux** — *Adaptive reconstruction under dynamic environments and base station location changes.*
&nbsp;&nbsp;📄 [Paper](https://ieeexplore.ieee.org/document/11282987/) &nbsp;|&nbsp; ![IEEE TCCN](https://img.shields.io/badge/IEEE-TCCN%202026-blue)

---

### 🌐 Extended Scenarios

**RadioDiff-3D** — *3D radio map construction with the UrbanRadio3D dataset.*
&nbsp;&nbsp;📄 [Paper](https://ieeexplore.ieee.org/document/11083758) &nbsp;|&nbsp; 💻 [Code](https://github.com/UNIC-Lab/UrbanRadio3D) &nbsp;|&nbsp; ![IEEE TNSE](https://img.shields.io/badge/IEEE-TNSE%202025-blue)

**RadioDiff-FS** — *Few-shot learning for radio map construction with limited measurements.*
&nbsp;&nbsp;📄 [Paper](https://arxiv.org/abs/2603.18865) &nbsp;|&nbsp; 💻 [Code](https://github.com/UNIC-Lab/RadioDiff-FS) &nbsp;|&nbsp; ![arXiv](https://img.shields.io/badge/arXiv-preprint-lightgrey)

---

### 📶 Sparse Measurement & Localization

**RadioDiff-Inverse** — *Sparse measurement-based radio map recovery for ISAC applications.*
&nbsp;&nbsp;📄 [Paper](https://arxiv.org/abs/2504.14298) &nbsp;|&nbsp; 💻 [Code](https://github.com/UNIC-Lab/radiodiff-inverse) &nbsp;|&nbsp; ![IEEE TWC](https://img.shields.io/badge/IEEE-TWC%202026-blue)

**RadioDiff-Loc** — *Sparse measurement-based NLoS localization using diffusion models.*
&nbsp;&nbsp;📄 [Paper](https://www.arxiv.org/abs/2509.01875) &nbsp;|&nbsp; ![arXiv](https://img.shields.io/badge/arXiv-preprint-lightgrey)

---

> 📚 For a comprehensive categorized overview of radio map research, visit [**Awesome-Radio-Map-Categorized**](https://github.com/UNIC-Lab/Awesome-Radio-Map-Categorized).


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
