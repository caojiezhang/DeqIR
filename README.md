## Deep Equilibrium Diffusion Restoration with Parallel Sampling
[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2311.11600)
[![download](https://img.shields.io/github/downloads/caojiezhang/DeqIR/total.svg)](https://github.com/caojiezhang/DeqIR/releases)
![visitors](https://visitor-badge.laobi.icu/badge?page_id=caojiezhang/DeqIR)

[Jiezhang Cao](https://scholar.google.com/citations?hl=en&user=IFYbb7oAAAAJ&view_op=list_works&sortby=pubdate), 
[Yue Shi](https://scholar.google.com/citations?user=BrQQHiEAAAAJ&hl=en), 
[Kai Zhang](https://cszn.github.io/), [Yulun Zhang](http://yulunzhang.com/), 
[Radu Timofte](http://people.ee.ethz.ch/~timofter/), 
[Luc Van Gool](https://scholar.google.com/citations?user=TwMib_QAAAAJ&hl=en)

Computer Vision Lab, ETH Zurich

I will release code next week.

---

> **Abstract:** Diffusion-based image restoration (IR) methods aim to use diffusion models to recover high-quality (HQ) images from degraded images and achieve promising performance. Due to the inherent property of diffusion models, most of these methods need long serial sampling chains to restore HQ images step-by-step. As a result, it leads to expensive sampling time and high computation costs. Moreover, such long sampling chains hinder understanding the relationship between the restoration results and the inputs since it is hard to compute the gradients in the whole chains. In this work, we aim to rethink the diffusion-based IR models through a different perspective, i.e., a deep equilibrium (DEQ) fixed point system. Specifically, we derive an analytical solution by modeling the entire sampling chain in diffusion-based IR models as a joint multivariate fixed point system. With the help of the analytical solution, we are able to conduct single-image sampling in a parallel way and restore HQ images without training. Furthermore, we compute fast gradients in DEQ and found that initialization optimization can boost performance and control the generation direction. Extensive experiments on benchmarks demonstrate the effectiveness of our proposed method on typical IR tasks and real-world settings. 
![](figs/comp_sampling.png)

## ⚒️ TODO

* [ ] Complete this repository

## 🔗 Contents

- [ ] Datasets
- [ ] Installation
- [ ] Training
- [ ] Testing
- [x] [Results](#Results)
- [x] [Citation](#Citation)

## 🔎 Results

We achieved state-of-the-art performance on many image restoration tasks. More results can be found in the paper.

<details>
<summary>Quantitative Comparison (click to expan)</summary>
<p align="center">
  <img width="900" src="figs/tab_sr_deblur.png">
</p>
<p align="center">
  <img width="900" src="figs/tab_inp_color.png">
</p>
</details>

<details>
<summary>Visual Comparison (click to expan)</summary>

- Classical image restoration

<p align="center">
  <img width="900" src="figs/fig_sr.png">
</p>
<p align="center">
  <img width="900" src="figs/fig_deblur.png">
</p>
<p align="center">
  <img width="900" src="figs/fig_inp.png">
</p>
<p align="center">
  <img width="900" src="figs/fig_color.png">
</p>

- Real-world image restoration
<p align="center">
  <img width="900" src="figs/fig_real.png">
</p>

- Generation diversity
<p align="center">
  <img width="900" src="figs/fig_diversity.png">
</p>

- Initialization optimization via Inversion
<p align="center">
  <img width="900" src="figs/fig_inv.png">
</p>

</details>

## 📎 Citation

If you find the code helpful in your resarch or work, please cite the following paper(s).

```
@article{cao2024deqir,
    title   = {Deep Equilibrium Diffusion Restoration with Parallel Sampling}, 
    author  = {Jiezhang Cao and Yue Shi and Kai Zhang and Yulun Zhang and Radu Timofte and Luc Van Gool},
    journal = {CVPR},
    year    = {2024},
}
```
