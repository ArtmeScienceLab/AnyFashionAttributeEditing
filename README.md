<!-- PROJECT LOGO -->
<p align="center">
  
  <h1 align="center">Any Fashion Attribute Editing: Dataset and Pretrained Models (TPAMI2025)</h1>
  <p align="center">
    <a href="https://github.com/FST-ZHUSHUMIN/"><strong>Shumin Zhu</strong></a>
    ·    
    <a href="https://github.com/AemikaChow/"><strong>Xingxing Zou</strong></a>
    ·
    <a href="https://github.com/flyywh"><strong>Wenhan Yang</strong></a>
    ·
    <a href="https://www.aidlab.hk/en/people-detail/prof-calvin-wong"><strong>Waikeung Wong</strong></a>
  </p>
  <!-- <h2 align="center">TPAMI 2025</h2> -->

  <div align="center">
    <img src="./all_result.jpg" alt="Why" width="100%">
  </div>

<br />

## Quick View

 <div align="center">
    <img src="./Why.jpg" alt="Why" width="100%">
  </div>
  
Our **main contributions** are as follows:

$\bullet$ We present the Any Fashion Editing Dataset (AFED), which comprises 830,000 high-quality full-body fashion images from two distinct domains.

$\bullet$ We propose Twin-Net, a GAN-based framework for high-quality fashion image inversion and editing.

$\bullet$ We design PairsPCA, a method capable of consistently mining the clear mapping relationship between latent and semantic meanings. 

<br />

This work focuses on `any' fashion attribute editing: 1) the ability to edit 78 fine-grained design attributes commonly observed in daily life; 2) the capability to modify desired attributes while keeping the rest of the components still; and 3) the flexibility to continuously edit the edited image. Comprehensive experiments, including comparisons with ten state-of-the-art image inversion methods and four editing algorithms, demonstrate the effectiveness of our Twin-Net and editing algorithm.

  <div align="center">
    <img src="./more_all.jpg" alt="Logo" width="100%">
  </div>

## Get Started



## Resources Download
-【[**Obtain AFED**](https://drive.google.com/drive/)】The AFED dataset consists of three sub-datasets, as illustrated in the Figure below.  It includes 300,000 (a) solid-colour women's clothing sketches (AFED-S-Colour), 300,000 (b) printed men's clothing sketches (AFED-S-Print), and 230,000 (c) human fashion images (AFED-H-Product). For the AFED-S-Colour and AFED-S-Print, the data were generated using a mature pipeline [**AiDA**](https://www.aida.com.hk/), which was previously utilised to create full-body sketch images with fine-grained fashion attributes. Generating these sketches required approximately 250 hours on a single NVIDIA GeForce GTX 3090 GPU. 

<div align="center">
    <img src="./dataset.jpg" alt="Logo" width="100%">
  </div>

-【[**Pretrained Models**](https://drive.google.com/drive/)】A total of 14 baselines were compared, including seven GAN inversion methods, three diffusion-based editing methods, and four GAN-based attribute editing methods. We trained 27 models across three datasets, comprising StyleGAN models and models for eight GAN inversion methods (including Twin-Net). 

## Citation
```bib

```

## Acknowledgments

This work is partially supported by the Laboratory for Artificial Intelligence in Design (Project Code: RP3-1), Innovation and Technology Fund, Hong Kong, SAR. This work is also partially supported by a grant from the Research Grants Council of the Hong Kong, SAR(Project No. PolyU/RGC Project PolyU 25211424).

<br>


<br>
