<!-- PROJECT LOGO -->
<p align="center">
  
  <h1 align="center">Any Fashion Attribute Editing: Dataset and Pretrained Models</h1>
  <p align="center">
    <a href="https://github.com/FST-ZHUSHUMIN/"><strong>Shumin Zhu</strong></a>
    ·    
    <a href="https://github.com/AemikaChow/"><strong>Xingxing Zou</strong></a>
    ·
    <a href="https://github.com/flyywh"><strong>Wenhan Yang</strong></a>
    ·
    <a href="https://www.aidlab.hk/en/people-detail/prof-calvin-wong"><strong>Waikeung Wong</strong></a>
  </p>
  <h2 align="center">TPAMI 2025</h2>
  <div align="center">
    <img src="./all_result.jpg" alt="Logo" width="100%">
  </div>


<br />
<br />

## Quick View
This work focuses on `any' fashion attribute editing: 1) the ability to edit 78 fine-grained design attributes commonly observed in daily life; 2) the capability to modify desired attributes while keeping the rest components still; and 3) the flexibility to continuously edit on the edited image. Comprehensive experiments, including comparisons with ten state-of-the-art image inversion methods and four editing algorithms, demonstrate the effectiveness of our Twin-Net and editing algorithm.

  <div align="center">
    <img src="./more_all.jpg" alt="Logo" width="100%">
  </div>

Our main contributions are as follows:

$\bullet$  We present the Any Fashion Editing Dataset (AFED), which contains 830K high-quality full-body fashion images from two domains.
    This dataset creates a new challenge scene of any fashion editing with complex fashion attributes, making it ideal for arbitrary fashion attribute editing research.
    It inspires the design of the method in two directions: 1) relax the trade-off between inversion and editing; 2) better fine-grained semantics disentanglement.
    \textbf{All data, source code, models} will be released for academic use.

$\bullet$ We propose Twin-Net, a GAN-based framework for high-quality fashion image inversion and editing. Different from existing methods taking random noise as a condition for editability, our approach leverages factorized semantics as conditions, preserving the person's identity while allowing flexible, unsupervised editing of fashion attributes.

$\bullet$ We design PairsPCA, a method that is capable of mining the clear mapping relationship between latent and semantic meanings consistently. 
    This modeling facilitates the targeted editing of individual attributes without unintended effects on others and also enables attribute editing with the least human effort.

$\bullet$ Comprehensive experiments validate the effectiveness of our proposed methods. A total of 14 baselines were compared, including seven GAN inversion methods, three diffusion-based editing methods, and four GAN-based attribute editing methods. We trained 27 models across three datasets, comprising StyleGAN models and models for eight GAN inversion methods (including Twin-Net). The experiments evaluated reconstruction quality, editing capability, accuracy, and disentanglement. The results demonstrate the effectiveness of our proposed framework and methods.

  <div align="center">
    <img src="./more_all.jpg" alt="Why" width="100%">
  </div>

## How to Use



## Citation
```bib

```

<br>


<br>
