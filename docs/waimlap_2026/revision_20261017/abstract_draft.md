# The Informational Cost of Spatial Prompts in SAM: A Controlled Study on Polyp Segmentation

[Author names and order pending confirmation]

[Affiliations pending confirmation] | Corresponding author: [pending confirmation]

## 1. INTRODUCTION

Removing spatial prompts from SAM changes available localization and target specification. For polyp segmentation, diagnosis should distinguish missing mask candidates from failures to select a useful candidate. Building on SAM [1], prior polyp evaluations [2] and automatic prompting approaches [3, 4], we study the quality lost when spatial information is removed or degraded and assess learned replacements. We investigate informational loss and computational cost; human annotation time is not measured. Our contribution is controlled empirical evidence, rather than architectural novelty.

## 2. METHODOLOGY

We use 400 fixed training and 100 development images from Kvasir-SEG [5]. SAM ViT-B is frozen, with FP32 encoding, caching and decoding at longest-side input 1024. Conditions include empty spatial input, GT-derived interior points and boxes, controlled perturbations and decoder-output selection. Eligible GT components receive separate prompts and their masks are united, so oracle prompts can disclose component count. Automatic methods use no evaluation GT: four static tokens, an image-conditioned four-token generator, a linear segmentation probe over frozen SAM features, and probe-derived box/point prompts. Token initialization uses training GT through meanpoint. A fixed-prior control uses median normalized train-mask coordinates, with one box/point query per evaluated image.

Three additional RGB controls are trained from scratch: compact U-Net [6], a depthwise-separable U-shaped variant and FCN-Lite [7]. They are declared local variants, not exact paper reproductions. All use input 128, GroupNorm, BCE+Dice, Adam at 0.001, batch eight, horizontal/vertical flips and up to 30 epochs, with minimum 15 and patience eight. SAM token models use six epochs and a development-selected conditioned learning rate. Seeds 0/1/2 are used for learned comparisons; checkpoints and probe thresholds are selected on development data.

Dice is averaged per image at original resolution. Confidence intervals bootstrap images after averaging training seeds. A fixed-point-free permutation changes generator context while holding the target image, decoder and weights fixed. Latency uses decoded RGB, CPU/four threads, FP32, batch one, five warmups and twenty timed images; preprocessing, full network and restoration are included, while disk decoding and metrics are excluded. SAM timing includes its image encoder.

## 3. RESULTS AND DISCUSSION

Table 1 reports development-selected scores. Probe-derived box/point prompts improve over the probe by 0.0690 Dice (paired 95% CI: 0.0471–0.0919). Their advantage over compact U-Net is 0.0365 [−0.0055, 0.0771]: the mean difference is unresolved by this interval, which also does not establish equivalence. Different pretraining and input sizes further limit causal architectural comparisons.

| Method | Development Dice [95% CI] |
| --- | --- |
| Empty spatial input, single output | 0.0089 [0.0007, 0.0227] |
| GT point, oracle | 0.6831 [0.6155, 0.7463] |
| GT box, oracle | 0.9352 [0.9279, 0.9420] |
| Static tokens | 0.3633 [0.2913, 0.4375] |
| Image-conditioned tokens | 0.5013 [0.4260, 0.5762] |
| Linear probe | 0.7489 [0.7067, 0.7879] |
| Probe → box+point → SAM | 0.8179 [0.7707, 0.8600] |
| FCN-Lite, RGB 128 | 0.6002 [0.5549, 0.6429] |
| DS-U-Net, RGB 128 | 0.6548 [0.6122, 0.6946] |
| Compact U-Net, RGB 128 | 0.7814 [0.7447, 0.8140] |

Fixed train-derived box/point prompts reach 0.3582, versus 0.8179 for image-specific automatic localization (difference 0.4597 [0.3809, 0.5370]). Without prompts, even GT selection of the best of four decoder outputs reaches only 0.1824. With GT points, best-of-three reaches 0.8600 versus 0.5401 under predicted-IoU selection. Thus, candidate availability and selection limit different conditioning regimes.

Permuting generator context reduces Dice by only 0.0060 [0.0008, 0.0112], insufficient to attribute its full advantage over static tokens to matched image context. Several models select the final training epoch; convergence is not established. Median full-pipeline latency was 8.37 s for SAM automatic localization, versus 12.11 ms for compact U-Net on this CPU; the inputs and pretraining differ.

## 4. CONCLUSIONS

Image-specific automatic localization substantially improves frozen SAM over a fixed spatial prior. Compact supervised segmentation is a competitive control whose mean quality difference from the SAM pipeline remains uncertain at this budget. These findings characterize conditioning dependence without isolating an intrinsic decoder defect. Development guided selection, the historical test was already inspected, and no new independent evaluation was performed. Larger-resolution/convergence studies, label-budget experiments and external evaluation are needed before stronger claims.

## REFERENCES

[1] A. Kirillov et al., “Segment Anything,” ICCV, 2023.

[2] T. Zhou et al., “Can SAM Segment Polyps?,” arXiv:2304.07583, 2023.

[3] Q. Wu, Y. Zhang, and M. Elbatel, “Self-Prompting Large Vision Models for Few-Shot Medical Image Segmentation,” arXiv:2308.07624, 2023.

[4] T. Shaharabany, A. Dahan, R. Giryes, and L. Wolf, “AutoSAM: Adapting SAM to Medical Images by Overloading the Prompt Encoder,” arXiv:2306.06370, 2023.

[5] D. Jha et al., “Kvasir-SEG: A Segmented Polyp Dataset,” MMM, 2020, pp. 451–462.

[6] O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional Networks for Biomedical Image Segmentation,” MICCAI, 2015.

[7] J. Long, E. Shelhamer, and T. Darrell, “Fully Convolutional Networks for Semantic Segmentation,” CVPR, 2015.
