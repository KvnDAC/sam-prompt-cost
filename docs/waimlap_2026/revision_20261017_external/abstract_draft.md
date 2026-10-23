# The Informational Cost of Spatial Prompts in SAM: A Controlled Study on Polyp Segmentation

[Author names and order pending confirmation]

[Affiliations pending confirmation] | Corresponding author: [pending confirmation]

## 1. INTRODUCTION

Removing spatial prompts from SAM changes available localization and target specification. For polyp segmentation, diagnosis should distinguish missing mask candidates from failures to select a useful candidate. Building on SAM [1], prior polyp evaluations [2] and automatic prompting approaches [3, 4], we study the quality lost when spatial information is removed or degraded and assess learned replacements. We investigate informational loss and computational cost; human annotation time is not measured. Our contribution is controlled empirical evidence, rather than architectural novelty.

## 2. METHODOLOGY

We use 400 fixed training and 100 development images from Kvasir-SEG [5]. SAM ViT-B is frozen, with FP32 encoding, caching and decoding at longest-side input 1024. Conditions include empty spatial input, GT-derived interior points and boxes, controlled perturbations and decoder-output selection. Eligible GT components receive separate prompts and their masks are united, so oracle prompts can disclose component count. Automatic methods use no evaluation GT: four static tokens, an image-conditioned four-token generator, a linear segmentation probe over frozen SAM features, and probe-derived box/point prompts. Token initialization uses training GT through meanpoint. A fixed-prior control uses median normalized train-mask coordinates, with one box/point query per evaluated image.

A compact U-Net [6] is trained from scratch and continued from identical epoch-30 weights, Adam and RNG states for 15 fixed epochs at 128 or 256. A label-efficiency study uses nested 25/100/400-image subsets and fixed update counts (300 probe; 1500 U-Net). Seeds 0/1/2 are used; checkpoints and thresholds are selected on development data.

Dice is measured at original resolution. Development intervals bootstrap images after averaging seeds. Frozen models are applied without adaptation to 612 CVC-ClinicDB frames from 29 sequences [7]; external intervals resample complete sequences. CPU latency includes preprocessing, the full model and restoration.

## 3. RESULTS AND DISCUSSION

Table 1 reports development scores. Probe-derived box/point prompts improve over the probe by 0.0690 Dice (paired 95% CI: 0.0471–0.0919). At the fixed 45-epoch endpoint, U-Net 256 reaches 0.8267 and SAM minus U-Net is −0.0088 [−0.0472, 0.0281]. The mean difference is unresolved by this interval, which also does not establish equivalence. Different pretraining and input sizes further limit causal architectural comparisons.

| Method | Development Dice [95% CI] |
| --- | --- |
| Empty spatial input, single output | 0.0089 [0.0007, 0.0227] |
| GT point, oracle | 0.6831 [0.6155, 0.7463] |
| GT box, oracle | 0.9352 [0.9279, 0.9420] |
| Static tokens | 0.3633 [0.2913, 0.4375] |
| Image-conditioned tokens | 0.5013 [0.4260, 0.5762] |
| Linear probe | 0.7489 [0.7067, 0.7879] |
| Probe → box+point → SAM | 0.8179 [0.7707, 0.8600] |
| Compact U-Net, RGB 256, fixed epoch 45 | 0.8267 [0.7944, 0.8546] |

A fixed train-derived box/point prior reaches 0.3582 versus 0.8179 for image-specific localization. Without prompts, GT selection of the best of four outputs reaches only 0.1824; with GT points, best-of-three reaches 0.8600 versus 0.5401 under predicted-IoU selection. Candidate availability and selection therefore limit different regimes.

Under fixed optimization, automatic SAM versus U-Net differs by +0.2718 [0.2185, 0.3243], +0.1863 [0.1333, 0.2390], and +0.1226 [0.0757, 0.1692] at 25, 100, and 400 masks. On CVC-ClinicDB, Dice drops to 0.4587 for SAM and 0.3405 for U-Net; the clustered difference is +0.1182 [0.0672, 0.1684]. SAM transfers better relatively, while neither score supports clinical adequacy. Median latency was 8.37 s for SAM versus 27.08 ms for U-Net.

## 4. CONCLUSIONS

Image-specific localization substantially improves frozen SAM over a fixed spatial prior. A mature U-Net matches its Kvasir development quality at far lower computational cost, whereas SAM has a clear advantage with few labels and under frozen cross-dataset transfer. Both systems degrade severely on CVC-ClinicDB. These findings characterize conditioning dependence and relative transfer robustness without isolating an intrinsic decoder defect or establishing clinical generalization. Specificity on negative examinations remains unmeasured.

## REFERENCES

[1] A. Kirillov et al., “Segment Anything,” ICCV, 2023.

[2] T. Zhou et al., “Can SAM Segment Polyps?,” arXiv:2304.07583, 2023.

[3] Q. Wu, Y. Zhang, and M. Elbatel, “Self-Prompting Large Vision Models for Few-Shot Medical Image Segmentation,” arXiv:2308.07624, 2023.

[4] T. Shaharabany, A. Dahan, R. Giryes, and L. Wolf, “AutoSAM: Adapting SAM to Medical Images by Overloading the Prompt Encoder,” arXiv:2306.06370, 2023.

[5] D. Jha et al., “Kvasir-SEG: A Segmented Polyp Dataset,” MMM, 2020, pp. 451–462.

[6] O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional Networks for Biomedical Image Segmentation,” MICCAI, 2015.

[7] J. Bernal et al., “WM-DOVA Maps for Accurate Polyp Highlighting in Colonoscopy,” Computerized Medical Imaging and Graphics, 2015.
