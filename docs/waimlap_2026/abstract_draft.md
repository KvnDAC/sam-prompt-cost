# The Informational Cost of Spatial Prompts in SAM: A Controlled Study on Polyp Segmentation

[Author names and order pending confirmation]

[Affiliations pending confirmation] | Corresponding author: [pending confirmation]

## 1. INTRODUCTION

Promptable segmentation models require information that identifies the intended target. Consequently, removing a spatial prompt changes both available localization and task specification. For polyp segmentation, an informative evaluation should distinguish inadequate mask candidates from failures to select a useful candidate. Building on SAM [1], earlier polyp evaluations [2], and automatic prompting approaches [3, 4], we investigate the informational cost of removing or degrading spatial prompts and assess learned alternatives. We define cost as loss in segmentation quality, rather than annotation time or monetary expense. Our contribution is a controlled empirical diagnosis and mitigation study, rather than a claim of architectural novelty.

## 2. METHODOLOGY

We study Kvasir-SEG [5] using 400 fixed training images and 100 development images. SAM ViT-B remains frozen, with FP32 image encoding, cached embeddings and decoding. We compare empty spatial input, GT-derived interior points and tight boxes, controlled prompt perturbations, and alternative decoder outputs. Eligible GT components receive separate prompts and their predicted masks are united; therefore, oracle prompts can reveal component count. Automatic methods use no evaluation GT: four static prompt tokens, an image-conditioned four-token generator, a linear segmentation probe over frozen SAM features, and probe-derived box/point prompts. Learned methods use training masks, including GT-informed meanpoint initialization for tokens.

Static and conditioned prompts use BCE+Dice, batch size eight and six training epochs. A three-rate search selects the conditioned learning rate on development data. Seeds 0, 1 and 2 are used for learned comparisons; checkpoints and probe thresholds are selected on development data. Dice is computed per image at original resolution. Confidence intervals bootstrap images after averaging training seeds, with seed dispersion considered separately. A fixed-point-free context permutation keeps the target image, decoder and generator weights fixed while changing the image embedding supplied to the generator.

## 3. RESULTS AND DISCUSSION

Table 1 reports development results; oracle prompts are privileged diagnostic references. Static tokens reached Dice 0.3633, whereas conditioned tokens reached 0.5013. All conditioned runs selected the final epoch, so convergence is not established. The probe-to-box-and-point pipeline outperformed the probe by 0.0690 Dice (paired 95% CI: 0.0471–0.0919), supporting automatic localization as a useful mitigation in this setting.

| Method | Development Dice [95% CI] |
| --- | --- |
| Empty spatial input, single output | 0.0089 [0.0007, 0.0227] |
| GT point, oracle | 0.6831 [0.6155, 0.7463] |
| GT box, oracle | 0.9352 [0.9279, 0.9420] |
| Static tokens | 0.3633 [0.2913, 0.4375] |
| Image-conditioned tokens | 0.5013 [0.4260, 0.5762] |
| Linear probe | 0.7489 [0.7067, 0.7879] |
| Probe → box+point → SAM | 0.8179 [0.7707, 0.8600] |

Spatial precision matters: shifting points by two interior radii reduced Dice to 0.1529, and doubling box dimensions yielded 0.3216. Without spatial prompts, even GT-based selection of the best of four decoder outputs reached only 0.1824. Thus, selection alone cannot recover box-guided quality from this candidate set. With GT points, the best of three candidates reached 0.8600, compared with 0.5401 when selected by predicted IoU, exposing a distinct selection gap under conditioning.

Permuting generator context reduced Dice from 0.501337 to 0.495295: a difference of 0.006042 (95% CI: 0.000840–0.011245). This modest sensitivity does not justify attributing the full conditioned-versus-static improvement to image-specific context; capacity and optimization also differ. A capacity-matched constant-context control remains necessary for a stronger explanation.

## 4. CONCLUSIONS

The observed failure depends on the conditioning regime: useful candidates are limited under empty input, while selection can limit point-conditioned predictions. Probe-derived spatial prompts provide the strongest automatic mitigation among the evaluated methods. These findings characterize frozen SAM behavior without isolating an intrinsic decoder defect. Results are exploratory: development data guided selection, the historical test set was previously inspected, and no new independent evaluation or human timing study was performed. Complete U-Net and SAM 2.1 comparisons, common-hardware cost measurements and external evaluation remain planned work.

## REFERENCES

[1] A. Kirillov et al., “Segment Anything,” ICCV, 2023.

[2] T. Zhou et al., “Can SAM Segment Polyps?,” arXiv:2304.07583, 2023.

[3] Q. Wu, Y. Zhang, and M. Elbatel, “Self-Prompting Large Vision Models for Few-Shot Medical Image Segmentation,” arXiv:2308.07624, 2023.

[4] T. Shaharabany, A. Dahan, R. Giryes, and L. Wolf, “AutoSAM: Adapting SAM to Medical Images by Overloading the Prompt Encoder,” arXiv:2306.06370, 2023.

[5] D. Jha et al., “Kvasir-SEG: A Segmented Polyp Dataset,” MMM, 2020, pp. 451–462.
