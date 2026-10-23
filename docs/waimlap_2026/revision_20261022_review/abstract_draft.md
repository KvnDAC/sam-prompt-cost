# The Informational Cost of Spatial Prompts in SAM: A Controlled Study on Polyp Segmentation

[Author names and order pending confirmation]

[Affiliations pending confirmation] | Corresponding author: [pending confirmation]

## 1. INTRODUCTION

Removing spatial prompts from Segment Anything (SAM) [1] changes both localization and target specification. For polyp segmentation, a useful diagnosis must distinguish missing mask candidates from failure to select a good candidate. Prior work shows that SAM struggles on polyps without prompts [2], that its chosen mask can trail the best available one [3], that it is sensitive to imprecise prompts [4, 5], and that prompts can be generated automatically [6, 7]. We combine these threads under shared image IDs: quality lost when spatial information is removed or degraded, automatic replacements, label efficiency, external transfer, false activation, and computational cost. Human annotation time is not measured; the contribution is controlled empirical evidence rather than a new prompting architecture.

## 2. METHODOLOGY

We use a fixed 400-image training subset and 100 development images from Kvasir-SEG [8]. SAM ViT-B is frozen and uses FP32 at longest-side input 1024. Conditions include empty spatial input, oracle GT points and boxes, prompt perturbations, output selection, static and image-conditioned tokens, a 257-parameter linear probe over frozen image features, and probe-derived box/point prompts. A fixed spatial prior uses median normalized training-mask coordinates. Oracle prompts can reveal component count and are diagnostic ceilings.

A compact U-Net [9] (487k parameters) is trained from scratch for 30 epochs at 128 pixels, then continued for 15 epochs at 128 or 256 pixels from identical weights, Adam states, and RNG states. A label-efficiency study uses nested 25/100/400-image subsets and fixed update budgets (300 probe; 1,500 U-Net, directly at 256 pixels). Seeds 0/1/2 are used. Development intervals bootstrap images after averaging seeds; model and threshold selection uses development data.

Frozen models are applied without adaptation to 612 CVC-ClinicDB frames from 29 sequences [10], with sequence-clustered intervals. False activation is measured on all 1,009 HyperKvasir cecum-class images [11] using predicted area and component size. Those class labels are a negative proxy, not pixel-reviewed negative masks; patient/video IDs are unavailable. CPU latency includes the full RGB-to-mask pipeline.

## 3. RESULTS AND DISCUSSION

Empty input nearly collapses (Dice 0.0089), while oracle points/boxes reach 0.6831/0.9352. Static tokens recover 0.3633, whose mean matches a fixed train-derived point (0.3633); conditioned tokens reach 0.5013, but permuting their image context lowers Dice by only 0.0060. The probe reaches 0.7489, and probe-derived box+point prompts raise Dice to 0.8179 (+0.0690 [0.0471, 0.0919]); a fixed train box+point prior reaches only 0.3582, identifying image-specific localization as the main source of recovery.

Candidate availability and selection fail differently: with empty input, the best of four outputs reaches only 0.1824; with a correct point, the best of three reaches 0.8600 versus 0.5401 under predicted-IoU selection. Ranking cannot repair missing candidates, but has headroom after localization.

The longer U-Net recipe at 256 pixels reaches 0.8267; SAM−U-Net is −0.0088 [−0.0472, 0.0281], resolving neither superiority nor equivalence. Under the fixed 1,500-update budget, SAM exceeds U-Net by +0.2718, +0.1863, and +0.1226 at 25, 100, and 400 masks (all CIs exclude zero). The 400-mask gap reflects that budget, since the longer recipe closes it with the same masks: SAM offers a stronger start when labels or training are limited, not a higher ceiling.

On CVC-ClinicDB, Dice falls to 0.4587 for SAM and 0.3405 for U-Net (difference +0.1182 [0.0672, 0.1684]). On HyperKvasir negatives, mean false area is 0.0521 versus 0.1733 (difference −0.1212 [−0.1267, −0.1159]); the gap persists if U-Net keeps only its largest component (0.1558). The probe fires on every negative image, so the always-prompt SAM pipeline always emits a mask, and its decoder enlarges the prompted component (+0.0136, post hoc); U-Net emits masks in 99.83% of images. SAM transfers better, but neither result supports clinical adequacy. Median CPU latency is 8.37 s versus 27.08 ms.

![Figure 1. Prompt dependence, label efficiency, frozen positive transfer, and negative false activation. Error bars show 95% bootstrap intervals. In B, U-Net uses the fixed 1,500-update budget; the open marker is the longer recipe with the same 400 masks. D labels give the rate with at least 1% false area.](figure_experimental_summary.png)

## 4. CONCLUSIONS

Image-specific localization recovers much of SAM's prompt gap, and its decoder improves a coarse probe. A U-Net reaches comparable Kvasir development Dice (difference unresolved) with about 300 times lower CPU latency, whereas SAM starts higher with few labels or short training and transfers better to CVC-ClinicDB. Neither rejects target absence. The contribution is an integrated diagnosis of prompt dependence, candidate generation/selection, utility, and cost under shared IDs—not architectural novelty. Next, a presence/abstention gate should be evaluated on grouped negative videos.

## REFERENCES

[1] A. Kirillov et al., “Segment Anything,” in Proc. ICCV, 2023.

[2] T. Zhou et al., “Can SAM Segment Polyps?,” arXiv:2304.07583, 2023.

[3] M. A. Mazurowski et al., “Segment Anything Model for Medical Image Analysis: An Experimental Study,” Medical Image Analysis, 102918, 2023.

[4] M. M. Rahman et al., “PP-SAM: Perturbed Prompts for Robust Adaptation of Segment Anything Model for Polyp Segmentation,” in Proc. CVPR Workshops, 2024.

[5] Q. Fan et al., “Stable Segment Anything Model,” arXiv:2311.15776, 2023.

[6] Q. Wu, Y. Zhang, and M. Elbatel, “Self-Prompting Large Vision Models for Few-Shot Medical Image Segmentation,” arXiv:2308.07624, 2023.

[7] T. Shaharabany et al., “AutoSAM: Adapting SAM to Medical Images by Overloading the Prompt Encoder,” arXiv:2306.06370, 2023.

[8] D. Jha et al., “Kvasir-SEG: A Segmented Polyp Dataset,” in Proc. MMM, 2020.

[9] O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional Networks for Biomedical Image Segmentation,” in Proc. MICCAI, 2015.

[10] J. Bernal et al., “WM-DOVA Maps for Accurate Polyp Highlighting in Colonoscopy: Validation vs. Saliency Maps from Physicians,” Comput. Med. Imaging Graph., 2015.

[11] H. Borgli et al., “HyperKvasir, a Comprehensive Multi-Class Image and Video Dataset for Gastrointestinal Endoscopy,” Scientific Data, 2020.
