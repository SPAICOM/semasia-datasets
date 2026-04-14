"""
get_model_metadata(model_name) -> dict
=======================================
Pure-Python parser for timm model names. No weights are downloaded.
timm is only imported to fill ``num_parameters`` and ``latent_dim``;
all other fields are parsed from the name string alone.

Timm model names follow the convention::

    {model_variant}.{pretrain_config}

    e.g.  vit_large_patch14_clip_224.openai_ft_in1k
          convnextv2_huge.fcmae_ft_in22k_in1k_384
          aimv2_1b_patch14_224.apple_pt

The variant encodes architecture structure; the config encodes how and
where the model was pretrained (and optionally fine-tuned).

Usage
-----
>>> from model_metadata import get_model_metadata
>>> meta = get_model_metadata('vit_base_patch16_rope_reg1_gap_256.sbb_in1k')
>>> meta['family']  # "ViT"
>>> meta['size']  # "Base"
>>> meta['positional_encoding']  # "RoPE"
>>> meta['num_registers']  # 1
>>> meta['head_type']  # "GAP"
>>> meta['pretrain_org']  # "timm SBB recipe"
>>> meta['pretrain_dataset']  # "ImageNet-1K"

Field reference
---------------
All fields are always present in the returned dict. Fields that could not
be inferred are ``None`` (or ``False`` for booleans).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IDENTITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

model_name : str
    The full original timm model name passed to the function.
    Example: "vit_large_patch14_clip_224.openai_ft_in1k"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARCHITECTURE  (parsed from the part before the dot)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

family : str
    Coarse architectural family. Derived from the leading prefix of the
    model variant using an ordered prefix table (~200 entries).
    Examples: "ViT", "ConvNeXt", "ResNet", "ResNeXt", "Swin", "DeiT",
              "EfficientNet", "MobileNet", "MaxViT", "AIMv2", "EVA".
    Falls back to "unknown" with a logged warning for unrecognised prefixes.

macro_family : str | None
    Coarse architectural category grouping families by their dominant
    computational motif. Values:
      "Vision Transformer"              → pure self-attention over patches
                                          (ViT, DeiT, CaiT, BEiT, EVA, …)
      "Hierarchical Vision Transformer" → multi-stage windowed / pooling
                                          attention (Swin, MViT, Hiera, …)
      "Hybrid CNN-Transformer"          → conv stages + transformer blocks
                                          (CoAtNet, MaxViT, FastViT, …)
      "MetaFormer"                      → abstract token-mixer backbone
                                          (PoolFormer, ConvFormer, …)
      "Convolutional"                   → pure convolutional backbone
                                          (ResNet, EfficientNet, ConvNeXt, …)
      "MLP"                             → pure MLP / token-mixing
                                          (MLP-Mixer, ResMLP, …)
      "State Space Model"               → selective state-space models
                                          (MambaOut, …)
    None for unrecognised families.

model_version : str | None
    The *architecture generation* tag encoded in the prefix — distinct
    from the size or the pretrain recipe version.
    Examples:
      "v2"  → ConvNeXtV2, SwinV2, MobileNetV2, BEiTv2, MobileViTv2 …
      "v3"  → BEiT3
      "III" → DeiT3 (DeiT-III)
      "RS"  → ResNet-RS (Revisited Scaling)
      "AA"  → ResNet-AA (Anti-Aliasing variant)
      "SE"  → SE-ResNet / SE-ResNeXt (Squeeze-and-Excitation)
      "ECA" → ECA-ResNet (Efficient Channel Attention)
      "GC"  → GC-ResNet (Global Context)
      "NF"  → NF-ResNet / NF-RegNet (Normalizer-Free)
      "CSP" → CSP-DarkNet / CSP-ResNeXt (Cross Stage Partial)
      "2"   → SAM2-Hiera
    None for families without a meaningful generation tag.

size : str | None
    Human-readable capacity label extracted from the variant string.
    Matched as a whole underscore-delimited token (never as a substring).
    Named sizes: "Tiny", "Small", "Base", "Medium", "Large", "Huge",
                 "Giant", "Gigantic", "XLarge", "XXLarge", "Nano",
                 "Pico", "Femto", "Atto", "Zepto", "Micro", "Mini",
                 "MediumD", "Betwixt", "Wee", "pWee", "dWee", "Little".
    Param-count sizes: "1B", "3B" (AIMv2, ViTamin gigantic variants).
    Special sizes: "SO400M", "SO150M", "SO150M2" (SigLIP capacity codes),
                   "Huge+" (ViT-Huge+), "Base+", "Small+", "Large2".
    EfficientNet B-series: "B0" – "B8".
    EfficientNet lite/edge: "Lite0" – "Lite4", "EL", "EM", "ES".
    TinyViT param sizes: "5M", "11M", "21M".
    MambaOut named variants: "Kobe" (Kobe model, ~100M params).
    None for families that use numeric depth codes instead (ResNet, NFNet).

depth_code : str | None
    Numeric or alphabetic depth identifier. Distinct from size — applies
    to families where the depth is the primary scaling axis rather than a
    named tier.
    Examples:
      "50", "101", "152", "270"  → ResNet depth (ResNet-50, ResNet-101 …)
      "f0" – "f6"                → NFNet depth (dm_nfnet_f4)
      "12", "24"                 → XCiT depth (number of blocks)
    None for families that use named size tiers.

width_code : str | None
    Channel width or group-width multiplier encoded in the variant.
    Formats vary by family:
      "32x4d"   → ResNeXt group notation  (32 groups of width 4d)
      "w44"     → HRNet width             (hrnet_w44)
      "14w_8s"  → Res2Net width×scale     (res2net50_14w_8s)
      "x16"     → CLIP-scaled ResNet      (resnet50x16_clip_gap)
      "075"     → MobileNet % multiplier  (tf_mobilenetv3_large_075)
    None when no width modifier is present.

patch_size : int | None
    ViT / transformer patch size in pixels. Parsed from ``_patch{N}_``
    tokens (standard ViT naming) and ``_p{N}_`` tokens (XCiT naming).
    Common values: 4 (Swin), 8, 14 (DINOv2 / CLIP), 16, 32.
    None for CNN families (ResNet, EfficientNet …) and patch-free models.

input_resolution : int | None
    The native input resolution the model was designed for, encoded
    directly in the variant string (e.g. ``vit_base_patch16_224`` → 224).
    Recognised values: 160, 192, 224, 240, 256, 288, 320, 336, 378, 384,
    396, 448, 480, 512, 576, 640, 768, 896, 1024.
    Note: this is the *architectural* resolution from the variant, not the
    pretrain or fine-tune resolution (see ``pretrain_resolution`` and
    ``pretrain_ft_resolution``).
    None when no resolution is encoded in the variant.

window_size : int | None
    Local attention window size for window-based transformers.
    Parsed from ``_window{N}_`` tokens, capturing the *first* (pretrain)
    window when cross-resolution notation is used (e.g. "window12to16"
    yields 12).
    Applies to: Swin, SwinV2, MaxViT, CoAtNet.
    None for global-attention models (ViT, ConvNeXt …).

stride_code : str | None
    Stage or stride code used by certain families as their primary depth /
    width dial, where neither a named size nor a numeric depth applies.
    Parsed from ``_s{N}_`` tokens (excluding families whose prefix starts
    with "swin" or "sam", where the token would be ambiguous).
    Examples:
      "s24", "s36", "s48"  → CaiT (24/36/48 self-attention blocks)
      "s16"                → gMLP stage count
      "s18", "s36"         → ConvFormer / CAFormer stage configuration
      "s4", "s3", "s2"     → ShViT stage code
    None when the family does not use this convention.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HEAD & ATTENTION  (parsed from the variant)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

head_type : str | None
    How the model pools its final representations before the classifier.
    Parsed from ``_gap_``, ``_cls_``, ``_clsgap_`` tokens.
      "GAP"      → Global Average Pooling over all patch tokens.
                   Common in DINOv2-style ViTs and SigLIP models.
      "CLS"      → Single [CLS] token (original BERT/ViT convention).
      "CLS+GAP"  → Both CLS and GAP concatenated (some hybrid models).
    None when the head type is the default for that family (e.g. most
    standard ViTs use CLS implicitly and do not encode it in the name).

num_registers : int | None
    Number of "register" tokens appended to the token sequence.
    Introduced in DINOv2 (Darcet et al., 2023) to absorb artefact
    attention patterns. Parsed from ``_reg{N}_`` tokens.
    Common values: 1, 4.
    Examples: vit_base_patch16_rope_reg1_gap_256 → 1
              vit_large_patch14_reg4_dinov2       → 4
    None for models without register tokens.

positional_encoding : str | None
    Non-default positional encoding scheme encoded in the variant name.
    Parsed from a priority-ordered list of token patterns:
      "RoPE-Mixed-APE"  → RoPE with mixed and absolute PE
      "RoPE-Mixed"      → Rotary PE with mixed frequencies
      "RoPE-APE"        → Rotary PE + Absolute PE
      "RoPE"            → Rotary Position Embedding (Su et al., 2021)
      "RelPos"          → Relative Position Bias (Swin-style)
      "SRelPos"         → Shared Relative Position Bias
      "RPN"             → Relative Position with Neighbours
      "SinCos"          → Fixed sinusoidal (non-learned)
      "APE"             → Absolute Position Embedding (explicit token)
    None for standard learned absolute PE, which is the default for most
    ViTs and is not encoded in the name.

activation : str | None
    Non-default activation function encoded in the variant name.
    Currently only "QuickGELU" is detected (``_quickgelu_`` token).
    QuickGELU is a fast approximation used in OpenAI CLIP models:
      GELU(x) ≈ x · σ(1.702x)
    It produces slightly different outputs from PyTorch's standard GELU,
    so models pretrained with it must use it at inference too.
    None for standard GELU / ReLU (not encoded in the name).

pe_scope : str | None
    Scope tag for the ViT-PE family (Vision Transformer with Position
    Embedding variants from Meta). Parsed from ``_pe_{scope}_`` tokens.
      "Lang"     → vit_pe_lang  — language-aligned PE for cross-modal tasks
      "Core"     → vit_pe_core  — core vision PE
      "Spatial"  → vit_pe_spatial — spatial-aware PE
    None for all other families.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VARIANT FLAGS  (parsed from the variant, all bool)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

is_distilled : bool
    True when the model was trained with knowledge distillation from a
    teacher model. Detected via ``_distilled_`` or ``_dist_`` tokens in
    the *variant* (not the config, where "dist" means distillation recipe).
    Example: deit_tiny_distilled_patch16_224 — DeiT trained with
    hard-label distillation from a RegNet teacher (Touvron et al., 2021).

is_pruned : bool
    True for structured-pruned variants that have had channels or heads
    removed to reduce FLOPs, typically with slight accuracy trade-offs.
    Detected via ``_pruned_`` token.
    Example: efficientnet_b2_pruned, ecaresnet50d_pruned.

is_legacy : bool
    True for models with a ``legacy_`` prefix. These are older
    implementations kept for backwards compatibility — usually ported from
    third-party checkpoints (Cadene, PyTorch-Image-Models legacy zoo)
    before timm adopted its current naming convention.
    Example: legacy_seresnet34, legacy_xception, legacy_seresnext50_32x4d.

is_gap : bool
    True when ``_gap_`` appears in the variant name, confirming the model
    uses Global Average Pooling as its classification head. Redundant with
    head_type == "GAP" but provided as a direct boolean for convenience.

uses_rmlp : bool
    True for MaxViT / CoAtNet variants that use a MLP Log-CPB position
    bias — a continuous log-coordinate relative position bias motivated
    by SwinV2, implemented as a small MLP rather than a lookup table.
    This allows the position bias to generalise across resolutions at
    inference time. Detected via ``_rmlp_`` token.
    Example: maxvit_rmlp_base_rw_224, maxxvitv2_rmlp_base_rw_224,
             coatnet_rmlp_2_rw_224.

uses_rw : bool
    True for timm-specific re-implementation variants with modelling
    adjustments made to favour PyTorch eager-mode execution. These were
    created by Ross Wightman while training initial reproductions of
    models whose original implementations targeted TensorFlow/JAX. They
    may have minor architectural differences from the paper reference
    (e.g. different stem, normalization placement, or head) to improve
    throughput and numerical stability in PyTorch. Detected via ``_rw_``
    token. Contrast with ``_tf_`` models, which exactly match the
    original TensorFlow weights.
    Example: maxxvitv2_rmlp_base_rw_224, coatnet_2_rw_224.

uses_cr : bool
    True for SwinV2 cross-resolution variants that were pretrained at one
    resolution and support fine-tuning at a different resolution via
    continuous relative position bias. Detected via ``_cr_`` token.
    Example: swinv2_cr_tiny_ns_224.

uses_ns : bool
    True for SwinV2-CR "norm-per-stage" variants that apply LayerNorm at
    the end of every stage (rather than only at the final output). This
    produces NCHW tensor layout at stage outputs instead of the standard
    NHWC layout. Detected via ``_ns_`` token.
    Example: swinv2_cr_tiny_ns_224.

uses_abswin : bool
    True for Hiera variants that use absolute window position embeddings
    in addition to the standard masked-unit attention. Introduced in the
    Hiera-AbsWin paper for improved fine-tuning at higher resolutions.
    Detected via ``_abswin_`` token.
    Example: hiera_small_abswin_256.

uses_quickgelu : bool
    True when ``_quickgelu_`` is in the variant. Mirrors the ``activation``
    field as a direct boolean flag.

uses_ts : bool
    True for timm BYOBNet variants with a **tiered 3-layer stem** —
    three stacked 3×3 convolutions without pooling, paired with SiLU
    activations, replacing the standard single 7×7 conv stem. This
    design improves gradient flow and is common in timm's BYOB
    (Bring-Your-Own-Blocks) re-implementations.
    Detected when a numeric token is immediately followed by "ts"
    (e.g. "33ts" in gcresnet33ts, "26ts" in seresnext26ts, "26ts" in
    eca_halonext26ts). The `ts` suffix stands for **Tiered Stem**.

uses_aa : bool
    True for variants with anti-aliasing in the downsampling stem or
    strided layers (Zhang, 2019). Detected via ``_aa_`` token.
    Example: resnetaa101d, darknetaa53.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PRETRAINING  (parsed from the part after the dot)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

pretrain_config : str
    The raw pretrain config string exactly as it appears after the dot.
    Provided for reference and debugging.
    Example: "sw_in12k_ft_in1k", "apple_pt", "fb_dist_in1k".

pretrain_org : str | None
    Organisation or individual responsible for training the checkpoint.
    Parsed by matching tokens against a priority-ordered lookup (longest
    keys first to prevent "ra" shadowing "ra4").
    Organisations: "Meta" (fb), "Apple", "OpenAI", "Microsoft" (ms/msft),
      "Google" / "Google/TF" (goog/tf), "NVIDIA" (nv), "Baidu" (bd),
      "DeepMind" (dm), "Naver", "SAIL", "MIIL", "Snap", "CVNets/Apple".
    Community / recipe authors: "Ross Wightman" (sw), "Chris Ha" (ch),
      "pycls" (Facebook Research pycls team), "MXNet community" (mx),
      "NaverAI" (nav), "Baidu/PaddlePaddle" (paddle).
    timm training recipes: "timm A1H recipe" (a1h), "timm A1 recipe" (a1),
      "timm A2/A3 recipe", "timm RA / RA2 / RA3 / RA4 recipe",
      "timm SBB / SBB-v2 recipe", "timm C1 / C2 / C2-NS recipe",
      "timm AH / D recipe", "timm B1K / B2K recipe",
      "Bag-of-Tricks recipe" (bt), "torchvision / torchvision-v2" (tv/tv2),
      "AGC recipe", "LAMB recipe", "RMSProp recipe".
    None when no matching token is found.

pretrain_dataset : str | None
    The primary dataset used for pretraining (before any fine-tuning).
    The "pre-ft" portion of the config string (before the first "_ft_"
    marker) is searched, so a fine-tune dataset is never accidentally
    returned here.
    Recognised datasets:
      ImageNet variants: "ImageNet-1K" (in1k), "ImageNet-21K" (in21k),
        "ImageNet-22K" (in22k), "ImageNet-12K" (in12k),
        "ImageNet-Winter-21K" (inw21k — used for MViTv2 weakly supervised).
      Web-scale: "LAION-2B" (laion2b), "LAION-400M" (laion400m),
        "LAION-Aesthetic" (laiona), "WebLI" (webli), "CC-12M" (cc12m),
        "DataComp-1B" (datacomp1b), "DataComp-XL" (datacompxl).
      CLIP-specific: "MetaCLIP" (metaclip), "MetaCLIP-2" (metaclip2),
        "DFN-2B" / "DFN-5B" / "DFN-DR-2B" (dfn2b / dfn5b / dfndr2b),
        "Merged-2B" (merged2b), "mCLIP-2" (mclip2).
      Self-supervised: "LVD-142M" (lvd142m — DINOv2 curated),
        "LVD-1689M" (lvd1689m — DINOv3), "SAT-493M" (sat493m).
      Google / Meta internal: "JFT" (jft), "PaLI" / "PaLI-2" (pali/pali2),
        "OGVL" (ogvl), "M-3B" / "M-3.8B" (m30m / m38m),
        "Instagram-1B" (ig1b), "SA-1B" (sa1b — SAM segments).
      Other: "YFCC-15M" / "YFCC-100M" (yfcc15m / yfcc100m),
             "Green-dataset" (green — HardCoreNAS efficiency subset).
    None when no recognised dataset token is found.

pretrain_dataset_size : str | None
    A scale qualifier that further describes the dataset or data mixture,
    where the dataset name alone is not fully specific.
    Examples:
      "400M"  → MetaCLIP trained on 400M image-text pairs
                (vit_large_patch14_clip_224.metaclip_400m)
      "s39b"  → DFN-2B subset of 39B samples
                (vit_large_patch14_clip_224.dfn2b_s39b)
      "2.1T"  → SAM2 pretrained on 2.1 trillion tokens  (2pt1)
      "2.5T"  → MetaCLIP / EVA pretrained on 2.5T tokens (2pt5)
      "6M"    → NextViT SSLD distilled on 6M unlabelled images
    None when no scale qualifier is present.

pretrain_method : str | None
    The training objective / self-supervised method used for pretraining.
    Parsed by scanning all config tokens for method keywords.
    Recognised methods (priority order, most specific first):
      "SigLIP"       → Sigmoid Loss for Language–Image Pretraining
                       (Zhai et al., 2023). Contrastive but sigmoid-based.
      "mCLIP"        → Mobile CLIP (Apple, multi-resolution CLIP variant)
      "CLIP"         → Contrastive Language–Image Pretraining (Radford 2021)
      "FCMAE"        → Fully Convolutional MAE (ConvNeXtV2)
      "MAE"          → Masked Autoencoder (He et al., 2021)
      "MIM"          → Masked Image Modelling (BEiT-style)
      "I-JEPA"       → Image Joint Embedding Predictive Architecture
      "DINOv3"       → DINOv3 self-distillation
      "DINOv2"       → DINOv2 self-distillation with registers
      "DINO"         → DINO self-distillation (Caron et al., 2021)
      "SWSL"         → Semi-Weakly Supervised Learning (Meta/Instagram)
      "WSL"          → Weakly Supervised Learning (Meta/Instagram)
      "SSL"          / "SSLD" → Generic / Baidu self-supervised distillation
      "SEER"         → Self-supERvised pretraining (Meta RegNet)
      "NadAMuon"     → Nesterov-adaptive Muon optimizer recipe
      "AugReg"       → Augmentation + Regularisation pretraining recipe
                       ("How to train your ViT", Steiner et al., 2021)
      "Distillation" → Knowledge distillation from a teacher (dist/distilled)
      "Pretraining"  → Generic pretrain-only checkpoint (pt token)
      "LTT"          → Label-Transfer Training
    None for standard supervised training on ImageNet (the default, not
    explicitly encoded in the config).

pretrain_ft : str | None
    Dataset used for supervised fine-tuning after pretraining.
    Detected by searching for a ``_ft_`` marker in the config and
    identifying the dataset token that follows it. Also catches implicit
    two-stage patterns like ``in22k_in1k`` (pretrain on IN-22K, evaluate
    on IN-1K) without an explicit "ft_" marker.
    Examples:
      "ImageNet-1K"   → most common fine-tune target
      "ImageNet-22K"  → intermediate fine-tune (EVA, ConvNeXtV2 recipes)
      "ImageNet-12K"  → timm SBB / SW recipes
    None for models that are either pretrain-only or directly pretrained
    on the target dataset without a separate fine-tune stage.

pretrain_resolution : int | None
    Input resolution used during pretraining, when different from the
    architectural resolution encoded in the variant.
    Parsed from explicit ``_r{N}_`` tokens (highest priority), then from
    a trailing ``_{N}`` in the pre-ft portion of the config.
    Examples:
      efficientnet_b0.ra4_e3600_r224_in1k  → 224  (explicit r224)
      mobilenetv4_conv_medium.e250_r384_in12k_ft_in1k → 384 (explicit r384)
      csatv2_21m.sw_r640_in1k             → 640  (explicit r640)
      sam2_hiera_tiny.fb_r896_2pt1        → 896  (explicit r896)
    None when the pretrain resolution is the same as the architectural
    resolution (the common case) or cannot be parsed.

pretrain_ft_resolution : int | None
    Input resolution used during the fine-tuning stage, when the model
    was fine-tuned at a different (usually higher) resolution than it was
    pretrained at. Parsed from a trailing ``_{N}`` after the ``_ft_``
    marker.
    Examples:
      convnext_tiny.fb_in22k_ft_in1k_384         → 384
      convnextv2_huge.fcmae_ft_in22k_in1k_384    → 384
      rdnet_large.nv_in1k_ft_in1k_384            → 384
    None when fine-tuning was done at the same resolution.

pretrain_epochs : int | None
    Number of training epochs encoded in the config string. Parsed from
    ``_e{N}_`` tokens (timm convention) and ``_{N}ep_`` tokens (FlexiViT
    convention). Only the training-phase epoch count is captured.
    Examples:
      hiera_small_abswin_256.sbb2_e200_in12k    → 200
      # training steps for MobileNet-style short-schedule recipes, not epochs
      efficientnet_b0.ra4_e3600_r224_in1k       → 3600
      flexivit_small.1200ep_in1k                → 1200
      mobilenetv4_conv_medium.e250_r384_in12k   → 250
    None when not encoded.

pretrain_tokens : str | None
    Data token-count qualifier used in SAM2 and similar models that report
    the number of training tokens rather than epochs. Parsed from
    ``_{N}pt{M}_`` tokens (where pt = "point", i.e. decimal separator).
    Examples:
      sam2_hiera_tiny.fb_r896_2pt1  → "2pt1"  (2.1 trillion tokens)
      (2pt5 → 2.5 trillion tokens, used in some MetaCLIP variants)
    None when not encoded.

pretrain_aug : str | None
    Data augmentation or regularisation regime used during pretraining,
    when explicitly encoded in the config. These tags come from the config
    side (after the dot) and describe the augmentation scheme chosen by
    the original authors — distinct from the timm training recipe org
    tokens (ra, ra2, ra3, ra4, sbb …) which appear on the same side.
      "AugReg"       → Augmentation + Regularization for ViT training.
                       From "How to Train Your ViT?" (Steiner et al.,
                       2021, arXiv:2106.10270). Combines RandAugment,
                       Mixup, and stochastic depth regularization; the
                       key finding is that AugReg on ImageNet-21K can
                       match models trained on 10× more data. Weights
                       ported from Google's JAX/TPU training.
                       Token: "augreg".
      "AdvProp"      → Adversarial Examples Improve Image Recognition
                       (Xie et al., 2019, arXiv:1911.09665). Treats
                       adversarial examples as additional training data
                       with a separate auxiliary BatchNorm to handle their
                       different distribution. Trained in TensorFlow/TPU
                       by Google Brain, ported to PyTorch by Ross Wightman.
                       Token: "ap".
      "NoisyStudent"  → Self-training with Noisy Student (Xie et al.,
                       2020, arXiv:1911.04252). Semi-supervised learning:
                       a teacher EfficientNet is trained on ImageNet-1K,
                       then used to pseudo-label 130M images from JFT-300M;
                       a larger student is trained on both with noise
                       (dropout, stochastic depth, RandAugment). Hence
                       these configs always include "jft" alongside "ns".
                       Token: "ns" (always paired with "jft" in config).
      "AutoAugment"  → AutoAugment (Cubuk et al., 2019). A learned
                       augmentation policy found by reinforcement learning
                       on a proxy task. Token: "aa".
      "RandAugment"  → RandAugment (Cubuk et al., 2020). Randomly applies
                       N augmentation transforms at magnitude M. Token: "ra"
                       — note this overlaps with the timm RA training recipe
                       org token (ra, ra2 …); the distinction is context:
                       org tokens precede dataset tokens, aug tokens follow.
    None when no augmentation qualifier is encoded.

pretrain_i18n : bool
    True for multilingual WebLI variants (``_i18n`` suffix). These SigLIP
    models were trained on the full multilingual WebLI corpus (109 languages)
    rather than the English-only subset, enabling cross-lingual image–text
    alignment.
    Example: vit_so400m_patch16_siglip_gap_256.webli_i18n

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TIMM FIELDS  (requires timm to be installed)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

num_parameters : int | None
    Total number of trainable parameters, obtained by instantiating the
    model with ``timm.create_model(model_name, pretrained=False)`` and
    summing ``p.numel()`` over all parameters. No weights are downloaded.
    None when timm is not installed or the model name is not recognised.

latent_dim : int | None
    Dimensionality of the output embedding — the feature vector produced by
    ``model.forward_features()`` (before the classification head), which is
    what you store in embedding datasets or pass to downstream tasks.
    Read from ``model.num_features`` (the timm standard), falling back
    through ``model.embed_dim``, ``model.hidden_size``, ``model.d_model``.
    Common values by ViT size:
      - ViT-Tiny:      192
      - ViT-Small:     384
      - ViT-Base:      768
      - ViT-Large:     1024
      - ViT-Huge:      1280
      - ViT-Giant:     1408
      - ViT-Gigantic:  1664
      - SO400M:        1152
    None when timm is not installed, or when ``use_timm=False`` is passed
    to ``get_model_metadata()``, or when the model cannot be instantiated.
"""

import logging
import re

logger = logging.getLogger(__name__)


# ===========================================================================
# FAMILY RULES  (prefix → family, model_version)
# ===========================================================================
# MACRO FAMILY — coarse architectural category, independent of family.
# Groups families by the dominant computational motif of their backbone.
# ===========================================================================

_MACRO_FAMILY_MAP: dict[str, str] = {
    # ── Pure Vision Transformers ──────────────────────────────────────────
    # Self-attention over patch tokens, no conv stages in backbone
    'ViT': 'Vision Transformer',
    'DeiT': 'Vision Transformer',
    'CaiT': 'Vision Transformer',
    'BEiT': 'Vision Transformer',
    'EVA': 'Vision Transformer',
    'FlexiViT': 'Vision Transformer',
    'AIMv2': 'Vision Transformer',
    'ViTamin': 'Vision Transformer',
    'SAM-ViT': 'Vision Transformer',
    'NaFlexViT': 'Vision Transformer',
    'CSAFormer': 'Vision Transformer',
    # ── Hierarchical Vision Transformers ─────────────────────────────────
    # Multi-stage with shifted / local windows or hierarchical pooling
    'Swin': 'Hierarchical Vision Transformer',
    'MViT': 'Hierarchical Vision Transformer',
    'Hiera': 'Hierarchical Vision Transformer',
    'SAM2-Hiera': 'Hierarchical Vision Transformer',
    'PVT': 'Hierarchical Vision Transformer',
    'Twins': 'Hierarchical Vision Transformer',
    'DaViT': 'Hierarchical Vision Transformer',
    'GCViT': 'Hierarchical Vision Transformer',
    'FocalNet': 'Hierarchical Vision Transformer',
    # ── Cross-Attention / Token Mixing Transformers ───────────────────────
    # Attention across scales, cross-image, or novel token-mixing operators
    'CrossViT': 'Vision Transformer',
    'XCiT': 'Vision Transformer',
    'TNT': 'Vision Transformer',
    'VOLO': 'Vision Transformer',
    'VisFormer': 'Vision Transformer',
    'Sequencer2D': 'Vision Transformer',
    # ── Hybrid CNN-Transformer ────────────────────────────────────────────
    # Convolutional stem/stages mixed with transformer blocks
    'CoAtNet': 'Hybrid CNN-Transformer',
    'CoAtNeXt': 'Hybrid CNN-Transformer',
    'MaxViT': 'Hybrid CNN-Transformer',
    'CoaT': 'Hybrid CNN-Transformer',
    'TinyViT': 'Hybrid CNN-Transformer',
    'NextViT': 'Hybrid CNN-Transformer',
    'FastViT': 'Hybrid CNN-Transformer',
    'MobileViT': 'Hybrid CNN-Transformer',
    'EfficientViT': 'Hybrid CNN-Transformer',  # MSRA EfficientViT (conv+attn)
    'LeViT': 'Hybrid CNN-Transformer',
    'NestNet': 'Hybrid CNN-Transformer',
    'PiT': 'Hybrid CNN-Transformer',
    'BotNet': 'Hybrid CNN-Transformer',
    'HaloNet': 'Hybrid CNN-Transformer',
    'ShViT': 'Hybrid CNN-Transformer',
    'SwiftFormer': 'Hybrid CNN-Transformer',
    'EfficientFormer': 'Hybrid CNN-Transformer',
    'RepViT': 'Hybrid CNN-Transformer',
    # ── MetaFormer / ConvFormer / PoolFormer ─────────────────────────────
    # Abstract token-mixer framework; token mixer can be conv, pool, or attn
    'MetaFormer': 'MetaFormer',
    # ── Pure MLP ─────────────────────────────────────────────────────────
    'MLP-Mixer': 'MLP',
    'ResMLP': 'MLP',
    'gMixer': 'MLP',
    # ── ConvNeXt family ──────────────────────────────────────────────────
    # Modernised pure-conv architectures inspired by ViT design choices
    'ConvNeXt': 'Convolutional',
    'ConViT': 'Hybrid CNN-Transformer',  # conv-attn attention gate
    'InceptionNeXt': 'Convolutional',
    'RDNet': 'Convolutional',
    # ── Classic / Standard CNNs ───────────────────────────────────────────
    'ResNet': 'Convolutional',
    'ResNeXt': 'Convolutional',
    'Wide-ResNet': 'Convolutional',
    'ResNeSt': 'Convolutional',
    'Res2Net': 'Convolutional',
    'Res2NeXt': 'Convolutional',
    'SENet': 'Convolutional',
    'DenseNet': 'Convolutional',
    'VGG': 'Convolutional',
    'Inception': 'Convolutional',
    'InceptionResNet': 'Convolutional',
    'NFNet': 'Convolutional',
    'HRNet': 'Convolutional',
    'DLA': 'Convolutional',
    'DPN': 'Convolutional',
    'RegNet': 'Convolutional',
    'EfficientNet': 'Convolutional',
    'HGNet': 'Convolutional',
    'HGNetV2': 'Convolutional',
    'TResNet': 'Convolutional',
    'MobileNet': 'Convolutional',
    'MobileOne': 'Convolutional',
    'MixNet': 'Convolutional',
    'MnasNet': 'Convolutional',
    'FBNet': 'Convolutional',
    'LCNet': 'Convolutional',
    'RexNet': 'Convolutional',
    'GENet': 'Convolutional',
    'GhostNet': 'Convolutional',
    'RepGhostNet': 'Convolutional',
    'RepVGG': 'Convolutional',
    'EdgeNeXt': 'Convolutional',
    'FasterNet': 'Convolutional',
    'VoVNet': 'Convolutional',
    'NASNet': 'Convolutional',
    'HardCoreNAS': 'Convolutional',
    'Xception': 'Convolutional',
    'TinyNet': 'Convolutional',
    'SPNASNet': 'Convolutional',
    'CSPNet': 'Convolutional',
    'DarkNet': 'Convolutional',
    'StarNet': 'Convolutional',
    # ── State Space Models ────────────────────────────────────────────────
    # Selective / gated state-space sequence models for vision
    'MambaOut': 'State Space Model',
    # ── CLIP / Contrastive image encoders ─────────────────────────────────
    # Architectures primarily used as CLIP image towers — kept in their
    # natural structural category (ViT or CNN); not a separate macro_family.
}


def _get_macro_family(family: str) -> str | None:
    """Return the coarse architectural category for a parsed family name."""
    return _MACRO_FAMILY_MAP.get(family)


# ===========================================================================
# model_version = architecture *generation* tag only (v2, III, RS …).
# All other variant info is parsed separately below.
# More specific prefixes MUST come before more general ones.

_FAMILY_RULES: list[tuple[str, str, str | None]] = [
    ('aimv2', 'AIMv2', 'v2'),
    ('sam2_hiera', 'SAM2-Hiera', '2'),
    ('eva02', 'EVA', 'v2'),
    ('eva', 'EVA', None),
    ('beitv2', 'BEiT', 'v2'),
    ('beit3', 'BEiT', 'v3'),
    ('beit', 'BEiT', None),
    # ViT — many capacity prefixes to prevent fall-through to "vit_" catch-all
    ('vit_so400m', 'ViT', None),
    ('vit_so150m', 'ViT', None),
    ('vit_gigantic', 'ViT', None),
    ('vit_giantopt', 'ViT', None),
    ('vit_huge_plus', 'ViT', None),
    ('vit_huge', 'ViT', None),
    ('vit_giant', 'ViT', None),
    ('vit_large', 'ViT', None),
    ('vit_base', 'ViT', None),
    ('vit_mediumd', 'ViT', None),
    ('vit_medium', 'ViT', None),
    ('vit_betwixt', 'ViT', None),
    ('vit_small', 'ViT', None),
    ('vit_wee', 'ViT', None),
    ('vit_pwee', 'ViT', None),
    ('vit_dwee', 'ViT', None),
    ('vit_dpwee', 'ViT', None),
    ('vit_little', 'ViT', None),
    ('vit_tiny', 'ViT', None),
    ('vit_pe', 'ViT', None),
    ('vit_intern', 'ViT', None),
    ('vit_relpos', 'ViT', None),
    ('vit_srelpos', 'ViT', None),
    ('naflexvit', 'ViT', None),
    ('vit_', 'ViT', None),
    ('deit3', 'DeiT', 'III'),
    ('deit', 'DeiT', None),
    ('swinv2', 'Swin', 'v2'),
    ('swin_s3', 'Swin', 'S3'),
    ('swin', 'Swin', None),
    ('convnextv2', 'ConvNeXt', 'v2'),
    ('convnext', 'ConvNeXt', None),
    ('caformer', 'MetaFormer', None),
    ('convformer', 'MetaFormer', None),
    ('poolformerv2', 'MetaFormer', 'v2'),
    ('poolformerv2', 'MetaFormer', 'v2'),
    ('poolformer', 'MetaFormer', None),
    ('efficientvit', 'EfficientViT', None),
    ('efficientformerv2_l', 'EfficientFormer', 'v2'),
    ('efficientformerv2', 'EfficientFormer', 'v2'),
    ('efficientformer', 'EfficientFormer', None),
    ('efficientnetv2', 'EfficientNet', 'v2'),
    ('tf_efficientnetv2', 'EfficientNet', 'v2'),
    ('tf_efficientnet', 'EfficientNet', None),
    ('efficientnet', 'EfficientNet', None),
    ('mobilenetv5', 'MobileNet', 'v5'),
    ('mobilenetv4', 'MobileNet', 'v4'),
    ('mobilenetv3', 'MobileNet', 'v3'),
    ('mobilenetv2', 'MobileNet', 'v2'),
    ('mobilenetv1', 'MobileNet', 'v1'),
    ('mobilenet_edgetpu', 'MobileNet', 'EdgeTPU'),
    ('tf_mobilenetv3', 'MobileNet', 'v3'),
    ('mobilenet', 'MobileNet', None),
    ('mobileone', 'MobileOne', None),
    ('mobilevitv2_050', 'MobileViT', 'v2'),  # 0.50x channel width
    ('mobilevitv2_075', 'MobileViT', 'v2'),  # 0.75x
    ('mobilevitv2_100', 'MobileViT', 'v2'),  # 1.00x
    ('mobilevitv2_125', 'MobileViT', 'v2'),  # 1.25x
    ('mobilevitv2_150', 'MobileViT', 'v2'),  # 1.50x
    ('mobilevitv2_175', 'MobileViT', 'v2'),  # 1.75x
    ('mobilevitv2_200', 'MobileViT', 'v2'),  # 2.00x
    ('mobilevitv2', 'MobileViT', 'v2'),
    ('mobilevit_xxs', 'MobileViT', None),  # extra-extra-small
    ('mobilevit_xs', 'MobileViT', None),  # extra-small
    ('mobilevit_s', 'MobileViT', None),  # small
    ('mobilevit', 'MobileViT', None),
    ('resnetv2', 'ResNet', 'v2'),
    ('resnetrs', 'ResNet', 'RS'),
    ('resnetaa', 'ResNet', 'AA'),
    ('resnetblur', 'ResNet', 'Blur'),
    ('resnext', 'ResNeXt', None),
    ('resnest', 'ResNeSt', None),
    ('resnet50_clip', 'ResNet', 'CLIP'),
    ('resnet50x', 'ResNet', 'CLIP-scaled'),
    ('resnet101_clip', 'ResNet', 'CLIP'),
    ('resnet', 'ResNet', None),
    ('wide_resnet', 'Wide-ResNet', None),
    ('regnety', 'RegNet', 'Y'),
    ('regnetx', 'RegNet', 'X'),
    ('regnetv', 'RegNet', 'V'),
    ('regnetz', 'RegNet', 'Z'),
    ('maxxvitv2', 'MaxViT', 'v2'),
    ('maxxvit', 'MaxViT', None),
    ('maxvit', 'MaxViT', None),
    ('vitamin', 'ViTamin', None),
    ('hiera', 'Hiera', None),
    ('mvitv2', 'MViT', 'v2'),
    ('mvit', 'MViT', None),
    ('coatnet', 'CoAtNet', None),
    ('coatnext', 'CoAtNeXt', None),
    ('coat_lite', 'CoaT', None),
    ('coat', 'CoaT', None),
    ('davit', 'DaViT', None),
    ('dm_nfnet', 'NFNet', None),
    ('eca_nfnet', 'NFNet', None),
    ('nfnet', 'NFNet', None),
    ('nf_resnet', 'ResNet', 'NF'),
    ('nf_regnet', 'RegNet', 'NF'),
    ('mambaout', 'MambaOut', None),
    ('samvit', 'SAM-ViT', None),
    ('xcit', 'XCiT', None),
    ('csatv2', 'CSAFormer', 'v2'),
    ('csat', 'CSAFormer', None),
    ('gc_efficientnetv2', 'EfficientNet', 'v2-GC'),
    ('gcvit', 'GCViT', None),
    ('focalnet', 'FocalNet', None),
    ('fastvit', 'FastViT', None),
    ('shvit', 'ShViT', None),
    ('starnet', 'StarNet', None),
    ('nextvit', 'NextViT', None),
    ('flexivit', 'FlexiViT', None),
    ('dpn', 'DPN', None),
    ('dla', 'DLA', None),
    ('pvt_v2', 'PVT', 'v2'),
    ('pvt', 'PVT', None),
    ('twins', 'Twins', None),
    ('sequencer2d', 'Sequencer2D', None),
    ('mixer', 'MLP-Mixer', None),
    ('gmlp', 'gMLP', None),
    ('gmixer', 'gMixer', None),
    ('resmlp', 'ResMLP', None),
    ('vgg', 'VGG', None),
    ('densenet', 'DenseNet', None),
    ('hrnet', 'HRNet', None),
    ('hgnetv2', 'HGNet', 'v2'),
    ('hgnet', 'HGNet', None),
    ('xception', 'Xception', None),
    ('inception_resnet', 'InceptionResNet', None),
    ('inception_next', 'InceptionNeXt', None),
    ('inception', 'Inception', None),
    ('nasnet', 'NASNet', None),
    ('pnasnet', 'PNASNet', None),
    ('repvgg', 'RepVGG', None),
    ('repvit', 'RepViT', None),
    ('repghostnet', 'RepGhostNet', None),
    ('convmixer', 'ConvMixer', None),
    ('convit', 'ConViT', None),
    ('crossvit', 'CrossViT', None),
    ('cait', 'CaiT', None),
    ('volo', 'VOLO', None),
    ('pit', 'PiT', None),
    ('tnt', 'TNT', None),
    ('visformer', 'VisFormer', None),
    ('fasternet', 'FasterNet', None),
    ('ghostnetv3', 'GhostNet', 'v3'),
    ('ghostnetv2', 'GhostNet', 'v2'),
    ('ghostnet', 'GhostNet', None),
    ('eca_botnext', 'BotNet', 'ECA'),
    ('eca_halonext', 'HaloNet', 'ECA'),
    ('eca_resnet', 'ResNet', 'ECA'),
    ('eca_resnext', 'ResNeXt', 'ECA'),
    ('ecaresnet', 'ResNet', 'ECA'),
    ('seresnet', 'ResNet', 'SE'),
    ('seresnext', 'ResNeXt', 'SE'),
    ('sehalonet', 'HaloNet', 'SE'),
    ('sebotnet', 'BotNet', 'SE'),
    ('gcresnet', 'ResNet', 'GC'),
    ('gcresnext', 'ResNeXt', 'GC'),
    ('tiny_vit', 'TinyViT', None),
    ('res2net', 'Res2Net', None),
    ('res2next', 'Res2NeXt', None),
    ('tresnet', 'TResNet', None),
    ('swiftformer', 'SwiftFormer', None),
    ('levit', 'LeViT', None),
    ('lambda_resnet', 'ResNet', 'Lambda'),
    ('halonet', 'HaloNet', None),
    ('halo2botnet', 'HaloNet', None),
    ('halobotnet', 'HaloNet', None),
    ('lamhalobotnet', 'HaloNet', None),
    ('bat_resnext', 'ResNeXt', 'BAT'),
    ('botnet', 'BotNet', None),
    ('cspdarknet', 'DarkNet', 'CSP'),
    ('cspresnext', 'ResNeXt', 'CSP'),
    ('cspresnet', 'ResNet', 'CSP'),
    ('cs3sedarknet', 'DarkNet', 'CS3-SE'),
    ('cs3darknet', 'DarkNet', 'CS3'),
    ('cs3edgenet', 'EdgeNet', 'CS3'),
    ('cs3se_edgenet', 'EdgeNet', 'CS3-SE'),
    ('darknetaa', 'DarkNet', 'AA'),
    ('darknet', 'DarkNet', None),
    ('gernet', 'GENet', None),
    ('haloregnetz', 'RegNet', 'HaloZ'),
    ('senet', 'SENet', None),
    ('spnasnet', 'SPNASNet', None),
    # test_* variants — timm internal regression / CI models,
    # mapped to the family they are testing
    ('test_byobnet', 'BYOBNet', 'test'),
    ('test_convnext', 'ConvNeXt', 'test'),
    ('test_efficientnet_evos', 'EfficientNet', 'test-EvoS'),
    ('test_efficientnet_gn', 'EfficientNet', 'test-GN'),
    ('test_efficientnet_ln', 'EfficientNet', 'test-LN'),
    ('test_efficientnet', 'EfficientNet', 'test'),
    ('test_nfnet', 'NFNet', 'test'),
    ('test_resnet', 'ResNet', 'test'),
    ('test_vit', 'ViT', 'test'),
    ('selecsls', 'SelecSLS', None),
    ('mixnet', 'MixNet', None),
    ('tf_mixnet', 'MixNet', None),
    ('skresnext', 'ResNeXt', 'SK'),
    ('skresnet', 'ResNet', 'SK'),
    ('edgenext', 'EdgeNeXt', None),
    ('rdnet', 'RDNet', None),
    ('rexnetr', 'RexNet', 'R'),
    ('rexnet', 'RexNet', None),
    ('fbnetv3', 'FBNet', 'v3'),
    ('fbnetc', 'FBNet', 'C'),
    ('semnasnet', 'MnasNet', 'SE'),
    ('mnasnet', 'MnasNet', None),
    ('lcnet', 'LCNet', None),
    ('hardcorenas', 'HardCoreNAS', None),
    ('nest', 'NestNet', None),
    ('ese_vovnet', 'VoVNet', 'ESE'),
    ('tinynet', 'TinyNet', None),
    ('csp', 'CSP-Net', None),
    # legacy_ prefix is stripped before matching; handled via is_legacy flag
]

# ===========================================================================
# SIZE RULES  (token → human label)
# ===========================================================================
# Matched as whole underscore-delimited tokens. More specific first.
_SIZE_RULES: list[tuple[str, str]] = [
    ('3b', '3B'),
    ('1b', '1B'),
    ('xxs', 'XXSmall'),
    ('xs', 'XSmall'),
    ('xxlarge', 'XXLarge'),
    ('xlarge', 'XLarge'),
    ('gigantic', 'Gigantic'),
    ('giantopt', 'GiantOpt'),
    ('huge_plus', 'Huge+'),
    ('huge', 'Huge'),
    ('giant', 'Giant'),
    ('so400m', 'SO400M'),
    ('so150m2', 'SO150M2'),
    ('so150m', 'SO150M'),
    ('large2', 'Large2'),
    ('large', 'Large'),
    ('base_plus', 'Base+'),
    ('base', 'Base'),
    ('mediumd', 'MediumD'),
    ('medium', 'Medium'),
    ('betwixt', 'Betwixt'),
    ('small_plus', 'Small+'),
    ('small', 'Small'),
    ('little', 'Little'),
    ('wee', 'Wee'),
    ('pwee', 'pWee'),
    ('dwee', 'dWee'),
    ('dpwee', 'dpWee'),
    ('tiny', 'Tiny'),
    ('femto', 'Femto'),
    ('atto', 'Atto'),
    ('pico', 'Pico'),
    ('nano', 'Nano'),
    ('micro', 'Micro'),
    ('mini', 'Mini'),
    ('zepto', 'Zepto'),
    # EfficientNet B-series
    ('b8', 'B8'),
    ('b7', 'B7'),
    ('b6', 'B6'),
    ('b5', 'B5'),
    ('b4', 'B4'),
    ('b3', 'B3'),
    ('b2', 'B2'),
    ('b1', 'B1'),
    ('b0', 'B0'),
    # EfficientFormer latency tiers (l1=fastest, l7=largest)
    ('l1', 'L1'),
    ('l3', 'L3'),
    ('l7', 'L7'),
    # EfficientFormerV2 large (single-letter — matched via family prefix rule)
    # MobileViT single-letter sizes (matched via family prefix rule above)
    # ViT-InternImage param size (matched via family prefix rule)
    # MobileViTv2 decimal-multiplier sizes — extract from variant numeric suffix
    ('050', '0.50x'),
    ('075', '0.75x'),
    ('100', '1.00x'),
    ('125', '1.25x'),
    ('150', '1.50x'),
    ('175', '1.75x'),
    ('200', '2.00x'),
    # EVA enormous
    ('enormous', 'Enormous'),
    # ViT xsmall
    ('xsmall', 'XSmall'),
    # MobileViT / MobileViTv2 decimal-multiplier sizes (050=0.50x, etc.)
    ('050', '0.50x'),
    ('075', '0.75x'),
    ('100', '1.00x'),
    ('125', '1.25x'),
    ('150', '1.50x'),
    ('175', '1.75x'),
    ('200', '2.00x'),
    # ViT-InternImage param size
    ('300m', '300M'),
    # EfficientNet L/M/S/XL
    ('xl', 'XL'),
    ('el', 'EL'),
    ('em', 'EM'),
    ('es', 'ES'),
    ('lite4', 'Lite4'),
    ('lite3', 'Lite3'),
    ('lite2', 'Lite2'),
    ('lite1', 'Lite1'),
    ('lite0', 'Lite0'),
    # MobileViT extra-small codes
    ('xxs', 'XXSmall'),
    ('xs', 'XSmall'),
    # MetaFormer / PoolFormer stageconfig codes (sN=mN=bN: small/medium/base)
    ('s12', 'S12'),
    ('s18', 'S18'),
    ('s24', 'S24'),
    ('s36', 'S36'),
    ('m36', 'M36'),
    ('m48', 'M48'),
    ('b36', 'B36'),
    ('b48', 'B48'),
    # MambaOut named sizes
    ('kobe', 'Kobe'),
    # TinyNet a–e
    ('tinynet_a', 'A'),
    ('tinynet_b', 'B'),
    ('tinynet_c', 'C'),
    ('tinynet_d', 'D'),
    ('tinynet_e', 'E'),
    # CoaT-Lite
    ('lite', 'Lite'),
    # Tiny-ViT param sizes
    ('5m', '5M'),
    ('11m', '11M'),
    ('21m', '21M'),
    # ViTamin named sizes
    ('xlarge', 'XLarge'),
    # AIMv2 / ViTamin named
    ('huge', 'Huge'),
]

# ===========================================================================
# HEAD TYPE RULES
# ===========================================================================
_HEAD_RULES: list[tuple[str, str]] = [
    ('clsgap', 'CLS+GAP'),
    ('gap', 'GAP'),
    ('cls', 'CLS'),
]

# ===========================================================================
# POSITIONAL ENCODING RULES
# ===========================================================================
_PE_RULES: list[tuple[str, str]] = [
    ('rope_mixed_ape', 'RoPE-Mixed-APE'),
    ('rope_mixed', 'RoPE-Mixed'),
    ('rope_ape', 'RoPE-APE'),
    ('rope_reg', 'RoPE'),
    ('rope', 'RoPE'),
    ('relpos', 'RelPos'),
    ('srelpos', 'SRelPos'),
    ('rpn', 'RPN'),
    ('sincos', 'SinCos'),
    ('ape', 'APE'),
]

# ===========================================================================
# VALID RESOLUTION VALUES
# ===========================================================================
_VALID_RES = frozenset(
    {
        160,
        192,
        224,
        240,
        256,
        288,
        320,
        336,
        378,
        384,
        396,
        448,
        475,
        480,
        512,
        576,
        640,
        768,
        896,
        1024,
    }
)

# ===========================================================================
# HELPER
# ===========================================================================


def _tok(kw: str, s: str) -> bool:
    """True if kw appears as a whole underscore-delimited token in s."""
    return bool(re.search(rf'(?:^|_){re.escape(kw)}(?:_|$)', s))


# ===========================================================================
# VARIANT PARSER
# ===========================================================================


def _parse_variant(model_variant: str) -> dict:
    v = model_variant.lower()

    # --- legacy prefix ---
    is_legacy = v.startswith('legacy_')
    v_clean = v[len('legacy_') :] if is_legacy else v

    # --- family + model_version ---
    family = 'unknown'
    model_version: str | None = None
    for prefix, fam, ver in _FAMILY_RULES:
        if v_clean.startswith(prefix):
            family, model_version = fam, ver
            break
    if family == 'unknown':
        logger.warning('Unknown family for model variant: %s', model_variant)

    # --- size ---
    size: str | None = None
    for kw, label in _SIZE_RULES:
        if _tok(kw, v_clean):
            size = label
            break

    # --- patch size ---
    m = re.search(r'(?:^|_)patch(\d+)(?:_|$)', v_clean)
    patch_size: int | None = int(m.group(1)) if m else None
    # also catch p8 / p16 style (XCiT)
    if patch_size is None:
        m = re.search(r'(?:^|_)p(\d+)(?:_|$)', v_clean)
        if m:
            patch_size = int(m.group(1))

    # --- native input resolution ---
    input_resolution: int | None = None
    for m in re.finditer(r'(?:^|_)(\d{3,4})(?:_|$)', v_clean):
        c = int(m.group(1))
        if c in _VALID_RES:
            input_resolution = c
            break

    # --- window size (Swin, MaxViT) ---
    window_size: int | None = None
    m = re.search(r'(?:^|_)window(\d+)(?:to\d+)?(?:_|$)', v_clean)
    if m:
        window_size = int(m.group(1))

    # --- stride / stage code (CaiT s24, gMLP s16, ConvFormer s18, ShViT s4…) ---
    stride_code: str | None = None
    m = re.search(r'(?:^|_)(s\d+)(?:_|$)', v_clean)
    if m and not v_clean.startswith('swin') and not v_clean.startswith('sam'):
        stride_code = m.group(1)

    # --- depth code (ResNet 50/101/152, NFNet f0–f6, TResNet L/M…) ---
    depth_code: str | None = None
    # Try explicit _fN (NFNet), _lN (NF-L), _dN depth suffixes first
    m = re.search(r'(?:^|_)(f\d+|l\d+)(?:_|$)', v_clean)
    if m:
        depth_code = m.group(1)
    else:
        # numeric depth token (50, 101, 152, 270…)
        m = re.search(r'(?:^|_)(\d{2,3})(?:_|$)', v_clean)
        if m:
            candidate = int(m.group(1))
            # exclude patch sizes and resolutions
            if candidate not in _VALID_RES and candidate not in {4, 8, 14, 16, 32}:
                depth_code = m.group(1)

    # --- width / channel multiplier ---
    width_code: str | None = None
    # ResNeXt grouping: 32x4d
    m = re.search(r'(?:^|_)(\d+x\d+d)(?:_|$)', v_clean)
    if m:
        width_code = m.group(1)
    else:
        # HRNet wNN, MobileNet 100/125/150d style
        m = re.search(r'(?:^|_)(w\d+)(?:_|$)', v_clean)
        if m:
            width_code = m.group(1)
        else:
            # Res2Net 14w_8s / 26w_4s
            m = re.search(r'(?:^|_)(\d+w(?:_\d+s)?)(?:_|$)', v_clean)
            if m:
                width_code = m.group(1)
            else:
                # x4, x8 style (ResNet-CLIP scaled)
                m = re.search(r'(?:^|_)(x\d+)(?:_|$)', v_clean)
                if m:
                    width_code = m.group(1)
                else:
                    # MobileNet 100 / 075 / 050 style percentage multipliers
                    m = re.search(r'(?:^|_)(\d{3})(?:_|$)', v_clean)
                    if m and int(m.group(1)) not in _VALID_RES:
                        width_code = m.group(1)

    # --- head type ---
    head_type: str | None = None
    for kw, label in _HEAD_RULES:
        if _tok(kw, v_clean):
            head_type = label
            break

    # --- register tokens ---
    m = re.search(r'(?:^|_)reg(\d+)(?:_|$)', v_clean)
    num_registers: int | None = int(m.group(1)) if m else None

    # --- positional encoding ---
    positional_encoding: str | None = None
    for kw, label in _PE_RULES:
        if _tok(kw, v_clean):
            positional_encoding = label
            break

    # --- activation ---
    activation: str | None = 'QuickGELU' if _tok('quickgelu', v_clean) else None

    # --- ViT-PE scope ---
    pe_scope: str | None = None
    m = re.search(r'(?:^|_)pe_(lang|core|spatial)(?:_|$)', v_clean)
    if m:
        pe_scope = m.group(1).capitalize()

    # --- boolean flags ---
    # is_distilled: variant only — config-side dist is handled in get_model_metadata()
    is_distilled = _tok('distilled', v_clean) or _tok('dist', v_clean)
    is_pruned = _tok('pruned', v_clean)
    is_gap = _tok('gap', v_clean)
    uses_rmlp = _tok('rmlp', v_clean)
    uses_rw = _tok('rw', v_clean)
    uses_cr = _tok('cr', v_clean)
    uses_ns = _tok('ns', v_clean)
    uses_abswin = _tok('abswin', v_clean)
    uses_quickgelu = _tok('quickgelu', v_clean)
    uses_ts = bool(re.search(r'\dts(?:_|$)', v_clean))
    # uses_aa: check _aa_ OR aa-prefixed names (resnetaa, darknetaa, seresnextaa)
    uses_aa = _tok('aa', v_clean) or bool(
        re.match(r'(?:resnetaa|darknetaa|seresnextaa)', v_clean)
    )

    return {
        'family': family,
        'macro_family': _get_macro_family(family),
        'model_version': model_version,
        'size': size,
        'depth_code': depth_code,
        'width_code': width_code,
        'patch_size': patch_size,
        'input_resolution': input_resolution,
        'window_size': window_size,
        'stride_code': stride_code,
        'head_type': head_type,
        'num_registers': num_registers,
        'positional_encoding': positional_encoding,
        'activation': activation,
        'pe_scope': pe_scope,
        'is_distilled': is_distilled,
        'is_pruned': is_pruned,
        'is_legacy': is_legacy,
        'is_gap': is_gap,
        'uses_rmlp': uses_rmlp,
        'uses_rw': uses_rw,
        'uses_cr': uses_cr,
        'uses_ns': uses_ns,
        'uses_abswin': uses_abswin,
        'uses_quickgelu': uses_quickgelu,
        'uses_ts': uses_ts,
        'uses_aa': uses_aa,
    }


# ===========================================================================
# PRETRAIN CONFIG PARSER
# ===========================================================================

_DATASET_MAP: dict[str, str] = {
    'in1k': 'ImageNet-1K',
    'in21k': 'ImageNet-21K',
    'in22k': 'ImageNet-22K',
    'in12k': 'ImageNet-12K',
    'inw21k': 'ImageNet-Winter-21K',
    'ig1b': 'Instagram-1B',
    'yfcc15m': 'YFCC-15M',
    'yfcc100m': 'YFCC-100M',
    'cc12m': 'CC-12M',
    'sa1b': 'SA-1B',
    'laion2b': 'LAION-2B',
    'laion400m': 'LAION-400M',
    'laion2': 'LAION-2B',
    'laiona': 'LAION-Aesthetic',
    'datacompxl': 'DataComp-XL',
    'datacomp1b': 'DataComp-1B',
    'metaclip2': 'MetaCLIP-2',
    'metaclip': 'MetaCLIP',
    'dfn5b': 'DFN-5B',
    'dfn2b': 'DFN-2B',
    'dfndr2b': 'DFN-DR-2B',
    'merged2b': 'Merged-2B',
    'webli': 'WebLI',
    'pali2': 'PaLI-2',
    'pali': 'PaLI',
    'ogvl': 'OGVL',
    'm38m': 'M-3.8B',
    'm30m': 'M-3B',
    'jft': 'JFT',
    'lvd142m': 'LVD-142M',
    'lvd1689m': 'LVD-1689M',
    'sat493m': 'SAT-493M',
    'green': 'Green-dataset',  # HardCoreNAS green subset
    'mclip2': 'mCLIP-2',
    'seer': 'Instagram-1B',  # SEER: Instagram images (no explicit dataset tag)
    'swag': 'Instagram-3.6B',  # SWAG: weakly-supervised on ~3.6B Instagram images
    'wit': 'WIT-400M',  # OpenAI WebImageText (private, 400M image-text pairs)
}
_DS_KEYS = sorted(_DATASET_MAP, key=len, reverse=True)

_ORG_MAP: dict[str, str] = {
    'apple': 'Apple',
    'openai': 'OpenAI',
    'msft': 'Microsoft',
    'ms': 'Microsoft',
    'goog': 'Google',
    'tf': 'Google/TF',
    'naver': 'Naver',
    'paddle': 'Baidu/PaddlePaddle',
    'sail': 'SAIL',
    'miil': 'MIIL',
    'snap': 'Snap',
    'nv': 'NVIDIA',
    'bd': 'Baidu',
    'fb': 'Meta',
    'ch': 'Chris Ha',
    'dm': 'DeepMind',
    'mx': 'MXNet community',
    'gluon': 'GluonCV/MXNet',  # Apache MXNet GluonCV authors
    'nav': 'NaverAI',
    'pycls': 'pycls',
    'cvnets': 'CVNets/Apple',
    'rmsp': 'RMSProp recipe',
    'lamb': 'LAMB recipe',
    'sbb2': 'timm SBB-v2 recipe',
    'sbb': 'timm SBB recipe',
    'ra4': 'timm RA4 recipe',
    'ra3': 'timm RA3 recipe',
    'ra2': 'timm RA2 recipe',
    'ra': 'timm RA recipe',
    'a1h': 'timm A1H recipe',
    'a1': 'timm A1 recipe',
    'a2': 'timm A2 recipe',
    'a3': 'timm A3 recipe',
    'b1k': 'timm B1K recipe',
    'b2k': 'timm B2K recipe',
    'bt': 'Bag-of-Tricks recipe',
    'tv2': 'torchvision-v2',
    'tv': 'torchvision',
    'sw': 'Ross Wightman',
    'c2ns': 'timm C2-NS recipe',
    'c2': 'timm C2 recipe',
    'c1': 'timm C1 recipe',
    'ah': 'timm AH recipe',
    'agc': 'AGC recipe',
    'd': 'timm D recipe',
}
_ORG_KEYS = sorted(_ORG_MAP, key=len, reverse=True)

_METHOD_KEYWORDS: list[tuple[str, str]] = [
    ('siglip', 'SigLIP'),
    ('mclip', 'mCLIP'),
    ('clip', 'CLIP'),
    ('fcmae', 'FCMAE'),
    ('mae', 'MAE'),
    ('mim', 'MIM'),
    ('ijepa', 'I-JEPA'),
    ('dinov3', 'DINOv3'),
    ('dinov2', 'DINOv2'),
    ('dino', 'DINO'),
    ('swsl', 'SWSL'),
    ('swag', 'SWAG'),
    ('wsl', 'WSL'),
    ('ssl', 'SSL'),
    ('ssld', 'SSLD'),
    ('seer', 'SEER'),
    ('nadamuon', 'NadAMuon'),
    ('augreg', 'AugReg'),
    ('distilled', 'Distillation'),
    ('dist', 'Distillation'),
    ('pt', 'Pretraining'),
    ('ltt', 'LTT'),
]

# Dataset-size qualifiers that appear after a dataset token
_DS_SIZE_PATTERNS: list[tuple[str, str]] = [
    (r'metaclip_400m', '400M'),
    (r'metaclip_2pt5b', '2.5B'),
    (r'dfn2b_s39b', 's39b'),
    (r'2pt5', '2.5T'),
    (r'2pt1', '2.1T'),
    (r'400m', '400M'),
    (r's39b', 's39b'),
    (r'6m', '6M'),
]

# Augmentation-recipe tokens in pretrain config (not the RA training recipe)
_AUG_PATTERNS: list[tuple[str, str]] = [
    ('augreg', 'AugReg'),
    ('ap', 'AdvProp'),
    ('ns', 'NoisyStudent'),
    ('aa', 'AutoAugment'),
    ('ra', 'RandAugment'),
]

_VALID_PRETRAIN_RES = frozenset(
    {
        160,
        192,
        224,
        240,
        256,
        288,
        320,
        336,
        378,
        384,
        396,
        448,
        475,
        480,
        512,
        576,
        640,
        896,
        1024,
    }
)


def _parse_pretrain_config(pretrain_config: str) -> dict:
    cfg = pretrain_config.lower()
    tokens = re.split(r'[_]', cfg)

    # --- training method ---
    method: str | None = None
    for kw, label in _METHOD_KEYWORDS:
        if kw in tokens or any(kw in t for t in tokens):
            method = label
            break

    # --- organisation (longest key first) ---
    org: str | None = None
    for key in _ORG_KEYS:
        if key in tokens:
            org = _ORG_MAP[key]
            break
    if org is None:
        for tok in tokens:
            for key in _ORG_KEYS:
                if tok.startswith(key) and len(tok) > len(key):
                    org = _ORG_MAP[key]
                    break
            if org:
                break

    # --- fine-tune target ---
    ft_start = cfg.find('_ft_')
    ft_target: str | None = None
    if ft_start != -1:
        ft_str = cfg[ft_start + 4 :]
        for key in _DS_KEYS:
            if ft_str.startswith(key):
                ft_target = _DATASET_MAP[key]
                break
    # implicit ft patterns: in22k_in1k, in22k_ft_in22k_in1k, mae_in1k_ft_in1k
    if ft_target is None:
        for pat, tgt in [
            (r'in22k_in1k', 'ImageNet-1K'),
            (r'in21k_in1k', 'ImageNet-1K'),
            (r'in12k_in1k', 'ImageNet-1K'),
            (r'mae_in1k', 'ImageNet-1K'),
        ]:
            if re.search(pat, cfg):
                ft_target = tgt
                break

    # --- primary pretrain dataset ---
    dataset: str | None = None
    pre_ft_cfg = cfg[:ft_start] if ft_start != -1 else cfg
    pre_ft_tok = set(re.split(r'[_]', pre_ft_cfg))
    for key in _DS_KEYS:
        if key in pre_ft_tok or key in pre_ft_cfg:
            dataset = _DATASET_MAP[key]
            break

    # Fallback: OpenAI configs always mean WIT-400M pretraining
    if dataset is None and org == 'OpenAI':
        dataset = 'WIT-400M'

    # Fallback: SAM2-Hiera backbone pretraining is always on SA-1B
    # (the config encodes resolution 'r896' and token count '2pt1' but not the dataset)
    # We detect this by checking if the pretrain_config starts with 'fb_r'
    if dataset is None and re.match(r'fb_r\d+', cfg):
        dataset = 'SA-1B'

    # Fallback: FCMAE/MAE/MIM configs where the pretrain dataset is implicit.
    # Only applies when ft_target is an *intermediate* dataset (not the final IN-1K),
    # i.e. the pattern is "method_ft_in22k_in1k" meaning pretrain→ft-22k→ft-1k.
    # When ft_target IS IN-1K already (e.g. fcmae_ft_in1k), there's no implicit
    # pretrain dataset we can infer reliably, so we leave dataset=None.
    if dataset is None and ft_target is not None and ft_target != 'ImageNet-1K':
        dataset = ft_target
        ft_target = 'ImageNet-1K'  # the real ft target is always IN-1K in these cases

    # --- dataset size qualifier ---
    ds_size: str | None = None
    for pat, label in _DS_SIZE_PATTERNS:
        if re.search(pat, cfg):
            ds_size = label
            break

    # --- pretrain resolution  (explicit r{N} token) ---
    pretrain_resolution: int | None = None
    m = re.search(r'(?:^|_)r(\d{3,4})(?:_|$)', pretrain_config)
    if m:
        pretrain_resolution = int(m.group(1))
    else:
        # fallback: trailing resolution in the pre-ft portion only
        pre_ft_part = pretrain_config[:ft_start] if ft_start != -1 else pretrain_config
        m2 = re.search(r'_(\d{3,4})$', pre_ft_part)
        if m2:
            c = int(m2.group(1))
            if c in _VALID_PRETRAIN_RES:
                pretrain_resolution = c

    # --- fine-tune resolution ---
    pretrain_ft_resolution: int | None = None
    if ft_start != -1:
        ft_suffix = pretrain_config[ft_start:]
        m3 = re.search(r'_(\d{3,4})$', ft_suffix)
        if m3:
            c = int(m3.group(1))
            if c in _VALID_PRETRAIN_RES:
                pretrain_ft_resolution = c

    # --- training epochs (e{N} or {N}ep tokens) ---
    pretrain_epochs: int | None = None
    m = re.search(r'(?:^|_)e(\d{2,4})(?:_|$)', cfg)
    if m:
        pretrain_epochs = int(m.group(1))
    else:
        m = re.search(r'(?:^|_)(\d{3,4})ep(?:_|$)', cfg)
        if m:
            pretrain_epochs = int(m.group(1))

    # --- token-count qualifier (2pt1, 2pt5) ---
    pretrain_tokens_qual: str | None = None
    m = re.search(r'(?:^|_)(\d+pt\d+)(?:_|$)', cfg)
    if m:
        pretrain_tokens_qual = m.group(1)

    # --- augmentation recipe used during pretraining ---
    pretrain_aug: str | None = None
    for kw, label in _AUG_PATTERNS:
        if kw in tokens:
            pretrain_aug = label
            break

    # --- i18n / multilingual flag (WebLI-i18n) ---
    pretrain_i18n = 'i18n' in tokens

    return {
        'pretrain_config': pretrain_config,
        'pretrain_org': org,
        'pretrain_dataset': dataset,
        'pretrain_dataset_size': ds_size,
        'pretrain_method': method,
        'pretrain_ft': ft_target,
        'pretrain_resolution': pretrain_resolution,
        'pretrain_ft_resolution': pretrain_ft_resolution,
        'pretrain_epochs': pretrain_epochs,
        'pretrain_tokens': pretrain_tokens_qual,
        'pretrain_aug': pretrain_aug,
        'pretrain_i18n': pretrain_i18n,
    }


# ===========================================================================
# TIMM FIELDS
# ===========================================================================


def _get_timm_info(model_name: str) -> tuple[int | None, int | None]:
    """
    Instantiate the model (weights=False) via timm to get:
    - num_parameters: total trainable parameter count
    - latent_dim: dimensionality of the penultimate feature vector fed to the
      classifier head. Equivalent to model.num_features in timm convention,
      which is what you get from forward_features() before the head.

    Requires timm to be installed. Returns (None, None) if timm is not
    available or the model name is not recognised.
    """
    try:
        import timm

        model = timm.create_model(model_name, pretrained=False)
        num_params = sum(p.numel() for p in model.parameters())
        # timm standard: num_features is the pre-head feature dim.
        # Fall back through a chain of common attribute names.
        latent_dim = (
            getattr(model, 'num_features', None)
            or getattr(model, 'embed_dim', None)
            or getattr(model, 'hidden_size', None)
            or getattr(model, 'd_model', None)
        )
        return num_params, latent_dim
    except Exception as exc:
        logger.warning("timm could not instantiate '%s': %s", model_name, exc)
        return None, None


# ===========================================================================
# PUBLIC API
# ===========================================================================


def get_model_metadata(model_name: str, use_timm: bool = True) -> dict:
    """
    Given a timm model name (``{model_variant}.{pretrain_config}``),
    return a flat dict with every extractable metadata field.
    See module docstring for the complete field list.

    Parameters
    ----------
    model_name : str
        Full timm model name in ``variant.pretrain_config`` format.
    use_timm : bool, default True
        If True, instantiate the model via timm (requires timm to be installed)
        to populate ``num_parameters`` and ``latent_dim``.
        Set to False for fast name-only parsing when those fields aren't needed.
    """
    if '.' not in model_name:
        raise ValueError(
            f"model_name must follow 'variant.pretrain_config', got: {model_name!r}"
        )
    model_variant, pretrain_config = model_name.split('.', 1)
    num_parameters, latent_dim = (
        _get_timm_info(model_name) if use_timm else (None, None)
    )

    variant_fields = _parse_variant(model_variant)
    pretrain_fields = _parse_pretrain_config(pretrain_config)

    # Cross-field inference:
    # 0. Config-side distillation token — CaiT, RepViT, SwiftFormer encode
    #    distillation in the config (e.g. fb_dist_in1k, dist_450e_in1k,
    #    dist_in1k) rather than in the variant.
    if not variant_fields['is_distilled'] and re.search(
        r'(?:^|_)dist(?:_|$)', pretrain_config.lower()
    ):
        variant_fields['is_distilled'] = True

    # 1. If variant contains '_clip_' or '_clip' and method is still None,
    #    infer CLIP (the model architecture was trained with CLIP objective
    #    but the config token doesn't spell it out, e.g. openai, openai_ft_in1k).
    if pretrain_fields['pretrain_method'] is None:
        v = model_variant.lower()
        if '_clip' in v or v.endswith('clip'):
            pretrain_fields['pretrain_method'] = 'CLIP'

    # 2. AIMv2 uses a multimodal autoregressive objective, not generic "Pretraining".
    if (
        variant_fields['family'] == 'AIMv2'
        and pretrain_fields['pretrain_method'] == 'Pretraining'
    ):
        pretrain_fields['pretrain_method'] = 'Autoregressive'

    # 3. DINOv2 method from variant dinov2 token
    if pretrain_fields['pretrain_method'] is None:
        if 'dinov2' in model_variant.lower():
            pretrain_fields['pretrain_method'] = 'DINOv2'
        elif 'dinov3' in model_variant.lower():
            pretrain_fields['pretrain_method'] = 'DINOv3'
        elif 'dino' in model_variant.lower() and 'dinov' not in model_variant.lower():  # noqa: E501
            pretrain_fields['pretrain_method'] = 'DINO'

    # 4. Size inference for families that use short suffixes SIZE_RULES can't
    #    safely match without false positives on other families.
    if variant_fields['size'] is None:
        v = model_variant.lower()
        fam = variant_fields['family']
        # EfficientFormerV2: _l, _s0, _s1, _s2 suffix
        if fam == 'EfficientFormer' and variant_fields.get('model_version') == 'v2':
            if v.startswith('efficientformerv2_l'):
                variant_fields['size'] = 'L'
            elif v.startswith('efficientformerv2_s0'):
                variant_fields['size'] = 'S0'
            elif v.startswith('efficientformerv2_s1'):
                variant_fields['size'] = 'S1'
            elif v.startswith('efficientformerv2_s2'):
                variant_fields['size'] = 'S2'
        # EfficientFormerV1: _l1, _l3, _l7 — already in SIZE_RULES so usually fine,
        # but add fallback here in case prefix match didn't fire
        # MobileViT: _s / _xs / _xxs suffix (already handled by prefix rules,
        # but size_rules don't fire — derive from prefix match)
        elif fam == 'MobileViT':
            if v.startswith('mobilevit_xxs'):
                variant_fields['size'] = 'XXSmall'
            elif v.startswith('mobilevit_xs'):
                variant_fields['size'] = 'XSmall'
            elif v.startswith('mobilevit_s'):
                variant_fields['size'] = 'Small'
            elif re.search(r'(?:^|_)(050|075|100|125|150|175|200)(?:_|$)', v):
                m = re.search(r'(?:^|_)(050|075|100|125|150|175|200)(?:_|$)', v)
                variant_fields['size'] = {
                    '050': '0.50x',
                    '075': '0.75x',
                    '100': '1.00x',
                    '125': '1.25x',
                    '150': '1.50x',
                    '175': '1.75x',
                    '200': '2.00x',
                }[m.group(1)]
        # ViT-InternImage: intern300m encodes param count
        elif fam == 'ViT' and 'intern300m' in v:
            variant_fields['size'] = '300M'

    return {
        'model_name': model_name,
        **variant_fields,
        'num_parameters': num_parameters,
        'latent_dim': latent_dim,
        **pretrain_fields,
    }


# ===========================================================================
# SMOKE TEST — covers every pattern found in the real model list
# ===========================================================================

if __name__ == '__main__':
    tests = [
        # AIMv2 — param size, Apple PT, token count
        'aimv2_1b_patch14_224.apple_pt',
        'aimv2_3b_patch14_224.apple_pt',
        # ViT sizes
        'vit_tiny_patch16_224.augreg_in21k_ft_in1k',
        'vit_base_patch16_224.dino',
        'vit_large_patch14_clip_224.openai_ft_in1k',
        'vit_large_patch14_clip_quickgelu_224.openai',  # QuickGELU
        'vit_so400m_patch14_siglip_224.v2_webli',  # SO400M SigLIP WebLI-v2
        'vit_huge_patch14_clip_224.laion2b',
        # ViT with registers + RoPE + GAP
        'vit_base_patch16_rope_reg1_gap_256.sbb_in1k',
        'vit_large_patch14_reg4_dinov2.lvd142m',
        # ViT-PE scope
        'vit_pe_lang_large_patch14_448.fb',
        # ViT distilled / dwee
        'vit_dwee_patch16_reg1_gap_256.sbb_nadamuon_in1k',
        # WebLI i18n
        'vit_so400m_patch16_siglip_gap_256.webli_i18n',
        # ConvNeXt — version + size separate
        'convnext_tiny.fb_in22k_ft_in1k',
        'convnext_large.fb_in22k_ft_in1k',
        'convnextv2_large.fcmae_ft_in22k_in1k',
        'convnextv2_huge.fcmae_ft_in22k_in1k_384',  # ft@384
        # Swin — version from rule, size from scan, window size, CR+NS
        'swinv2_base_window12to16_192to256.ms_in22k_ft_in1k',
        'swin_tiny_patch4_window7_224.ms_in1k',
        'swinv2_cr_tiny_ns_224.sw_in1k',  # CR + NS flags
        # ResNet — depth code, no size
        'resnet50.a1h_in1k',
        'resnet101d.ra2_in1k',  # d suffix
        'resnetrs270.tf_in1k',  # RS + depth
        'wide_resnet50_2.tv_in1k',
        # ResNeXt — width code
        'resnext50_32x4d.fb_swsl_ig1b_ft_in1k',
        # DeiT-III — version=III, size from scan
        'deit3_base_patch16_384.fb_in22k_ft_in1k',
        'deit3_huge_patch14_224.fb_in22k_ft_in1k',
        # DeiT distilled
        'deit_tiny_distilled_patch16_224.fb_in1k',
        # EVA — version from rule, size from scan
        'eva02_large_patch14_448.mim_m38m_ft_in1k',  # M38M dataset
        'eva_giant_patch14_clip_224.laion400m',
        # EfficientNet B-series, pruned, lite
        'efficientnet_b0.ra4_e3600_r224_in1k',
        'efficientnet_b2_pruned.in1k',
        'efficientnet_lite0.ra_in1k',
        # NFNet depth code
        'dm_nfnet_f4.dm_in1k',
        # MambaOut named sizes
        'mambaout_kobe.in1k',
        'mambaout_small.in1k',
        # SAM2-Hiera, token count qualifier
        'sam2_hiera_tiny.fb_r896_2pt1',
        # Hiera abswin flag
        'hiera_small_abswin_256.sbb2_e200_in12k',
        # MaxViT rmlp + rw flags
        'maxxvitv2_rmlp_base_rw_224.sw_in12k_ft_in1k',
        # FocalNet receptive field (lrf)
        'focalnet_base_lrf.ms_in1k',
        # CaiT / XCiT stride + p-patch style
        'cait_s24_384.fb_dist_in1k',
        'xcit_small_12_p16_224.fb_dist_in1k',
        # Res2Net width+scale
        'res2net50_14w_8s.in1k',
        # FlexiViT epoch-based pretrain tag
        'flexivit_small.1200ep_in1k',
        'flexivit_base.1000ep_in21k',
        # MobileNetV4 epochs + resolution
        'mobilenetv4_conv_medium.e250_r384_in12k_ft_in1k',
        # FastViT MCI (mobile CLIP image encoder)
        'fastvit_mci0.apple_mclip2_dfndr2b',
        # Legacy prefix
        'legacy_seresnet34.in1k',
        # MetaCLIP dataset size qualifier
        'vit_large_patch14_clip_224.metaclip_400m',
        # DFN dataset size qualifier s39b
        'vit_large_patch14_clip_224.dfn2b_s39b',
        # AdvProp augmentation
        'tf_efficientnet_b0.ap_in1k',
        # ViTamin with DataComp1B
        'vitamin_large2_256.datacomp1b_clip',
        # NextViT SSLD + 6M token qualifier
        'nextvit_base.bd_ssld_6m_in1k',
        # ft resolution in trailing suffix
        'convnext_tiny.fb_in22k_ft_in1k_384',
        'rdnet_large.nv_in1k_ft_in1k_384',
        # NoisyStudent augmentation
        'tf_efficientnet_b0.ns_jft_in1k',
    ]

    for name in tests:
        print(f'\n{"=" * 72}')
        print(f'  {name}')
        print('=' * 72)
        m = get_model_metadata(name)
        # Print only non-None / non-False fields for readability
        for k, v in m.items():
            if v is not None and v is not False and k != 'model_name':
                print(f'  {k:<30} {v}')
