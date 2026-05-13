# EFM3D: Deep Research Report
## A Benchmark for Measuring Progress Towards 3D Egocentric Foundation Models

**Paper**: [arXiv:2406.10224](https://arxiv.org/abs/2406.10224)
**Authors**: Julian Straub*, Daniel DeTone**, Tianwei Shen**, Nan Yang**, Chris Sweeney, Richard Newcombe
**Affiliation**: Meta Reality Labs Research
**Year**: 2024 (arXiv preprint, June 14)
**Stored PDF**: `/research/papers/efm3d_2406.10224.pdf`

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Field Landscape & Positioning](#2-field-landscape--positioning)
3. [Paper Deep-Dive: First-Principles Analysis](#3-paper-deep-dive-first-principles-analysis)
4. [Architecture: EVL (Egocentric Voxel Lifting)](#4-architecture-evl-egocentric-voxel-lifting)
5. [Code-Grounded Architecture Walkthrough](#5-code-grounded-architecture-walkthrough)
6. [Dataflow & Pipeline Analysis](#6-dataflow--pipeline-analysis)
7. [Training Pipeline](#7-training-pipeline)
8. [Inference & Evaluation Pipeline](#8-inference--evaluation-pipeline)
9. [Dataset Ecosystem](#9-dataset-ecosystem)
10. [Experimental Results & Critical Analysis](#10-experimental-results--critical-analysis)
11. [Strengths, Weaknesses & Research Taste Assessment](#11-strengths-weaknesses--research-taste-assessment)
12. [Future Directions & Open Problems](#12-future-directions--open-problems)
13. [Quick-Start Experiment Guide](#13-quick-start-experiment-guide)
14. [Key File Reference](#14-key-file-reference)

---

## 1. Executive Summary

EFM3D introduces two things:

1. **A benchmark** for 3D egocentric perception with two tasks: **3D oriented bounding box (OBB) detection** and **3D surface regression**, evaluated on high-quality annotated data from Meta's Project Aria wearable device ecosystem.
2. **EVL (Egocentric Voxel Lifting)**, a baseline model that lifts frozen 2D foundation model features (DINOv2) into an explicit 3D gravity-aligned voxel grid, then runs 3D CNNs with task-specific heads.

**Key insight**: The paper argues that wearable AR glasses produce a fundamentally different data modality than RGB-D scanners or autonomous vehicles — egocentric data has sparse semi-dense points (not dense depth), non-linear camera models (fisheye), and egocentric motion patterns. Existing 3D methods designed for RGB-D or LiDAR struggle with these characteristics.

**Core result**: EVL, trained only on synthetic ASE data (~10k sequences), generalizes to real-world ADT and AEO datasets, outperforming existing methods (ImVoxelNet, 3DETR, Cube R-CNN, NeuralRecon, ZoeDepth, SimpleRecon, ConsistentDepth) on both tasks.

---

## 2. Field Landscape & Positioning

### 2.1 Where EFM3D Sits in the Taxonomy

```
                    3D Scene Understanding
                    /        |          \
           Autonomous     Indoor RGB-D    Egocentric
           Vehicles       Scanning        Wearable
           (BEV/Voxel)    (ScanNet)       (THIS WORK)
           |              |               |
           BEVFormer      VoteNet         EFM3D/EVL
           TPVFormer      3DETR
           SurroundOcc    ImVoxelNet
```

### 2.2 Egocentric Perception Datasets

| Dataset | Type | Modality | 3D OBB | Surface | Scale |
|---------|------|----------|--------|---------|-------|
| Ego4D (CVPR'22) | Real | Video | ✗ | ✗ | 3,670 hrs |
| Ego-Exo4D (CVPR'24) | Real | Video+Points | ✗ | ✗ | 5,625 seqs |
| EgoObjects (ECCV'22) | Real | Video | ✗ | ✗ | 9k+ videos |
| ARKitScenes (NeurIPS'21) | Real | RGB-D | ✓ (52k) | Fused depth | 5,048 seqs |
| ScanNet (CVPR'17) | Real | RGB-D | ✓ (36k) | Fused depth | 1,513 seqs |
| **ASE** (this work) | Sim | Aria sensors | ✓ (3M) | CAD mesh | 100k seqs |
| **ADT** (ICCV'23) | Real | Aria sensors | ✓ (281) | CAD mesh | 6 seqs |
| **AEO** (this work) | Real | Aria sensors | ✓ (584) | ✗ | 26 seqs |

**Key gap EFM3D fills**: No prior dataset is both (a) egocentric with Aria-specific modalities and (b) annotated with both 3D OBBs and surface meshes. ScanNet/ARKitScenes use RGB-D (dense depth), which is fundamentally different from Aria's sparse semi-dense points.

### 2.3 3D Foundation Models

| Method | Approach | 3D Representation | Backbone | Egocentric? |
|--------|----------|-------------------|----------|-------------|
| 3D-LLM (NeurIPS'23) | Language-grounded 3D | Point cloud | LLM | ✗ |
| LEO (CVPR'24) | Embodied world model | Point cloud | LLM | Partial |
| Uni3D (ICLR'24) | Unified 3D repr. learning | Point cloud | ViT | ✗ |
| OpenScene (CVPR'23) | Open-vocab 3D understanding | Point cloud | CLIP | ✗ |
| Ponder (CVPR'23) | Pre-train with NeRF rendering | Dense features | ResNet | ✗ |
| **EVL** (this work) | Lift 2D FM to 3D voxels | Voxel grid | DINOv2 | ✓ |

**EVL's differentiator**: It is the only method specifically designed for egocentric sensor data (fisheye cameras, sparse points, multi-stream video from Aria). Most 3D FMs assume point cloud input from dense RGB-D, which doesn't exist in the egocentric setting.

### 2.4 Voxel-based 2D→3D Lifting Methods

| Method | Domain | Lifting Strategy | 3D Backbone |
|--------|--------|-----------------|-------------|
| ATLAS (CVPR'20) | Indoor scenes | Unproject to 3D grid | 3D CNN |
| ImVoxelNet (WACV'22) | Indoor scenes | Project voxels into images | 3D CNN |
| NeuralRecon (CVPR'21) | Indoor scenes | Local TSDF → learned fusion | GRU |
| BEVFormer (ECCV'22) | Autonomous driving | Deformable attention | Transformer |
| TPVFormer (CVPR'23) | Autonomous driving | Tri-perspective view | Transformer |
| VoxFormer (CVPR'23) | Autonomous driving | Depth-guided voxel queries | Transformer |
| SimpleRecon (ECCV'22) | Indoor scenes | Multi-view cost volume | 2D CNN |
| **EVL** (this work) | Egocentric | Project + sample + aggregate | 3D InvResNet-FPN |

**EVL vs. ImVoxelNet** (most similar): Both project 3D voxel centers into 2D images and sample features. EVL adds: (1) multi-stream support (RGB + 2 SLAM cameras), (2) semi-dense point masks as geometric priors, (3) fisheye camera model support, (4) DINOv2 frozen backbone vs. ResNet.

### 2.5 DINOv2 as a 3D Backbone

DINOv2 (Oquab et al., 2023) has become a dominant frozen feature extractor for 3D tasks:
- **Depth Anything** (CVPR'24): Monocular depth from DINOv2 features
- **Metric3D v2** (2024): Metric depth estimation
- **EVL** (this work): Lifts DINOv2 tokens to 3D voxel grid

The paper finds DINOv2 outperforms CLIP as the 2D backbone for 3D tasks (Section 4.3), likely because DINOv2's self-supervised objective produces more geometrically aware features.

---

## 3. Paper Deep-Dive: First-Principles Analysis

### 3.1 Problem Formulation

**Input**: A "snippet" of 1-2 seconds of egocentric data from Project Aria:
- T posed RGB frames (240×240×3) at 10Hz
- T posed grayscale SLAM frames (320×240×1, left + right) at 10Hz
- Camera calibration (fisheye model) per stream
- Semi-dense 3D points from Aria's SLAM system
- 6-DoF camera poses from Aria's SLAM system

**Output** (Task 1 — 3D OBB Detection):
- Set of oriented 3D bounding boxes, each with: center (x,y,z), dimensions (h,w,d), yaw rotation, semantic class

**Output** (Task 2 — Surface Regression):
- Occupancy value per voxel ∈ [0,1], which is fused into a TSDF and extracted as a mesh via marching cubes

### 3.2 Core Design Principles

1. **Explicit 3D representation**: Uses a gravity-aligned voxel grid (4m × 4m × 4m) rather than implicit representations. This is a deliberate choice — explicit voxels enable standard 3D CNN processing and direct geometric reasoning.

2. **Frozen 2D foundation model**: DINOv2 ViT-B/14 is kept frozen. Only the 2D upsampling CNN, 3D U-Net, and task heads are trained. This inherits strong 2D priors without needing massive 3D pre-training data.

3. **Multi-modal fusion in 3D space**: Rather than fusing at the 2D level, EVL lifts each camera stream independently to 3D and aggregates in voxel space using mean + std pooling across time and cameras. This naturally handles the different camera FOVs and resolutions.

4. **Geometric priors via semi-dense points**: The semi-dense point cloud is encoded as two binary masks — a surface point mask and a freespace mask — concatenated to the feature volume. This provides strong geometric cues without a point cloud encoder.

### 3.3 The "Lifting" Operation (Mathematical Detail)

Given a voxel grid $V \in \mathbb{R}^{D \times H \times W \times 3}$ of 3D positions and a camera with projection function $\pi$:

1. For each voxel center $v_{dhw} \in \mathbb{R}^3$, transform to camera coordinates: $v_{cam} = T_{cw} \cdot v_{dhw}$
2. Project to pixel coordinates: $u = \pi(v_{cam})$, respecting the valid radius of the fisheye model
3. Sample the 2D feature map $F_{2D} \in \mathbb{R}^{C \times H_{img} \times W_{img}}$ at $u$ via bilinear interpolation
4. This yields per-camera features $f_{dhw}^{cam,t} \in \mathbb{R}^C$ for each voxel, camera, and timestep
5. Aggregate across T timesteps using mean and standard deviation: $f_{dhw} = [\mu(f^{*,*}_{dhw}), \sigma(f^{*,*}_{dhw})]$

The resulting feature volume is $\mathbb{R}^{2F \times D \times H \times W}$ (where $2F$ = mean + std channels).

### 3.4 Loss Functions

**OBB Detection Loss** (Eq. 1 in paper):
$$L_{obj} = \frac{1}{N_v} \sum_n w_c \text{FL}(v_n^c, \hat{v}_n^c) + w_{iou} \text{IoU}(v_n^{obb}, \hat{v}_n^{obb}) + w_{cls} \text{FL}(v_n^{cls}, \hat{v}_n^{cls})$$

Where:
- $w_{cent} = 100$, $w_{iou} = 10$, $w_{cls} = 1.0$ (from code, `evl_loss.py:99-103`)
- FL = Focal Loss (for class imbalance)
- IoU = 3D Rotated IoU Loss (from mmdetection3d)

**Surface Loss** (Eq. 2 in paper):
$$L_{surf} = \frac{1}{N} \sum_n \text{FL}(p_{free}^n, 0.0) + \text{FL}(p_{surf}^n, 0.5) + \text{FL}(p_{occ}^n, 1.0)$$

Plus a Total Variation (TV) regularization for smoothness: $L_{tv}$ with weight 0.01.

**Key design choice**: The surface loss uses three-class focal loss at sub-voxel resolution. For each GT depth point, three supervision points are generated: a free-space point (in front of surface, target=0), a surface point (at surface, target=0.5), and an occupied point (behind surface, target=1.0). Both $p_{free}$ and $p_{occ}$ are sampled at distance $\delta$ (= voxel size) from the surface.

---

## 4. Architecture: EVL (Egocentric Voxel Lifting)

### 4.1 Architecture Diagram

```
┌────────────────────────────────────────────────────────────────────────┐
│                        EVL Architecture                                │
│                                                                        │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │ RGB frames   │    │ SLAM-L frames│    │ SLAM-R frames│              │
│  │ T×240×240×3  │    │ T×320×240×1  │    │ T×320×240×1  │              │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘              │
│         │                   │                   │                      │
│         ▼                   ▼                   ▼                      │
│  ┌──────────────────────────────────────────────────────┐              │
│  │           DINOv2 ViT-B/14 (FROZEN)                   │              │
│  │           → patch tokens per stream                   │              │
│  └──────────────────────────┬───────────────────────────┘              │
│                             │                                          │
│                             ▼                                          │
│  ┌──────────────────────────────────────────────────────┐              │
│  │        UpsampleCNN (per stream, trainable)            │              │
│  │        14×14 tokens → 240×240 feature maps            │              │
│  │        768-dim → out_dim (e.g., 64)                   │              │
│  └──────────────────────────┬───────────────────────────┘              │
│                             │                                          │
│         ┌───────────────────┼───────────────────┐                      │
│         │                   │                   │                      │
│         ▼                   ▼                   ▼                      │
│  ┌─────────────────────────────────────────────────────┐               │
│  │              LIFTER: 2D→3D Voxel Lifting             │               │
│  │                                                      │               │
│  │  1. Create gravity-aligned 3D voxel grid             │               │
│  │     (anchored at last RGB camera pose)               │               │
│  │  2. Project voxel centers → each image               │               │
│  │  3. Sample features via bilinear interpolation       │               │
│  │  4. Aggregate: mean + std across T frames            │               │
│  │  5. Concatenate point mask + freespace mask          │               │
│  │                                                      │               │
│  │  Output: B × (2F+2) × D × H × W                    │               │
│  │          e.g., B × 66 × 96 × 96 × 96               │               │
│  └──────────────────────────┬──────────────────────────┘               │
│                             │                                          │
│                             ▼                                          │
│  ┌──────────────────────────────────────────────────────┐              │
│  │         InvResnetFpn3d (3D U-Net "Neck")              │              │
│  │                                                       │              │
│  │  Block1: 66→64  (stride 1, 2 bottles)                │              │
│  │  Block2: 64→96  (stride 2, 2 bottles, ↓2x)          │              │
│  │  Block3: 96→128 (stride 2, 2 bottles, ↓4x)          │              │
│  │  Block4: 128→160 (stride 2, 2 bottles, ↓8x)         │              │
│  │  FPN3:  160→128 (↑2x + lateral from Block3)          │              │
│  │  FPN2:  128→96  (↑2x + lateral from Block2)          │              │
│  │  FPN1:  96→64   (↑2x + lateral from Block1)          │              │
│  │                                                       │              │
│  │  Output: B × 64 × D × H × W                         │              │
│  └──────────────────┬───────────────────────────────────┘              │
│                     │                                                  │
│          ┌──────────┴──────────┐                                       │
│          │                     │                                       │
│          ▼                     ▼                                       │
│  ┌───────────────┐    ┌────────────────┐                               │
│  │  OBB Head     │    │  Occupancy Head│                               │
│  │               │    │                │                               │
│  │ Centerness(1) │    │  Conv3d → 1ch  │                               │
│  │ BBox(7)       │    │  → sigmoid     │                               │
│  │ Class(29)     │    │                │                               │
│  │ → NMS         │    │  occ ∈ [0, 1]  │                               │
│  │ → OBBs        │    │  per voxel     │                               │
│  └───────────────┘    └────────────────┘                               │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Voxel Grid Configuration

| Parameter | Server (evl_inf.yaml) | Desktop (evl_inf_desktop.yaml) |
|-----------|----------------------|-------------------------------|
| Voxel resolution | 96 × 96 × 96 | 48 × 48 × 48 |
| Spatial extent | [-2, 2] × [0, 4] × [-2, 2] m | Same |
| Voxel size | ~4.17 cm | ~8.33 cm |
| Neck dims | [128, 256, 512] | [32, 64, 128] |
| Out dim | 64 | 32 |
| GPU memory | >20 GB | ~10 GB |

The voxel grid is **gravity-aligned** and centered at the last RGB camera position, extending 2m left/right, 4m forward, and 2m up/down. This captures a typical room-scale egocentric view.

---

## 5. Code-Grounded Architecture Walkthrough

### 5.1 Entry Point: `EVL.forward()` → `efm3d/model/evl.py:172-223`

```python
def forward(self, batch, obb_only=False):
    out = {}
    # Step 1: Run frozen DINOv2 on all camera streams
    backbone2d_out_all = self.backbone2d(batch)  # VideoBackboneDinov2
    for stream in ["rgb", "slaml", "slamr"]:
        if stream in backbone2d_out_all:
            batch[f"{stream}/feat"] = backbone2d_out_all[stream]

    # Step 2: Lift 2D features to 3D voxel grid
    backbone3d_out = self.backbone3d(batch)  # Lifter
    voxel_feats = backbone3d_out["voxel/feat"]  # B × C × D × H × W

    # Step 3: Process with 3D U-Net neck
    neck_feats1 = self.neck(voxel_feats)  # InvResnetFpn3d

    # Step 4a: Occupancy head (surface reconstruction)
    if not obb_only:
        occ_logits = self.occ_head(neck_feats1)
        occ_pr = torch.sigmoid(occ_logits)
        out["occ_pr"] = occ_pr

    # Step 4b: OBB detection heads
    cent_pr = torch.sigmoid(self.cent_head(neck_feats2))     # Centerness
    bbox_pr = self.bbox_head(neck_feats2)                     # Box params (7D)
    clas_pr = torch.nn.functional.softmax(self.clas_head(neck_feats2), dim=1)  # Class

    # Step 5: Post-process (NMS, convert voxels to OBBs)
    out = self.post_process(batch, out)
    return out
```

### 5.2 2D Feature Extraction: `VideoBackboneDinov2` → `efm3d/model/video_backbone.py:129-176`

```python
class VideoBackboneDinov2(VideoBackbone):
    def __init__(self, image_tokenizer, ...):
        # image_tokenizer wraps DINOv2 ViT-B/14
        self.image_tokenizer = instantiate(self.image_tokenizer)
        self.feat_dim = self.image_tokenizer.feat_dim()  # 768

    def forward_impl(self, img, stream):
        # img: B × T × C × H × W
        img_tokens = self.image_tokenizer.forward(img)
        # Returns BxTxHxWxC → rearranged to BxTxCxHxW
        # Token grid: 240/14 ≈ 17 × 17 tokens of 768-dim
        return {stream: einops.rearrange(img_tokens, "b t h w c -> b t c h w")}
```

### 5.3 The Lifter: 2D→3D Projection → `efm3d/model/lifter.py:86-533`

This is the **most critical module** — it implements the paper's core contribution.

**Key method: `forward()`** (line 458):
```python
def forward(self, batch):
    B, T, _, H, W = batch[ARIA_IMG[0]].shape

    # 1. Run upsampling head on DINOv2 features (17×17 → 240×240)
    for stream in self.streams:
        feats2d[stream] = batch[f"{stream}/feat"]
        feats2d[stream] = self.head.forward(feats2d[stream])  # UpsampleCNN

    # 2. Compute gravity-aligned voxel grid pose
    T_wv, selectT = self.get_voxelgrid_pose(cams, T_ws, Ts_sr)
    # T_wv: B × 12 (PoseTW) — only has yaw rotation, gravity-aligned

    # 3. Generate voxel grid world positions
    vox_v = create_voxel_grid(vW, vH, vD, self.voxel_extent, device)
    vox_w = T_wv * vox_v  # Transform to world coords

    # 4. Compute geometric priors (semi-dense point masks)
    point_masks = self.get_points_counts(batch, T_wv, ...)  # Binary surface mask
    free_masks = self.get_freespace_counts(batch, T_wv, ...) # Binary freespace mask

    # 5. Lift & aggregate: project voxels → images → sample features
    vox_feats, count_feats, _ = self.lift_aggregate_centers(batch, feats2d, vox_w, ...)
    # This projects each voxel center into each camera, samples features,
    # then aggregates across time with mean pooling

    # 6. Concatenate features + geometric masks
    vox_feats = torch.concatenate([vox_feats, point_masks, free_masks], dim=1)
    # Final shape: B × (2F + 2) × D × H × W
```

**Key method: `lift()`** (line 360):
```python
def lift(self, feats2d, vox_w, cam, Ts_wr, vD, vH, vW):
    # Transform voxel world positions to camera coordinates
    vox_cam = Ts_wc.inverse() * vox_w

    # Sample 2D features at projected voxel positions
    # Respects fisheye camera valid radius
    vox_feats, vox_valid = sample_images(
        feats2d, vox_cam, cam, n_by_c=False, warn=False, single_channel_mask=True
    )
    # vox_feats: B × T × F × D × H × W
    # vox_valid: B × T × 1 × D × H × W (binary: is voxel visible in this view?)
    return vox_feats, vox_valid
```

**Key method: `aggregate()`** (line 377):
```python
def aggregate(self, vox_feats, vox_valid):
    # Mean pooling across time dimension, respecting validity masks
    vox_feats, count = basic_mean(vox_feats, dim=1, valid=vox_valid)
    return vox_feats, count
```

> **Note on std aggregation**: The paper mentions using both mean and standard deviation for aggregation. In the code, the `lift_aggregate_centers` method handles this — the feature volume dimension is `2F` because mean and std are concatenated. This is controlled by the `aggregate` method and the stream configuration.

### 5.4 3D CNN Neck: `InvResnetFpn3d` → `efm3d/model/cnn.py:266-309`

```python
class InvResnetFpn3d(nn.Module):
    # Encoder-decoder with skip connections (FPN-style)
    def forward(self, x):
        x1 = self.block1(x)   # 66→64,  stride 1 (full res)
        x2 = self.block2(x1)  # 64→96,  stride 2 (½ res)
        x3 = self.block3(x2)  # 96→128, stride 2 (¼ res)
        x = self.block4(x3)   # 128→160, stride 2 (⅛ res)

        # Feature Pyramid Network (FPN) upsampling path
        x = self.fpn3(x, x3)  # 160→128, ↑2x + lateral from x3
        x = self.fpn2(x, x2)  # 128→96,  ↑2x + lateral from x2
        x = self.fpn1(x, x1)  # 96→64,   ↑2x + lateral from x1
        return x  # B × 64 × D × H × W
```

Each block uses **Inverted Bottleneck** residual blocks (`InvBottleNeck3d`), inspired by MobileNet. The expansion ratio is 2.0, meaning hidden dim = 2× input dim inside each bottle.

### 5.5 Task Heads: `VolumeCNNHead` → `efm3d/model/cnn.py:362-431`

Each head is a simple 2-layer 3D CNN:
```python
class VolumeCNNHead(nn.Module):
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))  # 64 → hidden_dim
        x = self.conv2(x)                        # hidden_dim → final_dim
        return x
```

**Head outputs**:
- `occ_head`: 1 channel → sigmoid → occupancy probability
- `cent_head`: 1 channel → sigmoid → centerness probability (bias=-5 initialization for sparse predictions)
- `bbox_head`: 7 channels → [3 dims (sigmoid, clamped to [0.1, 6.0]m), 3 offsets (tanh, clamped to ±2×voxel_size), 1 yaw (tanh, clamped to ±1.6 rad)]
- `clas_head`: 29 channels → softmax → class probabilities

### 5.6 OBB Post-Processing → `efm3d/model/evl.py:137-170`

```python
def post_process(self, batch, out):
    # 1. 3D NMS on centerness heatmap
    cent_pr_nms = simple_nms3d(cent_pr, nms_radius=self.splat_sigma + 1)

    # 2. Convert dense voxel predictions to sparse OBB objects
    obbs_pr_nms, _, clas_prob_nms = voxel2obb(
        cent_pr_nms, bbox_pr, clas_pr, self.ve,
        top_k=128, thresh=self.det_thresh,  # det_thresh=0.2
        return_full_prob=True,
    )

    # 3. Transform from voxel coords to snippet coords
    T_sv = T_ws.inverse() @ T_wv
    obbs_pr_nms_s = obbs_pr_nms.transform(T_sv)
```

---

## 6. Dataflow & Pipeline Analysis

### 6.1 Complete Data Flow

```
Raw Aria Data (.vrs or ATEK .tar)
        │
        ▼
┌───────────────────────────────────────────────────┐
│  VrsSequenceDataset / AtekWdsStreamDataset         │
│  - Extract snippets (1s @ 10Hz = 10 frames)        │
│  - Load RGB, SLAM-L, SLAM-R images                 │
│  - Load camera calibrations (fisheye params)        │
│  - Load 6-DoF poses (T_world_rig per frame)         │
│  - Load semi-dense 3D points                         │
│  - Load GT OBBs (if training/evaluation)             │
└───────────────────────┬───────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────┐
│  EfmModelAdaptor (for ATEK format)                  │
│  - Align gravity to [0, 0, -9.81]                   │
│  - Split T_world_rig → T_world_snippet + T_snip_rig│
│  - Pad semi-dense points to 50k per frame           │
│  - Map ATEK taxonomy → EFM taxonomy (29 classes)    │
│  - Pad OBBs to 128 max per snippet                  │
│  - Normalize images to [0, 1]                        │
└───────────────────────┬───────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────┐
│  preprocess() — Training augmentations               │
│  - ColorJitter (brightness, contrast, sat, hue)     │
│  - PointDropSimple (max 80% dropout)                 │
│  - PointJitter (depth noise injection)               │
│  - Move to GPU                                        │
└───────────────────────┬───────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────┐
│  EVL Model Forward Pass                              │
│  [See Section 5 for detailed code walkthrough]       │
└───────────────────────┬───────────────────────────┘
                        │
           ┌────────────┴────────────┐
           ▼                         ▼
    OBB Predictions           Occupancy Volume
    (per snippet)             (per snippet)
           │                         │
           ▼                         ▼
    ┌─────────────┐          ┌──────────────┐
    │ OBB Tracker │          │ TSDF Fusion  │
    │ (IoU match) │          │ (running avg)│
    └──────┬──────┘          └──────┬───────┘
           │                        │
           ▼                        ▼
    Scene-level OBBs         Global Mesh
    (tracked_scene_obbs.csv) (fused_mesh.ply)
           │                        │
           ▼                        ▼
    mAP Evaluation           Mesh Accuracy/
    (IoU thresholds           Completeness
     0.0 to 0.5)             (5cm threshold)
```

### 6.2 Key Data Structures

**Batch dictionary keys** (from `aria_constants.py`):
```
batch["rgb/img"]                    # B × T × 3 × 240 × 240  (RGB frames)
batch["slaml/img"]                  # B × T × 1 × 320 × 240  (SLAM left)
batch["slamr/img"]                  # B × T × 1 × 320 × 240  (SLAM right)
batch["rgb/calib"]                  # CameraTW: projection params, valid_radius, T_camera_rig
batch["snippet/t_world_snippet"]    # PoseTW: B × 1 × 12
batch["rgb/t_snippet_rig"]          # PoseTW: B × T × 12
batch["points/p3s_world"]           # B × T × 50000 × 3  (semi-dense points)
batch["obbs/padded_snippet"]        # ObbTW: B × T × 128 × 34  (GT OBBs, padded)
```

**Custom Tensor Wrappers** (`efm3d/aria/`):
- `PoseTW`: 12-dim representation (3×3 rotation matrix flattened + 3D translation)
- `CameraTW`: Camera calibration with fisheye projection support
- `ObbTW`: 34-dim oriented bounding box (bb3_object[6], T_world_object[12], bb2_rgb[4], bb2_slaml[4], bb2_slamr[4], sem_id[1], inst_id[1], score[1], pad_mask[1])

---

## 7. Training Pipeline

### 7.1 Training Script: `train.py`

**Configuration**:
```python
DATA_PATH = "./data/ase_train"        # ASE synthetic dataset (~7TB)
MAX_LR = 2e-4                          # Peak learning rate
MIN_LR = 2e-5                          # Minimum learning rate
BATCH_SIZE = 2                          # Per-GPU batch size
MAX_EPOCHS = 40                         # Total epochs
MAX_SAMPLES_PER_EPOCH = 100,000        # Cap per epoch
SAVE_EVERY_EPOCHS = 5                   # Checkpoint frequency
```

**Training loop** (simplified):
```python
for epoch in range(MAX_EPOCHS):
    for batch in train_dataloader:
        batch = preprocess(batch, device, aug_funcs=augmentations)  # Augment + to GPU
        output = model(batch)                                        # Forward pass
        losses, total_loss = raw_model.compute_losses(output, batch) # OBB + Occ losses
        total_loss.backward()                                        # Backward
        lr = get_lr(epoch, warmup=5, ...)                            # Cosine LR
        clip_grad_norm_(model.parameters(), max_norm=1.0)            # Gradient clipping
        optimizer.step()
```

**Data loading**: Uses WebDataset for streaming tar archives (scalable to 10k+ sequences). Supports distributed training via torchrun/SLURM.

### 7.2 Loss Computation: `efm3d/utils/evl_loss.py`

```python
class EVLTrain(EVL):
    def compute_losses(self, outputs, batch):
        # Occupancy losses
        occ_loss = compute_occupancy_loss_subvoxel(...)  # weight=10.0
        tv_loss = compute_tv_loss(occ)                    # weight=0.01

        # OBB losses
        cent_loss = focal_loss(cent_pr, cent_gt)          # weight=10.0
        clas_loss = focal_loss(clas_pr, clas_gt)           # weight=0.1
        iou_loss = RotatedIoU3DLoss(pred, gt)              # weight=0.5
```

The **centerness GT** is computed by "splatting" a Gaussian around each OBB center in voxel space (sigma = `splat_sigma` = max(1, int(0.12/voxel_meters)) ≈ 3 voxels).

### 7.3 Multi-GPU / Multi-Node Training

```bash
# Single GPU
python train.py

# 8 GPUs
torchrun --standalone --nproc_per_node=8 train.py

# Multi-node (SLURM) — see sbatch_run.sh
# 2 nodes × 8 GPUs = 16 GPUs (H100)
sbatch sbatch_run.sh
```

---

## 8. Inference & Evaluation Pipeline

### 8.1 Inference: `infer.py` → `efm3d/inference/pipeline.py:run_one()`

```python
def run_one(data_path, model_ckpt, ...):
    # 1. Load model from checkpoint
    model = hydra.utils.instantiate(model_config)
    model.load_state_dict(checkpoint["state_dict"])

    # 2. Create data streamer (VRS or ATEK format)
    streamer = create_streamer(data_path, snippet_length_s=1.0, stride=0.1)

    # 3. Per-snippet inference
    efm_inf = EfmInference(streamer, model, output_dir)
    efm_inf.run()  # Iterates over snippets, saves per-snippet OBBs + occupancy

    # 4. Track OBBs across snippets (IoU-based running average)
    track_obbs(output_dir)

    # 5. Fuse occupancy volumes into global mesh
    vol_fusion = VolumetricFusion(output_dir, voxel_res=0.04)
    vol_fusion.run()
    fused_mesh = vol_fusion.get_trimesh()  # Marching cubes extraction

    # 6. Evaluate against GT
    obb_metrics = evaluate_obb_csv(pred_csv, gt_csv, iou=0.2)
    mesh_metrics = eval_mesh_to_mesh(pred=fused_mesh, gt=gt_mesh)
```

### 8.2 Evaluation Metrics

**3D Object Detection**:
- **mAP** (mean Average Precision): averaged over IoU thresholds [0.0, 0.05, ..., 0.5]
- Evaluated at snippet-level and sequence-level (after tracking)
- Computed jointly across all sequences for final leaderboard metric

**Surface Reconstruction** (mesh-to-mesh comparison):
- **Accuracy** (Acc ↓): Mean distance from predicted mesh points to GT mesh
- **Completeness** (Comp ↓): Mean distance from GT mesh points to predicted mesh
- **Precision** (Prec ↑): Fraction of predicted points within 5cm of GT
- **Recall** (Rec ↑): Fraction of GT points within 5cm of predicted mesh

### 8.3 Full Evaluation Commands

```bash
# ASE evaluation (100 synthetic sequences)
python eval.py --ase --model_ckpt ./ckpt/model_release.pth

# ADT evaluation (6 real-world sequences, surface only)
python eval.py --adt --model_ckpt ./ckpt/model_release.pth

# AEO evaluation (25 real-world sequences, OBB only)
python eval.py --aeo --model_ckpt ./ckpt/model_release.pth

# Single sequence inference
python infer.py --input ./data/ase_eval/seq136 --output_dir ./output
```

---

## 9. Dataset Ecosystem

### 9.1 Aria Synthetic Environments (ASE)

- **Purpose**: Primary training dataset + in-domain evaluation
- **Scale**: ~10,000 training sequences, 100 eval sequences
- **Size**: ~7TB for training data
- **Content**: Procedurally generated indoor scenes with simulated Aria sensors
- **Annotations**: ~3M OBBs across 43 classes, surface mesh GT
- **Format**: ATEK WebDataset tar archives

### 9.2 Aria Digital Twin (ADT)

- **Purpose**: Real-world surface reconstruction evaluation
- **Scale**: 6 sequences from 1 high-quality scanned scene
- **Content**: Real Project Aria recordings in precisely reconstructed environments
- **Annotations**: CAD mesh ground truth
- **Format**: VRS video files

### 9.3 Aria Everyday Objects (AEO)

- **Purpose**: Real-world 3D object detection evaluation
- **Scale**: 26 diverse scenes, 25 evaluation sequences
- **Content**: Real egocentric recordings by non-expert data collectors
- **Annotations**: 584 OBBs across 9 classes, annotated by humans using multi-camera rig
- **Classes**: Chair, Table, Sofa, Bed, WallArt, Lamp, Plant, Window, Mirror
- **Key property**: Natural egocentric motion (no scanning guidance)

### 9.4 Semantic Taxonomy (29 classes from `ase_sem_name_to_id.csv`)

```
table(0), sofa(1), shelf(2), chair(3), bed(4), floor_mat(5),
exercise_weight(6), cutlery(7), container(8), clock(9), cart(10),
vase(11), tent(12), flower_pot(13), pillow(14), mount(15), lamp(16),
ladder(17), fan(18), cabinet(19), jar(20), picture_frame(21),
mirror(22), electronic_device(23), dresser(24), clothes_rack(25),
battery_charger(26), air_conditioner(27), window(28)
```

---

## 10. Experimental Results & Critical Analysis

### 10.1 3D Object Detection Results (Table 2)

| Method | Train | Modality | Decoder | ASE mAP (Snip) | ASE mAP (Seq) | AEO mAP (Seq) |
|--------|-------|----------|---------|-----------------|----------------|----------------|
| Cube R-CNN OTS | OTS | frame | 2D CNN | 0.01 | 0.02 | 0.05 |
| Cube R-CNN ASE | ASE | frame | 2D CNN | 0.21 | 0.36 | 0.08 |
| ImVoxelNet | ASE | snippet | 3D CNN | 0.30 | 0.64 | 0.15 |
| 3DETR | ASE | pts | Transformer | 0.24 | 0.33 | 0.16 |
| **EVL (ours)** | **ASE** | **snip+pts** | **3D CNN** | **0.40** | **0.75** | **0.22** |

**Key observations**:
1. EVL significantly outperforms all baselines on both synthetic (ASE) and real (AEO) data
2. The sim-to-real gap is substantial: -32 to -49 mAP drop for image-based methods, but only -17 for point-based 3DETR
3. OBB tracking improves snippet-level mAP by ~2× on ASE (0.40 → 0.75)
4. Point-cloud methods (3DETR) are more robust to sim-to-real transfer, but less accurate overall

### 10.2 Surface Reconstruction Results (Table 4)

| Method | Modality | ASE Acc↓ | ASE Comp↓ | ADT Acc↓ | ADT Comp↓ | ADT Prec↑ | ADT Rec↑ |
|--------|----------|----------|-----------|----------|-----------|-----------|----------|
| ZoeDepth OTS | frame | 0.368 | 1.225 | 0.130 | 0.417 | 0.200 | 0.076 |
| ConsistentDepth OTS | frame | 0.349 | 1.304 | 0.125 | 0.603 | 0.145 | 0.045 |
| SimpleRecon OTS | snippet | 0.539 | 3.064 | 0.064 | 0.326 | 0.257 | 0.064 |
| NeuralRecon OTS | snippet | 0.110 | 1.952 | 0.160 | 0.183 | 0.371 | 0.043 |
| NeuralRecon ASE | snippet | 0.212 | 1.103 | 0.241 | 0.307 | 0.474 | 0.061 |
| **EVL (ours)** | **snip+pts** | **0.057** | **0.877** | **0.182** | **0.405** | **0.594** | **0.106** |

**Key observations**:
1. EVL achieves best ASE accuracy (0.057m) and best ADT precision/recall
2. Depth-based methods (ZoeDepth) have better ADT completeness because they're not limited by a bounding volume
3. EVL's advantage comes from (a) semi-dense points as geometric prior and (b) DINOv2's strong features
4. The 4m³ bounding volume is a real limitation for completeness

### 10.3 Ablation Study (Table 3)

| Component | ASE mAP (Snip) | ASE mAP (Seq) |
|-----------|-----------------|----------------|
| No augmentation | 0.26 | 0.52 |
| + Photometric aug only | 0.38 | 0.67 |
| + Geometric aug only | 0.38 | 0.68 |
| + Both | **0.39** | **0.71** |
| Mean only (no std) | 0.26 | 0.52 |
| Std only (no mean) | 0.37 | 0.66 |
| Mean + Std | **0.39** | **0.71** |
| No point mask, no free mask | 0.38 | 0.72 |
| + Point mask only | 0.36 | 0.69 |
| + Free mask only | 0.38 | 0.69 |
| + Both | **0.39** | **0.71** |

**Critical findings**:
1. **Standard deviation aggregation is more important than mean** — std captures multi-view consistency, which is a strong cue for 3D structure
2. Geometric augmentation matters more than photometric for sim-to-real transfer
3. Point and freespace masks help but are not essential — the lifted image features already carry most of the information

---

## 11. Strengths, Weaknesses & Research Taste Assessment

### 11.1 Strengths

1. **Well-identified gap**: The paper correctly identifies that no existing benchmark evaluates 3D egocentric perception with Aria-specific modalities. This is a genuine gap.

2. **Elegant simplicity**: EVL is remarkably simple — frozen DINOv2 + project-and-sample lifting + 3D CNN. No attention mechanisms, no deformable queries, no learned positional encodings. Yet it beats more complex architectures. This is a sign of good research taste.

3. **Sim-to-real generalization**: Training on synthetic ASE data and generalizing to real ADT/AEO is the right approach for data-scarce settings. The paper demonstrates this transfer compellingly.

4. **Comprehensive evaluation**: Three datasets, two tasks, extensive ablations, comparison with diverse baselines (2D CNN, 3D CNN, Transformer, point-based). The experimental methodology is thorough.

5. **Good engineering**: The code is well-organized with clean abstractions (TensorWrapper hierarchy, modular dataset loading, configurable architecture). The WebDataset-based training pipeline is production-grade.

### 11.2 Weaknesses

1. **Limited novelty in the method**: The lifting mechanism is essentially ImVoxelNet with DINOv2 features and semi-dense point masks. The paper's novelty is primarily in the benchmark, not the model. The paper somewhat obscures this by presenting EVL as a "contribution" rather than a "baseline."

2. **Single-room scale**: The 4m³ voxel grid limits the approach to room-scale scenes. For hallways, outdoor areas, or multi-room scenarios, this is a significant constraint. The paper acknowledges this but doesn't address it.

3. **Static scene assumption**: EVL assumes a mostly static world. Dynamic objects (people, pets, moving furniture) are not modeled. For a benchmark targeting AR glasses, this is a notable gap.

4. **Benchmark bias**: ASE, ADT, and AEO are all Meta's own datasets. The benchmark evaluation is entirely within Meta's data ecosystem. External validation on non-Aria data would strengthen the claims.

5. **No temporal modeling**: The T frames in a snippet are aggregated via simple mean+std, with no attention to temporal ordering. A temporal model (e.g., GRU as in NeuralRecon) could better handle view-dependent effects and occlusion reasoning.

6. **Missing modalities**: Project Aria also provides IMU, audio, eye tracking, and WiFi/BT signals. The paper only uses RGB, SLAM cameras, and semi-dense points. A true "egocentric foundation model" should leverage all available modalities.

### 11.3 Research Taste Assessment

**Score: 7/10**

This is a solid systems paper that fills a genuine gap in the benchmark landscape. The research taste shows in:
- **What they didn't do**: They resisted the temptation to make EVL overly complex. The simple baseline is exactly what a benchmark paper needs.
- **Data quality**: The AEO dataset was collected by non-experts with natural egocentric motion, which is the right choice for evaluating real-world applicability.
- **Honest limitations**: The paper clearly states EVL's limitations (static assumption, bounded volume).

What could be improved:
- The paper would be stronger if it more clearly separated "benchmark contribution" from "method contribution"
- The claim of "3D Egocentric Foundation Model" is aspirational — EVL is not a foundation model in the modern sense (no pre-training, no generalization to new tasks, no language grounding)
- The paper doesn't explore the most interesting research question: what makes egocentric data fundamentally different from third-person data for 3D understanding?

---

## 12. Future Directions & Open Problems

### 12.1 For Researchers Working on EFM3D

1. **Temporal attention**: Replace mean+std aggregation with a learned temporal attention module. Key file to modify: `lifter.py:377-394`

2. **Larger spatial coverage**: Hierarchical or sliding-window voxel grids to cover larger scenes without memory explosion

3. **Dynamic objects**: Instance-level motion estimation, potentially by predicting per-voxel flow fields

4. **Language grounding**: Connect OBB detections with language models for open-vocabulary 3D understanding

5. **Multi-modal fusion**: Incorporate IMU data (gravity, motion priors), eye tracking (attention priors), and audio (spatial cues)

6. **Better 3D backbone**: Replace InvResnetFpn3d with a 3D vision transformer or sparse convolution network

### 12.2 For the Broader Field

1. **Bridging egocentric and allocentric**: Can a single model handle both first-person and third-person views?
2. **Real-time inference**: EVL is not real-time — making it deployable on AR glasses requires significant optimization
3. **Privacy-preserving 3D FM**: Foundation models for AR that don't store raw images

---

## 13. Quick-Start Experiment Guide

### 13.1 Environment Setup

```bash
cd /research/repos/efm3d

# Option A: Conda (recommended)
conda env create -f environment.yml
conda activate efm3d

# Option B: Pip
pip install -r requirements.txt
pip install -r requirements-extra.txt  # For training/evaluation

# Compile CUDA kernels (needed for training with IoU loss)
cd efm3d/thirdparty/mmdetection3d/cuda && pip install -e . && cd -
```

### 13.2 Download Pretrained Models

```bash
# Download from Project Aria (requires registration)
# Place files in ./ckpt/
# - model_release.pth (>20GB GPU)
# - model_lite.pth (~10GB GPU)
# - seq136_sample.zip (test sequence)

# Run setup script
bash prepare_inference.sh
```

### 13.3 Quick Inference Demo

```bash
# Run inference on sample sequence
python infer.py \
    --input ./data/ase_eval/seq136 \
    --model_ckpt ./ckpt/model_release.pth \
    --model_cfg ./efm3d/config/evl_inf.yaml \
    --output_dir ./output

# For desktop GPU (~10GB)
python infer.py \
    --input ./data/ase_eval/seq136 \
    --model_ckpt ./ckpt/model_lite.pth \
    --model_cfg ./efm3d/config/evl_inf_desktop.yaml \
    --output_dir ./output

# OBB-only mode (faster, skip surface reconstruction)
python infer.py \
    --input ./data/ase_eval/seq136 \
    --model_ckpt ./ckpt/model_release.pth \
    --obb_only \
    --output_dir ./output
```

### 13.4 Evaluation on Benchmarks

```bash
# Download evaluation data first (see data/README.md)

# ASE evaluation (3D OBB + surface)
python eval.py --ase

# ADT evaluation (surface only)
python eval.py --adt

# AEO evaluation (3D OBB only)
python eval.py --aeo
```

### 13.5 Training from Scratch

```bash
# Download ASE training data (~7TB) — see data/README.md

# Single GPU
python train.py

# Multi-GPU
torchrun --standalone --nproc_per_node=8 train.py

# Monitor with TensorBoard
tensorboard --logdir tb_logs/
```

---

## 14. Key File Reference

### Core Model Architecture
| File | Description | Lines |
|------|-------------|-------|
| `efm3d/model/evl.py` | **EVL inference model** — forward pass, OBB post-processing | 224 |
| `efm3d/model/evl_train.py` | **EVL training model** — extends EVL with loss computation + visualization | 743 |
| `efm3d/model/lifter.py` | **Core: 2D→3D Voxel Lifting** — the main architectural contribution | 534 |
| `efm3d/model/video_backbone.py` | **2D Backbone** — DINOv2 wrapper for feature extraction | 177 |
| `efm3d/model/cnn.py` | **3D CNN building blocks** — InvResnetFpn3d, VolumeCNNHead, UpsampleCNN | 521 |
| `efm3d/model/image_tokenizer.py` | **DINOv2 tokenizer** — loads and runs DINOv2 ViT-B/14 | — |
| `efm3d/model/dpt.py` | **DPT head** — alternative to UpsampleCNN for feature upsampling | — |

### Loss Functions & Utilities
| File | Description |
|------|-------------|
| `efm3d/utils/evl_loss.py` | **Loss computation** — OBB losses (centerness, IoU, class) + occupancy loss |
| `efm3d/utils/detection_utils.py` | **Detection utils** — NMS, voxel↔OBB conversion, focal loss |
| `efm3d/utils/reconstruction.py` | **Surface loss** — sub-voxel occupancy loss, TV regularization |
| `efm3d/utils/obb_metrics.py` | **OBB evaluation** — precision, recall, mAP computation |
| `efm3d/utils/mesh_utils.py` | **Mesh evaluation** — accuracy, completeness, precision, recall |

### Data Pipeline
| File | Description |
|------|-------------|
| `efm3d/dataset/efm_model_adaptor.py` | **ATEK→EFM conversion** — taxonomy mapping, coordinate transforms |
| `efm3d/dataset/vrs_dataset.py` | **VRS data loading** — native Aria video format support |
| `efm3d/dataset/atek_wds_dataset.py` | **WebDataset streaming** — for ATEK tar archives |
| `efm3d/dataset/augmentation.py` | **Training augmentations** — ColorJitter, PointDrop, PointJitter |

### Inference & Evaluation
| File | Description |
|------|-------------|
| `efm3d/inference/pipeline.py` | **Full inference pipeline** — model loading, streaming, fusion, evaluation |
| `efm3d/inference/model.py` | **Per-snippet inference** — batch processing, output saving |
| `efm3d/inference/fuse.py` | **Volumetric fusion** — TSDF fusion from per-snippet occupancy |
| `efm3d/inference/track.py` | **OBB tracking** — IoU-based temporal association |
| `efm3d/inference/eval.py` | **Evaluation** — OBB mAP, mesh metrics aggregation |

### Scripts
| File | Description |
|------|-------------|
| `train.py` | Training entry point (single/multi-GPU) |
| `eval.py` | Benchmark evaluation entry point |
| `infer.py` | Single-sequence inference entry point |
| `prepare_inference.sh` | Downloads DINOv2 weights + unpacks checkpoints |
| `sbatch_run.sh` | SLURM multi-node training script |

### Configuration
| File | Description |
|------|-------------|
| `efm3d/config/evl_train.yaml` | Training model config |
| `efm3d/config/evl_inf.yaml` | Server GPU inference config |
| `efm3d/config/evl_inf_desktop.yaml` | Desktop GPU inference config |
| `efm3d/config/taxonomy/ase_sem_name_to_id.csv` | 29-class semantic taxonomy |
| `efm3d/config/taxonomy/atek_to_efm.csv` | ATEK→EFM class mapping |

---

*Report generated on 2026-04-01. Paper PDF stored at `/research/papers/efm3d_2406.10224.pdf`.*
