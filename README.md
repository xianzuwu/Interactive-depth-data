# Interactive-depth-data Engineering Pipeline for Wrist-Mounted camera

## System Overview

* **Camera**: RealSense D435i (RGB-D + IMU)
* **Installation**: Mounted on robotic arm wrist (end-effector)
* **Robot Arm**: Any brand with high-precision TCP pose output
* **Constraints**: No external LiDAR or global shutter 3D sensors

---

## Overall Pipeline Structure

```
[ Robot Control + D435i Capture ]
          |
[ Calibration + Time Synchronization ]
          |
[ Data Preprocessing (Denoising, Filtering, Alignment) ]
          |
[ Industrial Training Dataset ]
          |
[ Neural Network Training Loader ]
```

---

## 1. Data Acquisition

### Synchronized Recording

* RGB frames (RealSense aligned frames)
* Depth frames
* IMU data (accelerometer + gyroscope)
* TCP pose (robot controller output)
* Timestamps for all data streams

### Raw Data Folder Structure

```
dataset_raw/
  scene_01/
    rgb/ (png)
    depth/ (png, aligned to RGB)
    imu/ (csv)
    pose/ (txt, TCP pose)
    timestamps.csv
```

---

## 2. Calibration

### 2.1 Intrinsic Calibration

* Tools: RealSense SDK + OpenCV Charuco
* Board: A3 size 7x7 Charuco board (20mm squares)
* Output: `intrinsics.json`

### 2.2 Hand-Eye Calibration

* Tools: Kalibr / OpenCV `calibrateHandEye()`
* Output: `extrinsic_handeye.json`

### 2.3 Time Synchronization Calibration

* Prefer hardware triggering (best)
* Otherwise: software timestamp interpolation
* Output: Unified aligned timestamps per frame

---

## 3. Data Preprocessing (Industrial Version)

| Step                     | Method                                 | Purpose                      |
| ------------------------ | -------------------------------------- | ---------------------------- |
| Depth Denoising          | RealSense Spatial + Temporal Filtering | Stable clean depth           |
| Pose Alignment           | Timestamp interpolation                | Synchronize pose with vision |
| IMU Smoothing (optional) | Kalman Filter                          | Smooth IMU noise             |
| Frame Outlier Removal    | Depth mask + motion consistency        | Clean invalid frames         |

### Cleaned Dataset Structure

```
dataset/
  scene_01/
    rgb/
    depth/
    imu/
    pose/  # Pose transformed to camera frame
  calib/
    intrinsics.json
    extrinsic_handeye.json
  
./data/dataset/{object_category}/{instance}/images/frameXXXXXX.jpg
./data/dataset/{object_category}/{instance}/images/frameXXXXXX.npz
```

frameXXXXXX.jpg：Real RGB Image.
frameXXXXXX.npz：camera intrinsics（pose and intrinsics）：

```Python
input_metadata = np.load(impath.replace('jpg', 'npz'))
camera_pose = input_metadata['camera_pose'].astype(np.float32)
intrinsics = input_metadata['camera_intrinsics'].

```
---

## 4. Training Data Loader Design

### Enhanced Data Loader (PyTorch Example)

```python

class Co3d(BaseStereoViewDataset):
    def __init__(self, mask_bg=False, *args, ROOT=DATA_ROOT, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        assert mask_bg in (True, False, 'rand')
        self.mask_bg = mask_bg

        # load all scenes
        with open(osp.join(self.ROOT, f'selected_seqs_{self.split}.json'), 'r') as f:
            self.scenes = json.load(f)
            self.scenes = {k: v for k, v in self.scenes.items() if len(v) > 0}
            self.scenes = {(k, k2): v2 for k, v in self.scenes.items()
                           for k2, v2 in v.items()}
        self.scene_list = list(self.scenes.keys())

        if self.split =="train":
            # for each scene, we have 100 images ==> 360 degrees (so 25 frames ~= 90 degrees)
            # we prepare all combinations such that i-j = +/- [5, 10, .., 90] degrees
            self.combinations = [(i, j)
                                for i, j in itertools.combinations(range(100), 2)
                                if 0 < abs(i-j) <= 30 and abs(i-j) % 5 == 0]
        elif self.split =="test":
            # we random sample 10 images and prepare all combinations for 45 pairs
            sampled_idx = random.sample(range(1, 201), 10)
            self.combinations = list(itertools.combinations(sampled_idx, 2))[:45]
        else:
            print("invalid split, exit!")
            exit()
      
        self.invalidate = {scene: {} for scene in self.scene_list}

    def __len__(self):
        return len(self.scene_list) * len(self.combinations)
    
    def _get_maskpath(self, obj, instance, view_idx):
        return osp.join(self.ROOT, obj, instance, 'masks', f'frame{view_idx:06n}.png')
    
    def _read_depthmap(self, depthpath, input_metadata):
        depthmap = imread_cv2(depthpath, cv2.IMREAD_UNCHANGED)
        depthmap = (depthmap.astype(np.float32) / 65535) * np.nan_to_num(input_metadata['maximum_depth'])
        return depthmap

    def _get_views(self, idx, resolution, rng):
        # choose a scene
        obj, instance = self.scene_list[idx // len(self.combinations)]
        image_pool = self.scenes[obj, instance]
        im1_idx, im2_idx = self.combinations[idx % len(self.combinations)]

        # add a bit of randomness
        last = len(image_pool)-1

        if resolution not in self.invalidate[obj, instance]:  # flag invalid images
            self.invalidate[obj, instance][resolution] = [False for _ in range(len(image_pool))]

        # decide now if we mask the bg
        mask_bg = (self.mask_bg == True) or (self.mask_bg == 'rand' and rng.choice(2))

        views = []
        imgs_idxs = [max(0, min(im_idx + rng.integers(-4, 5), last)) for im_idx in [im2_idx, im1_idx]]
        imgs_idxs = deque(imgs_idxs)
        while len(imgs_idxs) > 0:  # some images (few) have zero depth
            im_idx = imgs_idxs.pop()

            if self.invalidate[obj, instance][resolution][im_idx]:
                # search for a valid image
                random_direction = 2 * rng.choice(2) - 1
                for offset in range(1, len(image_pool)):
                    tentative_im_idx = (im_idx + (random_direction * offset)) % len(image_pool)
                    if not self.invalidate[obj, instance][resolution][tentative_im_idx]:
                        im_idx = tentative_im_idx
                        break

            view_idx = image_pool[im_idx]

            impath = osp.join(self.ROOT, obj, instance, 'images', f'frame{view_idx:06n}.jpg')

            # load camera params
            input_metadata = np.load(impath.replace('jpg', 'npz'))
            camera_pose = input_metadata['camera_pose'].astype(np.float32)
            intrinsics = input_metadata['camera_intrinsics'].astype(np.float32)

            # load image and depthmap
            rgb_image = imread_cv2(impath)
            depthmap = self._read_depthmap(depthpath, input_metadata)

            if mask_bg:
                # load object mask
                maskpath = self._get_maskpath(obj, instance, view_idx)
                maskmap = imread_cv2(maskpath, cv2.IMREAD_UNCHANGED).astype(np.float32)
                maskmap = (maskmap / 255.0) > 0.1

                # update the depthmap with mask
                depthmap *= maskmap
            H, W = rgb_image.shape[:2]

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=impath)
            
            num_valid = (depthmap > 0.0).sum()
            if num_valid == 0:
                # problem, invalidate image and retry
                self.invalidate[obj, instance][resolution][im_idx] = True
                imgs_idxs.append(im_idx)
                continue

            views.append(dict(
                img=rgb_image,
                depthmap=depthmap,
                camera_pose=camera_pose,  # cam2world
                camera_intrinsics=intrinsics,
                dataset='Co3d_v2',
                label=osp.join(obj, instance),
                instance=osp.split(impath)[1],
            ))
        return views
```

---

## 5. Full Industrial Workflow Deployment

| Stage              | Recommended Toolchain                          |
| ------------------ | ---------------------------------------------- |
| Robot Control      | ROS MoveIt / native robot SDK                  |
| D435i Capture      | RealSense SDK + custom capture module          |
| Calibration Tools  | OpenCV, Kalibr, Charuco board                  |
| Preprocessing      | Reloc3r `datasets_preprocess` enhanced version |
| Data Validation    | Custom consistency checker                     |
| Training Framework | PyTorch + enhanced Reloc3r dataset loader      |

---

## 6. Future Upgrade Modules

| Module                       | Description                                     |
| ---------------------------- | ----------------------------------------------- |
| Rolling Shutter Compensation | Model rolling shutter effects in network        |
| IMU-Vision Fusion Model      | Use IMU constraints to stabilize network        |
| Scene Augmentation           | Synthetic data generation for domain robustness |
| Online Self-Calibration      | Live fine-tuning hand-eye calibration           |

---

## Key Takeaway

With this full industrial-grade pipeline, your D435i + Robot Arm system can reach state-of-the-art performance for Relative Pose Network training, even without global shutter or LiDAR hardware.

---

## Special Thanks

* This pipeline design references the reloc3r project: [https://github.com/ffrivera0/reloc3r](https://github.com/ffrivera0/reloc3r)
