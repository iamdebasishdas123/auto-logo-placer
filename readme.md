# Automated Consistent Logo Placement on Products

## Problem Statement

At our company, we face the challenge of automatically placing custom logos at user-specified positions onto various products—such as T-shirts, bottles, mugs, etc.—in locations like top, bottom, or top-right. This is achieved using object detection models like **YOLO** and **Detectron**.

The aim is to provide a pipeline where:
- Users upload a product image and a custom logo.
- Users specify the desired placement.
- The system overlays the logo onto the product at the specified spot, without changing original product features or introducing visual artifacts.

---


## Data Structure (COCO Format Example)

Typical COCO-style dataset directory:
---
![Data Directory](assets\image.png)
---
- train.json # COCO-format training   
- val.json # COCO-format validation 
- tshirt/ # Product images for training
- val/ # Product images for validation

## Fine-Tuning Parameters and Their Explanation

| Parameter                                    | Description |
|-----------------------------------------------|-------------|
| **cfg.merge_from_file(...)**                  | Loads a model config from Detectron2’s model zoo, defining architecture and base settings. |
| **cfg.DATASETS.TRAIN**                        | List of training datasets to use, registered in Detectron2. |
| **cfg.DATASETS.TEST**                         | List of validation/test datasets for evaluation during and after training. |
| **cfg.DATALOADER.NUM_WORKERS**                | Number of worker processes for data loading—higher values speed up data loading on powerful machines. |
| **cfg.SOLVER.IMS_PER_BATCH**                  | Batch size—the number of images processed together before an optimizer update. |
| **cfg.SOLVER.BASE_LR**                        | Initial learning rate for the optimizer, controlling step size in learning updates. |
| **cfg.SOLVER.MAX_ITER**                       | Total number of training iterations. Longer training (higher value) may be needed for bigger datasets. |
| **cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE**  | Number of region proposals per image used for loss calculation in detection. |
| **cfg.MODEL.ROI_HEADS.NUM_CLASSES**           | Total number of object classes in your training data (excluding background). |
| **cfg.OUTPUT_DIR**                            | Directory where trained models, logs, and outputs are saved. |

> **How to Use:**  
> Adjust these parameters based on dataset size, machine specs, and how quickly/well you want to tune your model. For example, you need to set NUM_CLASSES to match your dataset’s categories, and MAX_ITER high enough for your data.

---
