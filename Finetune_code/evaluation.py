import os
import json
import numpy as np
import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
import matplotlib.pyplot as plt
import cv2
import torch

# Initialize logger
setup_logger()
logger = setup_logger(name=__name__)

# Configuration setup
def setup_config(config_path, weights_path, confidence_threshold=0.5, num_classes=12):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_path))
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes  # CRITICAL FIX
    return cfg

# Evaluation function
def evaluate_model(cfg, dataset_name, output_dir):
    # Create predictor
    predictor = DefaultPredictor(cfg)
    
    # Build data loader
    data_loader = build_detection_test_loader(cfg, dataset_name)
    
    # Initialize evaluator
    evaluator = COCOEvaluator(
        dataset_name,
        output_dir=output_dir,
        distributed=False,
        use_fast_impl=False
    )
    
    # Run evaluation
    results = inference_on_dataset(
        predictor.model, 
        data_loader, 
        evaluator
    )
    
    # Save results
    results_file = os.path.join(output_dir, f"results_{dataset_name}.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    return results

# Calculate additional metrics
def calculate_additional_metrics(results):
    bbox_results = results.get('bbox', {})
    metrics = {
        'AP': bbox_results.get('AP', 0),
        'AP50': bbox_results.get('AP50', 0),
        'AP75': bbox_results.get('AP75', 0),
        'APs': bbox_results.get('APs', 0),
        'APm': bbox_results.get('APm', 0),
        'APl': bbox_results.get('APl', 0),
        'ARmax1': bbox_results.get('ARmax1', None),
        'ARmax10': bbox_results.get('ARmax10', None),
        'ARmax100': bbox_results.get('ARmax100', None),
        'ARs': bbox_results.get('ARs', None),
        'ARm': bbox_results.get('ARm', None),
        'ARl': bbox_results.get('ARl', None),
    }
    # Remove keys with None values to avoid confusion
    metrics = {k: v for k, v in metrics.items() if v is not None}
    metrics['mean_IoU'] = 0  # Placeholder if you don't calculate IoU separately
    return metrics


# Visualization function
def visualize_predictions(cfg, dataset_name, num_images=3):
    predictor = DefaultPredictor(cfg)
    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)
    
    for i, d in enumerate(dataset_dicts[:num_images]):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        
        # Visualize predictions
        v = Visualizer(
            im[:, :, ::-1], 
            metadata=metadata, 
            scale=0.8
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        
        # Save visualization
        os.makedirs("visualizations", exist_ok=True)
        vis_path = f"visualizations/{dataset_name}_{i}.png"
        cv2.imwrite(vis_path, out.get_image()[:, :, ::-1])
        logger.info(f"Saved visualization to {vis_path}")

# Main evaluation function
def main():
    # Configuration
    CONFIG_PATH = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    WEIGHTS_PATH = "/kaggle/working/output/model_final.pth"
    TRAIN_DATASET = "logo_train"
    TEST_DATASET = "logo_val"
    OUTPUT_DIR = "./evaluation_output"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Setup config
    cfg = setup_config(CONFIG_PATH, WEIGHTS_PATH)
    
    # Evaluate on test data
    logger.info("Evaluating on TEST dataset...")
    test_results = evaluate_model(cfg, TEST_DATASET, OUTPUT_DIR)
    test_metrics = calculate_additional_metrics(test_results)
    
    # Evaluate on training data
    logger.info("Evaluating on TRAIN dataset...")
    train_results = evaluate_model(cfg, TRAIN_DATASET, OUTPUT_DIR)
    train_metrics = calculate_additional_metrics(train_results)
    
    # Print results comparison
    logger.info("\n=== Evaluation Results ===")
    logger.info(f"{'Metric':<15} | {'Train':<12} | {'Test':<12}")
    for metric in test_metrics:
        logger.info(f"{metric:<15} | {train_metrics.get(metric, 0):<10.4f} | {test_metrics[metric]:<10.4f}")
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    visualize_predictions(cfg, TEST_DATASET)
    visualize_predictions(cfg, TRAIN_DATASET)

if __name__ == "__main__":
    main()
