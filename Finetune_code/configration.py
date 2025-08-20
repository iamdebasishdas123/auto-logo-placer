# Step 1: Install and import required libraries

import detectron2

from detectron2.utils.logger import setup_logger
setup_logger()

# Imports
import os
import cv2
import random
import torch
import numpy as np
import matplotlib.pyplot as plt

from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog