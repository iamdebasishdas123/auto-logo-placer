# data_dir = "/kaggle/input/t-shirt-dataset/Sample Data"  
data_dir = "/kaggle/input/new-way-dataset1/main_training -1" 
register_coco_instances("logo_train", {}, "/kaggle/input/new-way-dataset1/main_training -1/annotations/bottle_tshirt_training.json", "/kaggle/input/new-way-dataset1/main_training -1/train_data")
register_coco_instances("logo_val", {}, "/kaggle/input/new-way-dataset1/main_training -1/annotations/bottle_tshirt_val.json", "/kaggle/input/new-way-dataset1/main_training -1/val_data")
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

cfg.DATASETS.TRAIN = ("logo_train",)
cfg.DATASETS.TEST = ("logo_val",)
cfg.DATALOADER.NUM_WORKERS = 2

cfg.SOLVER.IMS_PER_BATCH = 8
cfg.SOLVER.BASE_LR = 0.002
cfg.SOLVER.MAX_ITER = 3000  # increase this if needed
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 12  # Set this correctly
cfg.OUTPUT_DIR = "./output"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
