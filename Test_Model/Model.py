import cv2
import torch
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.structures import Instances, Boxes
from detectron2 import model_zoo
from PIL import Image
from transparent_background import Remover
from matplotlib.pyplot import plt


class Model():
    def __init__(self, model_name='COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(model_name))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)
        self.predictor = DefaultPredictor(self.cfg)
        self.metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
        self.remover = Remover()
        
    def load_model(weights_path, num_classes=6):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.WEIGHTS = weights_path
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        return DefaultPredictor(cfg)
    
    def model_output(image_path, predictor):
        image = cv2.imread(image_path)
        outputs = predictor(image)
        instances = outputs["instances"].to("cpu")
        return instances
    def filter_prediction(instances,number_of_class, path):    
        MetadataCatalog.get("val").thing_classes = [
    'top', 'Center', 'Bottom', 'top_left', 'top_right', 'top_bottom'
    ]
        metadata = MetadataCatalog.get("val")
        image=cv2.imread(path)
        class_names = metadata.thing_classes
        
        # Create subplot grid for 6 classes
        fig, axes = plt.subplots(2, 3, figsize=(20, 15))
        axes = axes.ravel()
        
        filter_bounding_boxes=[]
        for class_id in range(number_of_class):
            # Filter instances for current class
            class_mask = instances.pred_classes == class_id
            #print("Class mask",class_mask)
            class_instances = instances[class_mask]
            #print("Class instances",class_instances)
            class_image = image.copy()
            if len(class_instances) > 0:
                # Create new Instances object with only first detection
                single_instance = Instances(
                    instances.image_size,
                    pred_boxes=class_instances.pred_boxes[:1],
                    pred_classes=class_instances.pred_classes[:1],
                    scores=class_instances.scores[:1]
                )
                filter_bounding_boxes.append(single_instance)
                
                # Visualize only this class
                v = Visualizer(class_image[:, :, ::-1], metadata, scale=1)
                out = v.draw_instance_predictions(single_instance.to("cpu"))
                vis_image = out.get_image()[:, :, ::-1]
            else:
                # Show original image with warning text
                vis_image = class_image.copy()
                text = f"No detection: {class_names[class_id]}"
                cv2.putText(vis_image, text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            # Plot in subplot
            axes[class_id].imshow(vis_image)
            axes[class_id].set_title(f"Class {class_id}: {class_names[class_id]}")
            axes[class_id].axis('off')

            plt.tight_layout()
            plt.show()

        return filter_bounding_boxes
    
    def get_bbox_for_class(instances_list, class_name, category_map):
            class_id = get_key_by_value(category_map, class_name)
            for inst in instances_list:
                if int(inst.pred_classes[0]) == class_id:
                    return inst.pred_boxes.tensor.cpu().numpy()[0]  # Returns [x1, y1, x2, y2]
            return None
    def bbox_midpoint(bbox):
        x1, y1 = bbox[0:2]
        x2, y2 = bbox[2:]
        return ((x1 + x2) /2 , (y1 + y2) / 2)
    def resize_logo_to_bbox(logo_img, bbox):
        """
        Resize logo to fit inside the bounding box while maintaining aspect ratio.

        Args:
            logo_path (str): Path to the logo image.
            bbox (tuple): (x1, y1, x2, y2) coordinates of the bounding box.

        Returns:
            PIL.Image: Resized logo image.
        """
        bbox_width = int(bbox[2] - bbox[0])
        bbox_height = int(bbox[3] - bbox[1])

        # Calculate scaling factor to fit logo inside bbox
        logo_w, logo_h = logo_img.size
        scale = min(bbox_width / logo_w, bbox_height / logo_h)
        new_size = (int(logo_w * scale), int(logo_h * scale))

        # Handle Pillow version compatibility for resampling
        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            resample = Image.LANCZOS

        resized_logo = logo_img.resize(new_size, resample)
        return resized_logo,new_size
    
    
def place_logo(main_image_path, logo_path, position):
    
    # Open the main image and logo
    main_img = Image.open(main_image_path).convert("RGBA")
    logo_img = Image.open(logo_path).convert("RGBA")
    
    # # Transparent Background
    # remover = Remover()
    # logo_img = remover.process(logo_img)  # Returns logo with transparent background
    
    

    # # Resize the logo size
    # logo_img,new_size=resize_logo_to_bbox(logo_img,position)
    
    
    
    x, y = bbox_midpoint(position)
    x = int(x - new_size[0] / 2)
    y = int(y - new_size[1] / 2)


    # logo_img = logo_img.convert("RGBA")

    # Use the alpha channel as the mask
    # alpha = logo_img.split()[3]




    # Create a transparent layer the size of the main image
    layer = Image.new("RGBA", main_img.size, (0, 0, 0, 0))
    layer.paste(logo_img, (int(x), int(y)), logo_img)

    # Composite the logo onto the main image
    result = Image.alpha_composite(main_img, layer)
    

    # Convert PIL Image to NumPy array
    result_np = np.array(result.convert("RGB"))
    
    # Convert RGB to BGR for OpenCV
    result_cv = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
    
    plt.figure(figsize=(10,10))
    plt.imshow(result_cv)
    plt.axis("off")
    plt.show()





        