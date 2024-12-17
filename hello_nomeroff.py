from nomeroff_net import pipeline
from nomeroff_net.tools import unzip

number_plate_detection_and_reading = pipeline("number_plate_detection_and_reading", 
                                              image_loader="opencv")

(images, images_bboxs, 
 images_points, images_zones, region_ids, 
 region_names, count_lines, 
 confidences, texts) = unzip(number_plate_detection_and_reading(['/home/aniix/alprs/nomeroff-net/images/Cars420.png']))
 
print(texts)