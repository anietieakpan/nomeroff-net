from nomeroff_net import pipeline
from nomeroff_net.tools import unzip

number_plate_detection_and_reading = pipeline("number_plate_detection_and_reading", 
                                              image_loader="opencv")

(images, images_bboxs, 
 images_points, images_zones, region_ids, 
 region_names, count_lines, 
 confidences, texts) = unzip(number_plate_detection_and_reading(['./data/examples/oneline_images/example1.jpeg', 
                                                                 './data/examples/oneline_images/example2.jpeg']))
 
print(texts)