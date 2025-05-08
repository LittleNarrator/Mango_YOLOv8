from ultralytics import YOLO

model = YOLO(r"E:\Project-xiang\Xjy_mango\ultralytics-8.2.81\ultralytics-8.2.81\runs\detect\train4\weights\best.pt") # pass any model type
model.val(data="flowers.yaml")
