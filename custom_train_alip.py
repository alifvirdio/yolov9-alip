from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8m.yaml")  # build a new model from scratch
model = YOLO("yolov9c.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="D:\KULIAHAJA\SEMESTER7_PROYEKAKHIR\Dataset\DatasetKBD\DatasetKBD.v1-70-15-15-v9.yolov9\data.yaml", epochs=100, imgsz=(640,640), plots=True)  # train the model
# metrics = model.val()  # evaluate model performance on the validation set
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
# model.export(format="onnx", opset=17)  # export the model to ONNX format


