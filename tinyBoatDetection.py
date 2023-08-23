import os
import time
import numpy as np
from numpy import asarray

from ultralytics import YOLO, checks
from roboflow import Roboflow
import torch

from sahi.utils.yolov8 import download_yolov8s_model
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import (get_prediction, 
                          get_sliced_prediction, 
                          predict)

from PIL import Image, ImageDraw as D
# import cv2
# from IPython.display import Image
from matplotlib import image

import argparse

# dict_keys(['name', 'version', 'model_format', 'location'])

API_KEY = ""
MODEL_PATH = ""
CONFIDENCE = 0.25

def performChecks():
    print('Torch with Cuda support version: {}'.format(torch.__version__))
    print('GPU available: {}'.format(torch.cuda.is_available()))
    print('Performing ultralytics check...')
    print(checks())

def downloadDataset(the_workspace = "insight-robotics", project_name="boat-detection-sample", version_number = 6, model_format="yolov8"):
    rf = Roboflow(api_key = API_KEY)
    project = rf.workspace(the_workspace = the_workspace).project(project_name = project_name)
    dataset = project.version(version_number=version_number).download(model_format = model_format)
    print('Dataset \'{}\' saved at \'{}\''.format(dataset.name, dataset.location))

def trainModel(model = "yolov8n.pt", data = "./datasetv6/data.yaml", epochs = 260, imgsz = 640, device = 0):
    model = YOLO('yolov8n-p2.yaml').load('yolov8n.pt')
    try:
        model.train(data = data, epochs = epochs, imgsz = imgsz, device = device)
    except: # pylint: disable=bare-except
        model.train(data = data, epochs = epochs, imgsz = imgsz, device = "cpu")

def runTiledInference(testingImagesFilePath, savePredictionsFilePath, xPieces = 3, yPieces = 3):
    for imageName in os.listdir(testingImagesFilePath):
        t0 = time.time()

        filename, file_extension = os.path.splitext(imageName)
        testingImagePath = os.path.join(testingImagesFilePath, imageName)
        im = Image.open(testingImagePath)

        imgwidth, imgheight = im.size
        height = imgheight // yPieces
        width = imgwidth // xPieces
        tileList = []

        model = YOLO(MODEL_PATH)

        for i in range(0, yPieces):
            for j in range(0, xPieces):
                box = (j * width, i * height, (j + 1) * width, (i + 1) * height)
                tile = im.crop(box)
                try: ## Draw bounding box
                    results = model.predict(source = tile, conf = CONFIDENCE, save = False, verbose = False)

                    boxes = results.__getitem__(0).boxes.xyxy
                    confidence = results.__getitem__(0).boxes.conf
                    font_scale = 1
                    thickness = 1
                    color = (191,62,255)

                    for box, conf in zip(boxes, confidence):
                        if conf >= int(CONFIDENCE)/100:
                            draw = D.Draw(tile)
                            draw.rectangle(xy = [(int(box[0]), int(box[1])), (int(box[2]), int(box[3]))], outline = "red", width = 1)

                    tileList.append(asarray(tile))

                except:
                    tileList.append(asarray(tile))
 
        ## Reassemble tiles
        v = []

        for i in range(0, yPieces):
            h = []

            for j in range(0, xPieces):
                h.append(tileList.pop(0))

            v.append(np.hstack(tuple(h)))
        
        i = np.vstack(tuple(v))
        im = Image.fromarray(i)

        for i in range(1, yPieces):
            for j in range(1, xPieces):
                centreBox = (j * width - width//2, i * height - height//2, j * width + width//2, i * height + height//2)
                tile = im.crop(centreBox)
                results = model.predict(source = tile, conf = CONFIDENCE, save = False, verbose = False)

                boxes = results.__getitem__(0).boxes.xyxy
                confidence = results.__getitem__(0).boxes.conf

                for box, conf in zip(boxes, confidence):
                    if conf >= int(CONFIDENCE)/100:
                        draw = D.Draw(im)
                        draw.rectangle(xy = [(int(box[0]) + j * width - width//2, int(box[1]) + i * height - height//2), (int(box[2]) + j * width - width//2, int(box[3]) + i * height - height//2)], outline = "red", width = 1)

        im.save(os.path.join(savePredictionsFilePath, filename + file_extension))

        ft = time.time() - t0
        print('{} saved. Inference took {} seconds'.format(filename + file_extension, ft))

def overlapTilesWithSAHI(testingImagesFilePath, savePredictionsFilePath):
    yolov8_model_path = MODEL_PATH
    download_yolov8s_model(yolov8_model_path)

    detection_model = AutoDetectionModel.from_pretrained(
        model_type = "yolov8",
        model_path = yolov8_model_path,
        confidence_threshold = CONFIDENCE,
        device = "cuda:0",
    )

    for imgName in os.listdir(testingImagesFilePath):
        imgarray = image.imread(os.path.join(testingImagesFilePath, imgName))
        height, width = list(imgarray.shape)[0], list(imgarray.shape)[1]
    
        result = get_sliced_prediction(
            os.path.join(testingImagesFilePath, imgName),
            detection_model,
            slice_height = int(height//3 * 1.2),
            slice_width = int(width//3 * 1.2),
            overlap_height_ratio = 0.2,
            overlap_width_ratio = 0.2
        )

        result.export_visuals(export_dir = savePredictionsFilePath, file_name = os.path.splitext(imgName)[0])

        print("{} saved.".format(os.path.splitext(imgName)[0]))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest = "command", help = "program functionalities")

    parser_checks = subparsers.add_parser('checks', help = "perform ultralytics + pytorch with cuda checks")
    parser_checks.set_defaults(func = performChecks)
    # py boatDetection.py checks

    parser_downloads = subparsers.add_parser('downloads', help = "download dataset")
    parser_downloads.add_argument('--the_workspace', type = str, default = "insight-robotics", required = False, help = "workspace on roboflow")
    parser_downloads.add_argument('--project_name', type = str, default = "boat-detection-sample", required = False, help = "project name on roboflow")
    parser_downloads.add_argument('--version_number', type = int, default = 6, required = False, help = "version of project on roboflow")
    parser_downloads.add_argument('--model_format', type = str, default = "yolov8", required = False, help = "model format")
    parser_downloads.set_defaults(func = downloadDataset)
    # py boatDetection.py downloads

    parser_train = subparsers.add_parser('train', help = "training mode")
    parser_train.add_argument('--model', type = str, default = "yolov8n.pt", required = False, help = "model used for training")
    parser_train.add_argument('--data', type = str, required = True, help = "relative path to data.yaml file")
    parser_train.add_argument('--epochs', type = int, default = 50, required = False, help = "number of epochs for training")
    parser_train.add_argument('--imgsz', type = int, default = 640, required = False, help = "dimension of square image")
    parser_train.add_argument('--device', type = str, default = "cpu", choices=['cpu', '0'], required = False, help = "cpu or \'0\' for cuda enabled gpu")
    parser_train.set_defaults(func = trainModel)
    # py boatDetection.py train --data=./datasetv6/data.yaml --epochs=260 --device=0

    parser_infer = subparsers.add_parser('infer', help = "infer mode")
    parser_infer.add_argument('--testingImagesFilePath', type = str, required = True, help = "relative path to testing images folder")
    parser_infer.add_argument('--savePredictionsFilePath', type = str, required = True, help = "relative path to saved predictions folder")
    parser_infer.add_argument('--xPieces', type = int, default = 3, required = False, help = "number of horizontal tiling pieces")
    parser_infer.add_argument('--yPieces', type = int, default = 3, required = False, help = "number of vertical tiling pieces")
    parser_infer.set_defaults(func = runTiledInference)
    # py boatDetection.py infer --testingImagesFilePath=collectedImages/dataset2/test --savePredictionsFilePath=predictions/testargparse

    args = parser.parse_args()
    args_ = vars(args).copy()
    args_.pop("command", None)
    args_.pop("func", None)
    args.func(**args_)



    # testingImagesFilePath = "collectedImages"
    # predictionFile3x3 = "predictions/predictions3x3(bestv6)"
    # predictionFileSAHI = "predictions/predictionssahi(bestv6)"

    # performChecks()
    # downloadDataset()

    # trainModel()

    # for subfolder in os.listdir(testingImagesFilePath):
    #     subpath = os.path.join(testingImagesFilePath, subfolder, "test")
    #     runTiledInference(testingImagesFilePath = subpath, savePredictionsFilePath = predictionFile3x3)
    #     # overlapTilesWithSAHI(testingImagesFilePath = subpath, savePredictionsFilePath = predictionFileSAHI)

    # runTiledInference(testingImagesFilePath = "", savePredictionsFilePath = "")
    # overlapTilesWithSAHI(testingImagesFilePath = "", savePredictionsFilePath = "")