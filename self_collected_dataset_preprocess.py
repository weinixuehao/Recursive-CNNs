from genericpath import exists
import glob
import cv2
import os
import shutil
import csv
import random
import numpy as np

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def args_processor():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", help="dataput")
    parser.add_argument("-o", "--output-dir", help="Directory to store results")
    return parser.parse_args()

def orderPoints(pts, centerPt):
    # size = len(pts)
    # centerPt = [0, 0]
    # for pt in pts:
    #     centerPt[0] += pt[0] / size
    #     centerPt[1] += pt[1] / size
    # cv2.circle(img, tuple(list((np.array(centerPt)).astype(int))), 2, (255, 0, 0), 2)
    # cv2.imshow("img", img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    orderedDict = {}
    for pt in pts:
        index = -1
        if pt[0] < centerPt[0] and pt[1] < centerPt[1]:
            index = 0
        elif pt[0] > centerPt[0] and pt[1] < centerPt[1]:
            index = 1
        elif pt[0] < centerPt[0] and pt[1] > centerPt[1]: 
            index = 3
        elif pt[0] > centerPt[0] and pt[1] > centerPt[1]:
            index = 2
        if index in orderedDict:
            targetKeys = [0, 1, 2, 3]
            for i in range(4):
                exists = False
                for key in orderedDict.keys():
                    if key == targetKeys[i]:
                        exists = True
                        break
                if exists is False:
                    index = targetKeys[i]
                    break
        orderedDict[index] = pt
    orderedPts = list(dict(sorted(orderedDict.items())).values())
    assert len(orderedPts) == 4
    return orderedPts

def isAvaibleImg(pts, img, centerPt):
    h, w = img.shape[:2]
    for i, pt in enumerate(pts):
        if pt[0] > (w - 1) or pt[0] < 1:
            return False
        if pt[1] > (h - 1) or pt[1] < 1:
            return False
        if pt[0] == centerPt[0] or pt[1] == centerPt[1]:
            return False
        for _i, _pt in enumerate(pts):
            if i == _i:
                continue
            if abs(pt[0] - _pt[0]) <= 3:
                return False
            if abs(pt[1] - _pt[1]) <= 3:
                return False
    return True

def getCenterPt(pts):
    size = len(pts)
    centerPt = [0, 0]
    for pt in pts:
        centerPt[0] += pt[0] / size
        centerPt[1] += pt[1] / size
    return centerPt

def process(imgpaths, out):
    for imgpath in imgpaths:
        csv_path = imgpath.split(".")[0] + ".csv"
        if os.path.isfile(csv_path) == False:
            continue
        with open(csv_path, "r") as f:
            reader = csv.reader(f, delimiter="\t")
            pts = []
            for i, line in enumerate(reader):
                split = line[0].split(" ")
                pt = [float(split[0]), float(split[1])]
                pts.append(pt)
        assert len(pts) == 4
        img = cv2.imread(imgpath)
        centerPt = getCenterPt(pts)
        if isAvaibleImg(pts, img, centerPt) is False:
            # print(f"{bcolors.WARNING}{imgpath} discard {bcolors.ENDC}")
            continue
        orderedPts = orderPoints(pts, centerPt)
        # for count, pt in enumerate(orderedPts):
        #     cv2.putText(img, f'{count}', (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        # cv2.imshow('img',img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        fileName = os.path.basename(imgpath).split(".")[0]
        out_imgpath = f"{out}/{fileName}.jpg"
        with open(f"{out_imgpath}.csv", "w") as csv_out:
            for pt in orderedPts:
                csv_out.write(f"{pt[0]} {pt[1]}")
                csv_out.write('\n')
        cv2.imwrite(out_imgpath, img)


if __name__ == "__main__":
    args = args_processor()
    imgpaths = glob.glob(f"{args.input_dir}/*.jpg") + glob.glob(
        f"{args.input_dir}/*.png"
    )
    train_dataset_out = f"{args.output_dir}/train"
    test_dataset_out = f"{args.output_dir}/test"
    shutil.rmtree(args.output_dir, ignore_errors=True)
    os.mkdir(args.output_dir)
    os.mkdir(train_dataset_out)
    os.mkdir(test_dataset_out)

    imgpaths_num = len(imgpaths)
    test_num = int(imgpaths_num * 0.2)
    test_imgpaths = imgpaths[0:test_num]
    train_imgpaths = imgpaths[test_num:imgpaths_num]
    process(train_imgpaths, train_dataset_out)
    process(test_imgpaths, test_dataset_out)
