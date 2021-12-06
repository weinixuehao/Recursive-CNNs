import shutil
import os
import glob
import cv2
import numpy as np
import random
import imgaug.augmenters as iaa


def visualizeImg(list=[]):
    for item in list:
        cv2.imshow(item[0], item[1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def transformation(src):
    height, width = src.shape[:2]
    srcPts = np.array([[0, 0], [width, 0], [width, height], [0, height]]).astype(
        np.float32
    )
    float_random_num = random.uniform(0.0, 0.3)
    float_random_num2 = random.uniform(0.0, 0.3)
    float_random_num3 = random.uniform(0.7, 1)
    float_random_num4 = random.uniform(0.0, 0.3)
    float_random_num5 = random.uniform(0.7, 1)
    float_random_num6 = random.uniform(0.7, 1)
    float_random_num7 = random.uniform(0.0, 0.3)
    float_random_num8 = random.uniform(0.7, 1)
    dstPts = np.array(
        [
            [width * float_random_num, height * float_random_num2],
            [width * float_random_num3, height * float_random_num4],
            [width * float_random_num5, height * float_random_num6],
            [width * float_random_num7, height * float_random_num8],
        ]
    ).astype(np.float32)
    M = cv2.getPerspectiveTransform(srcPts, dstPts)
    # warp_dst = cv2.warpPerspective(src, M, (src.shape[1], src.shape[0]), cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 255)
    warp_dst = cv2.warpPerspective(
        src,
        M,
        (width, height),
        flags = cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue = [0, 0, 0, 0]
    )
    # for pt in dstPts:
    #     warp_dst = cv2.circle(warp_dst, (int(pt[0]), int(pt[1])), radius=4, color=(0, 0, 255), thickness=-1)
    # visualizeImg([("warp_dst", warp_dst)])
    return warp_dst


def blending(img1, img2):
    # I want to put logo on top-left corner, So I create a ROI
    rows, cols, channels = img2.shape
    roi = img1[0:rows, 0:cols]
    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 0, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask)
    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg, img2_fg)
    img1[0:rows, 0:cols] = dst
    return img1


def smoothEdge(blended_img):
    up_sample_img = cv2.pyrUp(blended_img)
    blur_img = up_sample_img.copy()
    for i in range(4):
        blur_img = cv2.medianBlur(blur_img, 21)
    down_sample_img = cv2.pyrDown(blur_img)
    return down_sample_img


if __name__ == "__main__":
    dataDir = (
        "/Users/imac-1/workspace/hed-tutorial-for-document-scanning/sample_images/"
    )
    bk_imgs_folder = "background_images"
    rect_folder = "rect_images"
    bk_img_paths = glob.glob(f"{dataDir+bk_imgs_folder}/*.jpg")
    rect_img_paths = glob.glob(f"{dataDir+rect_folder}/*.jpg")
    outputDir = f"{dataDir}output"
    shutil.rmtree(outputDir, ignore_errors=True)
    os.makedirs(outputDir)
    for bk_img_path in bk_img_paths:
        bk_img = cv2.imread(bk_img_path)
        for rect_img_path in rect_img_paths:
            rect_img = cv2.imread(rect_img_path)
            warpedImg = transformation(rect_img)
            resized_img = cv2.resize(warpedImg, bk_img.shape[1::-1])
            blended_img = blending(bk_img.copy(), resized_img)
            final_img = smoothEdge(blended_img)
            rectImgName = os.path.basename(rect_img_path).split(".")
            bkImgName = os.path.basename(bk_img_path).split(".")
            outputFile = f"{outputDir}/{bkImgName[0]}_{rectImgName[0]}.jpg"
            cv2.imwrite(outputFile, final_img)
