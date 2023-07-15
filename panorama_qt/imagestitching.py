import imutils
import numpy as np
from cv2 import (
	imread, imshow, waitKey, Stitcher_create, Stitcher_SCANS, Stitcher_PANORAMA, split, merge, copyMakeBorder
)
import cv2
class OralImgStitch(object):
	def __init__(self, img_list):
		super(OralImgStitch, self).__init__()
		self.img_list = img_list
		self.images = []
		for imagePath in self.img_list:
			img = imread(imagePath)
			self.images.append(img)

	def stitchImage(self, stitch_mode=Stitcher_PANORAMA, is_crop=True):
		# 可以切换拼接模式
		self.stitcher = Stitcher_create(stitch_mode)
		(status, stitched) = self.stitcher.stitch(self.images)
		if is_crop:
			stitched = self.cropimage(stitched)
		# 使用opencv查看较大尺寸图片时非常不友好，这里替换成了使用matplotlib查看并保存
		# if self.status == 0:
		# 	imshow("result", self.stitched)
		# 	waitKey(0)
		b,g,r = split(stitched)
		output = merge([r,g,b])
		return output, status

	def cropimage(self, stitched_img):
		self.input_img = cv2.copyMakeBorder(stitched_img, 2, 2, 2, 2,
								  cv2.BORDER_CONSTANT, (0, 0, 0))

		# convert the stitched image to grayscale and threshold it
		# such that all pixels greater than zero are set to 255
		# (foreground) while all others remain 0 (background)
		gray = cv2.cvtColor(self.input_img, cv2.COLOR_BGR2GRAY)
		thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

		# find all external contours in the threshold image then find
		# the *largest* contour which will be the contour/outline of
		# the stitched image
		cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
							cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		c = max(cnts, key=cv2.contourArea)

		# allocate memory for the mask which will contain the
		# rectangular bounding box of the stitched image region
		mask = np.zeros(thresh.shape, dtype="uint8")
		(x, y, w, h) = cv2.boundingRect(c)
		cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

		# create two copies of the mask: one to serve as our actual
		# minimum rectangular region and another to serve as a counter
		# for how many pixels need to be removed to form the minimum
		# rectangular region
		minRect = mask.copy()
		sub = mask.copy()

		# keep looping until there are no non-zero pixels left in the
		# subtracted image
		while cv2.countNonZero(sub) > 0:
			# erode the minimum rectangular mask and then subtract
			# the thresholded image from the minimum rectangular mask
			# so we can count if there are any non-zero pixels left
			minRect = cv2.erode(minRect, None)
			sub = cv2.subtract(minRect, thresh)

		# find contours in the minimum rectangular mask and then
		# extract the bounding box (x, y)-coordinates
		cnts = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL,
								cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		c = max(cnts, key=cv2.contourArea)
		(x, y, w, h) = cv2.boundingRect(c)

		# use the bounding box coordinates to extract the our final
		# stitched image
		output_img = self.input_img[y:y + h, x:x + w]
		return output_img