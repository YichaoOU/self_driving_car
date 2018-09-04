
import cv2
import numpy as np
import multiprocessing


class grass():
	def __init__(self,cap,vis_flag=True):
		self.cap = cap
		self.Y1m = 150
		self.Y2m = 100
		self.Y3m = 50
		self.slice_width = 50
		self.mask_array = [-1]*3
		sensitivity = 20
		self.lower = np.array([60-sensitivity,40,80], dtype=np.uint8)
		self.upper = np.array([60+sensitivity, 255, 255], dtype=np.uint8)
		self.width = 640
		self.left_lane_point= [-1]*3
		self.right_lane_point = [-1]*3		
		self.vis_flag = vis_flag	
	def detect_green(self,sliced_frame):
		hsv = cv2.cvtColor(sliced_frame, cv2.COLOR_BGR2HSV)
		mask = cv2.inRange(hsv, self.lower, self.upper)
		mask = cv2.medianBlur(mask,9)
		left_point = -1
		right_point = -1
		left_green_list = []
		right_green_list = []
		for j in range(0,int(self.width/2),int(self.slice_width/2)):
			try:
				patch = mask[:,j:j+self.slice_width]
				count = np.count_nonzero(patch)
				# print ("patch",patch)
				# print ("count",count)
				if count < 1000:
					continue
				left_green_list.append(j)

			except:

				continue

		try:	
			# print ("#########################",left_green_list)
			# print ("#########################",max(left_green_list))			
			left_point = max(left_green_list)
		except:
			# print ("left_green_list:",left_green_list)
			a=[]
			# print ("left_point not exist")
		for j in range(int(self.width/2),self.width,int(self.slice_width/2)):
			try:
				patch = mask[:,j:j+self.slice_width]
				count = np.count_nonzero(patch)
				# print ("count",count)
				if count < 1000:
					continue
				right_green_list.append(j)
			except:
				# print ("right########wrong")
				continue
		
		try:		
			right_point = min(right_green_list)
		except:
			# print ("right_green_list:",right_green_list)
			a=[]
			# print ("right_point not exist")

		return mask,left_point,right_point


	def run(self):
		ret, self.frame = self.cap.read()
		if ret == False:
			print ("Nothing read in")
			return 0		# print (self.frame)
		# parallel to compute lines
		slice_array = [self.frame[self.Y1m:self.Y1m+self.slice_width,:,:],self.frame[self.Y2m:self.Y2m+self.slice_width,:,:],self.frame[self.Y3m:self.Y3m+self.slice_width,:,:]]
		# pool = multiprocessing.Pool(processes=3)		
		# result_list = pool.map(self.detect_green, slice_array)
		result_list = [-1]*3
		result_list[1] = self.detect_green(slice_array[1])
		result_list[2] = self.detect_green(slice_array[2])
		result_list[0] = self.detect_green(slice_array[0])
		for i in range(3):
			mask,left_point,right_point = result_list[i]
			self.left_lane_point[i] = left_point
			self.right_lane_point[i] = right_point
			self.mask_array[i] = mask
			if self.vis_flag:
				cv2.imshow('frame'+str(i),mask)
		print ("left:",self.left_lane_point)
		print ("right:",self.right_lane_point)

if __name__ == "__main__":
	cap = cv2.VideoCapture("../TX1_test2.mp4")
	v = grass(cap)
	while True:
		v.run()
		cv2.imshow('frame',v.frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break












