
import cv2
import depthai as dai
from easyocr import Reader
import numpy as np
reader = Reader(['en'])
import time

downscaleColor = False
fps = 30
# The disparity is computed at this resolution, then upscaled to RGB resolution
monoResolution = dai.MonoCameraProperties.SensorResolution.THE_400_P

# Create pipeline
pipeline = dai.Pipeline()
device = dai.Device()
queueNames = []

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
ctrl = dai.CameraControl()
ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO)
#ctrl.setAutoWhiteBalanceMode(dai.CameraControl.AutoWhiteBalanceMode.FLUORESCENT)
#left = pipeline.create(dai.node.MonoCamera)
#right = pipeline.create(dai.node.MonoCamera)
#stereo = pipeline.create(dai.node.StereoDepth)

rgbOut = pipeline.create(dai.node.XLinkOut)
#depthOut = pipeline.create(dai.node.XLinkOut)

rgbOut.setStreamName("rgb")
queueNames.append("rgb")
#depthOut.setStreamName("depth")
#queueNames.append("depth")

#Properties
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

#camRgb.setAs(False)
camRgb.setFps(fps)
if downscaleColor: camRgb.setIspScale(4, 13)

# Linking
camRgb.isp.link(rgbOut.input)
"""left.out.link(stereo.left)
right.out.link(stereo.right)
stereo.depth.link(depthOut.input)"""

	# Connect to device and start pipeline
with device:
	print(device.getConnectedCameraFeatures())
	device.startPipeline(pipeline)

	frameRgb = None
	depthFrame = None

	#device.setIrLaserDotProjectorBrightness(0) # in mA, 0..1200
	#device.setIrFloodLightBrightness(0) # in mA, 0..1500

	#x_shape, y_shape = (1280,800)
	#x_shape, y_shape = (1248, 936) #with set outputsize
	latestPacket = {}
	latestPacket["rgb"] = None
	latestPacket["depth"] = None
	HFOV = np.deg2rad(60.0)
	#HFOV = np.deg2rad(90.0)
	frame_count = 0
	start_time = time.time()

	while True:

		# Perform object detection every odd frame
		#if counter % 2 == 0:
		queueEvents = device.getQueueEvents(("rgb"))
		for queueName in queueEvents:
			packets = device.getOutputQueue(queueName).tryGetAll()
			if len(packets) > 0:
				latestPacket[queueName] = packets[-1]
			
		if latestPacket["rgb"]:
			frameRgb = latestPacket["rgb"].getCvFrame()
			# setting up parameters
			start = time.time()
			
			results_top = reader.readtext(frameRgb, paragraph=True, y_ths=.1)

			# Display the frame
			cv2.imshow("framergb", cv2.resize(frameRgb, (0, 0), fx=0.5, fy=0.5))
			print(results_top)


		if cv2.waitKey(1) == ord('q'):

			break
	cv2.destroyAllWindows()  # Closes displayed windows
