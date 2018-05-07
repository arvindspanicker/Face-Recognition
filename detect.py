import face_recognition
import cv2
import os
import string
import threading
import Queue
import time

#Variables
inputQueue = Queue.Queue()
outputQueue = Queue.Queue()
camera = cv2.VideoCapture('rtmp://192.168.20.144:1935/flash/12:admin:EYE3inapp')



counter = 0 
known_face_names = []
known_face_encodings = []

for root,dirs,files in os.walk('images'):
	if counter >=1:
		break
	known_face_names = dirs
	counter += 1

for item in known_face_names:
	image_name = face_recognition.load_image_file('images/'+ item + '/' + item + '.jpg')
	image_face_encoding = face_recognition.face_encodings(image_name)[0]
	known_face_encodings.append(image_face_encoding)


face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

def get_video():
	while True:
		ret, frame = camera.read()
		inputQueue.put(frame)
		#print inputQueue.qsize()
		

def face_detect():
	process_this_frame = True
	while True:
		if not inputQueue.empty():
			frame = inputQueue.get()
			small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
			rgb_small_frame = small_frame[:, :, ::-1]
			if process_this_frame:
				face_locations = face_recognition.face_locations(rgb_small_frame)
				face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
				face_names = []
				for face_encoding in face_encodings:
					matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
					name = "Unknown"
					if True in matches:
						first_match_index = matches.index(True)
						name = known_face_names[first_match_index]
					face_names.append(name)
			process_this_frame = not process_this_frame
			for (top, right, bottom, left), name in zip(face_locations, face_names):
				top *= 4
				right *= 4
				bottom *= 4
				left *= 4
				cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
				cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), -1)
				font = cv2.FONT_HERSHEY_DUPLEX
				cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
				outputQueue.put(frame)



video_capture = threading.Thread(target = get_video, args = ())
run_face_detection = []
for i in range(0,1):
	t = threading.Thread(target = face_detect , args = ())
	run_face_detection.append(t)

video_capture.daemon = True
for i in range(0,1):
	run_face_detection[i].daemon = True
	
for i in range(0,1):
	run_face_detection[i].start()


video_capture.start()

while True:
	if not outputQueue.empty():
		frame = outputQueue.get()
		cv2.imshow('Output', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break


video_capture.release()
cv2.destroyAllWindows()

