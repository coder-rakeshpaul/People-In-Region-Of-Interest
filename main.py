from get_area import get_area
from detect import Detect
input_video = 'production_id 4910145 (1080p).mp4'
roi = get_area(input_video= input_video)

track = Detect(classes= 0 , input_video= input_video , roi= roi)
track.plotbb()