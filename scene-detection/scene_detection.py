def split_scenes(video_path):
	#split into scenes
	scene_list = []
	
	# Usually use one detector, but multiple can be used.
	detector_list = [
	    scenedetect.detectors.ContentDetector()
	]

	video_framerate, frames_read = scenedetect.detect_scenes_file(video_path, scene_list, detector_list)

	# scene_list now contains the frame numbers of scene boundaries.
	print scene_list
	print video_framerate
	print frames_read
	video_fps = video_framerate
	# create new list with scene boundaries in milliseconds instead of frame #.
	scene_list_msec = [(1000.0 * x) / float(video_fps) for x in scene_list]
	print scene_list_msec
	# create new list with scene boundaries in timecode strings ("HH:MM:SS.nnn").
	scene_list_tc = [scenedetect.timecodes.get_string(x) for x in scene_list_msec]
	print scene_list_tc


if __name__ == '__main__':
	video_path = '/home/dev/figo_ad.flv'  # Path to video file.
	split_scenes(video_path)
