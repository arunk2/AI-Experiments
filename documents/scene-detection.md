## What is scene detection?
Scene Detection - automatically detect the transitions between shots/scenes in a video. 

**Definition from wikipedia:** Shot transition detection (or simply shot detection) also called cut detection is a field of research of video processing. Its subject is the automated detection of transitions between shots in digital video with the purpose of temporal segmentation of videos.

## Scene Detection Algorithms
Any scene algorithm falls under 2 category.
- Threshold based detector
- Content based detector

## Threshold  Detector
Threshold-based scene detector - is the traditional scene detection method. This is done by comparing intensity of the current frame with a set threshold, and record a transition when it crosses. This works under the assumption of the existence of a black frame between 2 scenes. e.g. 'ffmpeg blackframe' filter uses this approach.

## Content based Detector
Content-aware scene detector - is little time consuming process, where it converts every frame to YUV space and takes two consecutive frames, and check for their belonging to the same scene, or different scenes.
The content-aware scene detector finds areas where the difference between two subsequent frames exceeds the threshold value.

In our case we used content based detector, which gives us good results. We use 'pySceneDetect' library, an free and open source implementation for our purpose. It is coming with BSD license. If needed we can implement the algorithm as well.


## Sample code:

```
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

```

> NOTE: Most algorithms achieve good results with hard cuts and fail in some soft cuts.

**Reference: **
- https://github.com/Breakthrough/PySceneDetect
- https://pyscenedetect.readthedocs.io/en/latest/
