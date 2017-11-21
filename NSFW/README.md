## What is NSFW?
One of important requirement required for filtering a video/image is to make sure it doesn't have any pornographic/adults only contents. Typically these contents are categorized as Not Safe For Work(NSFW).

## What we do?
We automatically detect the suitable/safe for work (NSFW) for a given image. 

Defining NSFW material is subjective and the task of identifying these images is non-trivial. Moreover, what may be objectionable in one context can be suitable in another. For this reason, the model we describe below focuses only on one type of NSFW content: pornographic images. The identification of NSFW sketches, cartoons, text, images of graphic violence, or other types of unsuitable content is not addressed with this model.


> NOTE: We have used the opensourced DeepNN model (open_nsfw) from yahoo as base for this component.

**Technology:** Deep Neural Networks, OpenCV

**References: **
- https://github.com/yahoo/open_nsfw
- https://yahooeng.tumblr.com/post/151148689421/open-sourcing-a-deep-learning-solution-for
