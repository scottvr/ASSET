# ASSET - Audio Separation Stemfile Enhancement Toolkit

## Stemprover 
a PoC for misusing image diffusion techniques for audio DSP, taking advantage of the huge investments in compute and dollars that have been made 
training models such as Stable Diffusion. We adapt the ControlNet architecture to the task of removing audio artifacts commonly introduced 
during the stem separation process by tools such as Spleeter and Demucs.  We accomplish this by converting audio in the amplitude 
domain to the frequency domain, storing as an image file, inpainting the image file with one of our trained ControlNet LoRAs for 
specific separation artifacts, and converts back to audio amplitude wave file. As if we weren't already far enough into novelty territory, 
an additional innovation stemming (sorry) from this project is our encoding of phase information in the Alpha channel of the RGB PNG storing the spectrogram.

More to come, once we have results to share.
