# CTReconstruction
Programs demonstrating reconstruction of CT images from sinograms
First example program written in Mathematica shows reduction of intrapetrous artifact by manipulation of the sinogram.
The program uses the RADON transform and inverse RADON transform to process the images.  I do not have raw CT sinograms to work with.

There was a change in handling of DICOM CT data between Mathematica v.11.3 and version 12+.  I have not tested the algorithms on Mathematica 12 +.  I suspect there will be problems because of changes in the way the stored DICOM data is normalized to image data (from 0.0 to 1.0).
