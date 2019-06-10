
# Traditional Machine learning(SVM) employed on age classification
We implement HOG to extract the face region of input images and feed into a SVM machine and Random Forest machine.SVM model is saved because of the better performance. The model is trained by HOG_SVM_2.ipynb and is saved as train_model.m.

To test existing photo, you can use $ python3</> 
To use front camera of che computer, you can use RealTime_main.py


### Dependencies
python 3.6.7<br />
numpy 1.13.3<br />
keras 2.2.4<br />
opencv 3.3.0<br />
imutils 0.4.6

## Running the tests

Please run the main.py in either command line or any python development environment. Follow the instrustion print on screen
to input the image directory to get age prediction. 


## References

[1]’Imdb-Wiki dataset’, https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/<br />
[2] ‘UTK face dataset’, https://susanqq.github.io/UTKFace/<br />
