
## Multipose-Detection
The initial steps will be as follows :
<Assuming that the python 3.7 (64 bit is must !) is already installed, OpenCV is functional and libs like numpy are installed using pip3>

-Install python3, pip3 and virtualenv >

-Install git
 (pip install gitpython)
 
-Install TensorFlow from https://www.tensorflow.org/install/pip
 (...While Creating a virtual environment, the path I used was: $virtualenv --system-site-packages C:\Python\Python37\summer\VirtualEnvDir\Venv1.venv 
...In the next step, <for activating the environment>, append the : "\Scripts\activate" to the path shown in the terminal)
 For those still facing difficulty in installing TensorFlow, refer to this link : https://stackoverflow.com/questions/38896424/tensorflow-not-found-using-pip
 
-protobuf, python3-tk will already be installed if using python3.7
 (use >>> import tkinter)
 
-install slidingwindow using pip

-install git for python 
 (follow https://hackernoon.com/install-git-on-windows-9acf2a1944f0
  but only till the "Commonly Asked Questions". Add the path "C:\Program Files\Git\usr\bin" )

-Clone library from (https://github.com/ildoonet/tf-pose-estimation)to python directory (C:\Python\Python37\summer)

-Change dir to the cloned downloaded folder
 ($ cd tf-pose-estimation)

-change directry to the cloned folder
 (C:\Python\Python37\summer\PoseEstimation)

--install Cython using pip install Cython

-Install Visual Studio (for C++ 14) 

-Open the "Requirements.txt" from the "tf-pose-estimation" folder and delete pycocotools because this installation is not meant for windows. Instead, Install pycocotools using:

pip3 install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"



-install requirements 
 ($ pip3 install -r requirements.txt)

-Download "swig" inside C Drive (or your Home dir...look at the Environment Variable.png file in the repo above) from "https://sourceforge.net/projects/swig/files/swigwin/swigwin-4.0.0/swigwin-4.0.0.zip/download?use_mirror=nchc" and follow instructions on "https://simpletutorials.com/c/2135/Installing+SWIG+on+Windows".
(BEWARE! Swig application file will be executed without any window popping up. i.e, as soon as you download swig and click on application file, a screen will flash and disappear. It means it has done its work.) Rest of the steps should be followed from the link above.

-change dir. to pafprocess
 (cd C:\Python\Python37\summer\PoseEstimation\tf_pose) and execute "swig -python -c++ pafprocess.i && python setup.py build_ext --inplace"

-install wget using : "https://www.addictivetips.com/windows-tips/install-and-use-wget-in-windows-10/". But while in the environment variable step, don't follow his step and add path in "SYSTEM VARIABLE AND NOT USER VARIABLE !". For changes to appear, re-open cmd.

-I moved the image to be processed in the same dir. where run.py was saved in the PoseEstimation folder. and then run the command : 
python run.py --model=mobilenet_thin --resize=432x368 --image=p1.jpg
(Although the histogram was not visible, due to some backend issue of matplotlib library ... can be solved here "https://www.pyimagesearch.com/2015/08/24/resolved-matplotlib-figures-not-showing-up-or-displaying/") : FOR LINUX USERS

"https://stackoverflow.com/a/56422557/9625777" FOR WINDOWS USERS

-To analyse the real time webcam, :$ python run_webcam.py --model=mobilenet_thin --resize=432x368 --camera=0
-The latest version boasts of running Pose Detection and Object Detection simultaneously. To do so, run :
python run_webcam.py --model=mobilenet_thin --resize=432x368 --camera=0 --prototxt MobileNetSSD_deploy.prototxt.txt --mmodel MobileNetSSD_deploy.caffemodel

