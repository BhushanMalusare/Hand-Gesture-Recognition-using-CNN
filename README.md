# Hand Gesture Recognition

## Steps to implement code

1.  Create a main folder that will contain all the code and files required to run our inference.py file. e.g Hand gesture recognition

2.  Create 2 sub folders in "Hand Gesture recognition" folder. Name one of the folder as "demo" and other one as "output"

    * The demo folder will contain the files of our finetuned objection model.

    * Output folder is the folder in which we will store our result images which has detections in it. These images will be generated from inference.py file.

3. Create 3 more folders in "demo" folder named :- 
    * fine_tuned_model (for storing our fine tuned model files like checkpoint files,pipeline.config and saved_model.pb file)
    * images (This folder will contain the images on which we want to run our model inference.)
    * tfrecords (This file will contain the label.pbtxt, test.record and train.record file.)

4. Put the inference.py file in the same "demo" folder.

5. After running the inference.py file the resultant images will be stored in "output" folder.

## Steps to dockerize our code

1. As we saw above, we created a parent folder named "Hand gesture recognition". We need to place our docker file in this folder only.

2. The docker file will contain all the necessary libraries that are required to run our inference.py file.

3. Steps to dockerize our code:-
    * First start the docker application
    * Run the command after navigating to the working folder to create a docker image which copies all the contents from demo (inference.py,        tfrecords, images,fine_tuned_model) to a new folder 'model' (just in the image)
        - Command - **docker build -t newimage**
    * After creating the 'newimage' we start a docker container using the following command (takes some time for installations)
        - Command - **docker run -ti newimage /bin/bash**
    * After this command is executed create a directory with the name 'output' (same name as in inference.py file) using 'mkdir output'.
    * Navigate to the model directory and execute the inference.py using
        - Command - **python3 Inference.py**
    * After running the inference.py output images are generated and they are stored in the output folder that is created previously.
    * The output images should be copied to the host from the container, open a new terminal and use the below command with paths file paths accordingly,
            - Command - **docker cp {container_id}:output {path_in_host}**
    * The images can be seen in output folder in the {Hand gesture recognition} directory in the host machine.





