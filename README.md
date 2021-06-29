# MIPLAB_FETS
Code from the FETS Challenge for Medical Image Processing Lab, University of Calgary

Credits for parts of the code go to openfl, FETS Challenge organizers and other parties 

To Download the environment use the below command in the installation directory

gdown --id  "10iK3s4HZqvwQ1jvrxsi3Mgx3gigZ9kXY"

For sequential running of the code use runner.sh

runner.sh calls main.py for training

For parallel running use parallel_runner.sh

parallel_runner.sh calls parallel_subrunner.sh as a job array which runs part_main.py for some split of collaborators
