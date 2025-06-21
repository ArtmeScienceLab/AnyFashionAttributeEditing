
REAL="" #Path to real images
FAKE="" #Path to fake images

fidelity --gpu 0 --kid --kid-subset-size 30 --input1 $REAL --input2 $FAKE
