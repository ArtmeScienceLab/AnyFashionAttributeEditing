
REAL="" #Path to real images
FAKE="" #Path to fake images

python -m evaluation.fid.fid --paths $REAL $FAKE
