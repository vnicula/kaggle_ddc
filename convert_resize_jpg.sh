#!/bin/sh
for i in *.jpg; do convert "$i" -resize 224x224! "${i%.*}.png"; done

# Pad to 256
# mogrify -extent 256x256 -background black train/dfdc_train_part_10/256/0/bcipnhosil.mp4_8.png