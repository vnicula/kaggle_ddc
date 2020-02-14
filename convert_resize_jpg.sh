#!/bin/sh
for i in *.jpg; do convert "$i" -resize 224x224! "${i%.*}.png"; done