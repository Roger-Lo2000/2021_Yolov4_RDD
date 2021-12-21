#!/bin/bash
i=1
#echo "$i"
times=$(date +%m%d)
mkdir $times
while [ $i > 0 ];
do
if [ -f ./inputs/$times"_"$i.jpeg ]; then
	python yolo_image.py -i ~/Yolov4_RDD_detection/inputs/$times"_"$i.jpeg -o ./$times/$times"_"$i.jpeg -T $times -f $times"_"$i
	echo $times"_out"$i done
else
	i=0
	break
fi
i=$((i+1))
done
