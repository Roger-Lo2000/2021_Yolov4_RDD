#!/bin/bash

while true
do
time=$(date +%M)
direct=$(date +%m%d)
echo $time

if [[ $time == 12 ]];then
	gsutil -m cp gs://roaddamagedetect.appspot.com/$direct/*.jpeg ~/Yolov4_RDD_detection/inputs/
	bash run.sh
	gsutil -m mv -r ./$direct gs://output-rdd/
	gsutil -m mv -r ./label/* gs://auto_label/
	rm -r $direct
	sleep 3600
else
	echo "not yet"
	fi
sleep 3
done
