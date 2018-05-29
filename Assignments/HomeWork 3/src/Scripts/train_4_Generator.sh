#!/usr/bin/env bash

rm -rf $BASEDIR"/Train_4"
mkdir $BASEDIR"/Train_4"
echo
echo "Generating data for Train_4 folder with 60,000 items of NON-STICKY class."
rm -rf $BASEDIR"/out1.txt"
rm -rf $BASEDIR"/out2.txt"
python $BASEDIR"/sticky_snippet_generator.py" 30000 0 0 $BASEDIR"/out1.txt"
python $BASEDIR"/sticky_snippet_generator.py" 30000 0 0 $BASEDIR"/out2.txt"
cat $BASEDIR"/out1.txt" $BASEDIR"/out2.txt" > $BASEDIR"/Train_4/file1.txt"
echo "Dataset of Train_4 Generated"
echo
echo