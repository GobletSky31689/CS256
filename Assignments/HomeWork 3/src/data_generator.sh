#!/usr/bin/env bash
BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export BASEDIR
echo
echo "Starting train_1_Generator"
chmod a+x $BASEDIR"/Scripts/train_1_Generator.sh"
/bin/bash $BASEDIR"/Scripts/train_1_Generator.sh"

echo "Starting train_2_Generator"
chmod a+x $BASEDIR"/Scripts/train_2_Generator.sh"
/bin/bash $BASEDIR"/Scripts/train_2_Generator.sh"

echo "Starting train_3_Generator"
chmod a+x $BASEDIR"/Scripts/train_3_Generator.sh"
/bin/bash $BASEDIR"/Scripts/train_3_Generator.sh"


echo "Starting train_4_Generator"
chmod a+x $BASEDIR"/Scripts/train_4_Generator.sh"
/bin/bash $BASEDIR"/Scripts/train_4_Generator.sh"

echo "Generating test_1_Generator"
chmod a+x $BASEDIR"/Scripts/test_1_Generator.sh"
/bin/bash $BASEDIR"/Scripts/test_1_Generator.sh"

echo "Generating test_2_Generator"
chmod a+x $BASEDIR"/Scripts/test_2_Generator.sh"
/bin/bash $BASEDIR"/Scripts/test_2_Generator.sh"

echo "Generating test_3_Generator"
chmod a+x $BASEDIR"/Scripts/test_3_Generator.sh"
/bin/bash $BASEDIR"/Scripts/test_3_Generator.sh"

echo "Entire Data set Generated"
echo