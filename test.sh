#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

./build.sh

VOLUME_SUFFIX=$(dd if=/dev/urandom bs=32 count=1 | md5sum | cut --delimiter=' ' --fields=1)
MEM_LIMIT="30g"  # Maximum is currently 30g, configurable in your algorithm image settings on grand challenge

docker volume create hybrid_cnn-output-$VOLUME_SUFFIX

echo "Volume created, running evaluation"
# Do not change any of the parameters to docker run, these are fixed
docker run --rm \
        --memory="${MEM_LIMIT}" \
        --memory-swap="${MEM_LIMIT}" \
        --network="none" \
        --cap-drop="ALL" \
        --security-opt="no-new-privileges" \
        --shm-size="128m" \
        --pids-limit="256" \
        --gpus="all" \
        -v $SCRIPTPATH/test/input/:/input/ \
        -v hybrid_cnn-output-$VOLUME_SUFFIX:/output/ \
        hybrid_cnn

echo "Evaluation done, checking results"

docker build -f Dockerfile.eval -t hybrid_cnn_eval .
docker run --rm -it \
        -v hybrid_cnn-output-$VOLUME_SUFFIX:/output/ \
        -v $SCRIPTPATH/test/expected_output_nnUNet/:/expected_output/ \
        hybrid_cnn_eval python3 -c """
import SimpleITK as sitk
import os
file = os.listdir('/output/images/automated-petct-lesion-segmentation')[0]
output = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join('/output/images/automated-petct-lesion-segmentation/', file)))
expected_output = sitk.GetArrayFromImage(sitk.ReadImage('/expected_output/images/TCIA_001.nii.gz'))
mse = sum(sum(sum((output - expected_output) ** 2)))
if mse == 0.0:
    print('Test passed!')
else:
    print('Test failed!, MSE: ' + str(mse))
"""

docker volume rm hybrid_cnn-output-$VOLUME_SUFFIX
