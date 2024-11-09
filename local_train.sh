export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=$PYTHONPATH:./

gpus=(${CUDA_VISIBLE_DEVICES//,/ })
gpu_num=${#gpus[@]}
echo "number of gpus: "${gpu_num}

config=projects/configs/$1.py
checkpoint=$2

if [ ${gpu_num} -gt 1 ]
then
    bash ./tools/dist_train.sh \
        ${config} \
        ${gpu_num} \
        --work-dir=work_dirs/$1
else
    python ./tools/train.py \
        ${config}
fi
# bash local_train.sh sparse4d_r101_H1.py