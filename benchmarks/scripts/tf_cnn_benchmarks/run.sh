set -e

export CUDA_VISIBLE_DEVICES=2

function test() {
  cfg=tf_cnn_benchmarks.py
  model=$1
  batch_size=$2
  variable_update=$3
  num_batches=$4
  data_dir=$5
  python $cfg --model=$model --batch_size=$batch_size --variable_update=$variable_update \
      --num_batches=$num_batches --data_dir=$data_dir | tee logs/${model}-1gpu-${batch_size}-${variable_update}-${num_batches}-realdata.log 2>&1
}

if [ ! -d "logs" ]; then
  mkdir logs
fi

# alexnet
batch_size=(32 64 128 256 512)
#for (( i = 0; i <= 4; i++))
#{
#        test alexnet ${batch_size[$i]} independent 1000
#}
test alexnet 256 independent 200 /home/dl/data/dataset2/imagenet/tensorflowdata

#test alexnet.py 64 alexnet
#test alexnet.py 128 alexnet
#test alexnet.py 256 alexnet
#test alexnet.py 512 alexnet
#
## googlenet
#test googlenet.py 64 googlenet
#test googlenet.py 128 googlenet
#
## smallnet 
#test smallnet_mnist_cifar.py 64 smallnet
#test smallnet_mnist_cifar.py 128 smallnet
#test smallnet_mnist_cifar.py 256 smallnet
#test smallnet_mnist_cifar.py 512 smallnet
