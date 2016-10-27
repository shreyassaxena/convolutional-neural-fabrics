CAFFE=../../caffe_own/build/tools/caffe  
SOLVER=CPU_solver_6x9_27channels_LR_0.01.prototxt

$CAFFE train -solver $SOLVER 2>&1 | tee result.txt

