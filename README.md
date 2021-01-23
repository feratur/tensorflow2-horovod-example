# tensorflow2-horovod-example
An example of distributed learning on CPU using Tensorflow 2 and Horovod

#### What it does
* Prepares https://hub.docker.com/r/tensorflow/tensorflow Docker image for distributed learning using Horovod
* Downloads CIFAR10 dataset
* Divides it into 3 (see NUM_WORKERS env var in run.sh) parts
* Launches distributed model learning on CPU (locally using 3 workers (see NUM_WORKERS env var in run.sh))
* Evaluates model accuracy

#### How to run
Just execute the following sequence (on Mac or Linux - only Docker is needed):
```
git clone https://github.com/feratur/tensorflow2-horovod-example.git
cd tensorflow2-horovod-example
docker run --rm -it -v $(pwd):/tf --ipc=host tensorflow/tensorflow bash /tf/run.sh
```
