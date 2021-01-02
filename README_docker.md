# Docker usage
Build the docker image with:
```
build . -t machine-learning
```
Set some volumes to mount with extra resources
```
ML_RESOURCES=()
ML_RESOURCES+=(-v LOCAL_PATH_1:DOCKER_PATH_1)
...
export ML_RESOURCES
```
Then build container with (eg.):
```
docker container rm -f machine-learning
docker create -it \
    -p 8888:8888 \
    -e DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(pwd):/ml-utils ${ML_RESOURCES[@]} \
    --security-opt seccomp=unconfined \
    --cpus 4 \
    --name=machine-learning machine-learning
```

Start container with:
```
docker start -i machine-learning
```
Launch notebook with:
```
jupyter-notebook --ip=0.0.0.0 --allow-root &
```
and paste the last URL printed into a web browser.

# Convert docker image to singularity image file
1. Archive docker image
   ```
   docker save machine-learning -o machine-learning.tar
   ```
2. Copy to fram
3. Build singularity image file
   ```
   singularity build machine-learning.sif docker-archive://machine-learning.tar
   ```