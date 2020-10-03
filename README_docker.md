Build the docker image with:
```
build . -t machine-learning
```
Then build container with:
```
docker container rm -f machine-learning
docker create -it \
    -p 8888:8888 \
    -e DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(pwd):/root/machine-learning \
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
