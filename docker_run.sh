docker rm -f cosmic

docker run -it \
    --name cosmic \
    --network=host \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v volume_uploads:/app/data/cosmic/backend/uploads \
    -v volume_configs:/app/scripts/configs \
    --privileged opensicbr/cosmic:demo