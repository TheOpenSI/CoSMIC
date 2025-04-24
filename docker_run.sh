docker rm -f cosmic

docker run -d -p 3000:3000 \
    --name cosmic \
    opensicbr/cosmic:demo \
    # -v /var/run/docker.sock:/var/run/docker.sock \
    # -v volume_uploads:/app/data/cosmic/backend/uploads \
    # -v volume_configs:/app/scripts/configs \
    # --privileged opensicbr/cosmic:demo