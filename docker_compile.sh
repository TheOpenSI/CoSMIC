docker rm -f open-webui cosmic
docker rmi -f opensicbr/cosmic:demo open-webui:latest
docker build -t opensicbr/cosmic:demo .