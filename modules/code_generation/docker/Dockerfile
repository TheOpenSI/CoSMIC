FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y python3 python3-pip && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app

# Always run chmod before executing start.sh
ENTRYPOINT ["sh", "-c", "chmod +x /usr/src/app/start.sh && /usr/src/app/start.sh"]