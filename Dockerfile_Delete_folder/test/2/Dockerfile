FROM ubuntu:14.04
LABEL maintainer="test"
COPY entrypoint.sh /mnt/
RUN chmod +x /mnt/entrypoint.sh
ENTRYPOINT ["/mnt/entrypoint.sh"]
