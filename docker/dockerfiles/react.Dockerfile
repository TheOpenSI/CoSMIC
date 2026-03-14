FROM node:22-trixie AS base
WORKDIR /app

FROM base AS dev_install
RUN mkdir -p /temp/dev
COPY ./frontend/package.json ./frontend/yarn.lock /temp/dev/
RUN cd /temp/dev && yarn install

FROM base AS dev_platform
COPY --from=dev_install /temp/dev/node_modules /app/node_modules/
COPY ./frontend ./

EXPOSE 5173/tcp
CMD [ "yarn", "dev", "--host", "0.0.0.0" ]
