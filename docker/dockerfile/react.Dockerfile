FROM oven/bun:debian AS base
WORKDIR /app

FROM base AS dev_install
RUN mkdir -p /temp/dev
COPY ./frontend/package.json ./frontend/bun.lock /temp/dev/
RUN cd /temp/dev && bun install --frozen-lockfile

FROM base AS dev_platform
COPY --from=dev_install /temp/dev/node_modules /app/node_modules/
COPY ./frontend ./
RUN bun run build

EXPOSE 5173/tcp
CMD [ "bun", "run", "dev", "--host", "0.0.0.0" ]
