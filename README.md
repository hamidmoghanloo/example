# Tiny model HTTP API

This example downloads `sshleifer/tiny-gpt2` on first startup, caches it under
`/workspace/.cache/huggingface`, and serves it through FastAPI. It uses CUDA
when the assigned node exposes a GPU and falls back to CPU otherwise.

## Endpoints

- `GET /health` reports whether the process is running, the model id, and the device.
- `POST /generate` accepts JSON such as `{"prompt":"A small model can","max_new_tokens":32}`.
- `GET /docs` provides the interactive API reference.

The container listens on `0.0.0.0:8080`. The workload must declare container
port `8080` as an exposed TCP port to receive an endpoint through the platform.

## Run locally on a GPU node

```sh
docker build -t aptar-tiny-model-api .
docker run --rm --gpus all --shm-size 1g -p 35000:8080 \
  -v aptar-model-cache:/workspace/.cache/huggingface aptar-tiny-model-api
```

The first start downloads the model. Then check the health endpoint and generate
a short response:

```sh
curl http://127.0.0.1:35000/health
curl -X POST http://127.0.0.1:35000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"A small model can","max_new_tokens":32}'
```

The named Docker volume keeps the downloaded model between container runs.

## Submit it to Aptar

Push this project to a Git repository that the Aptar build service can read.
On the GPU node, configure the IP clients can reach and allow inbound TCP ports
`35000-65000` in the host firewall and cloud security group:

```sh
sudo aptar node advertise --ip YOUR_NODE_PUBLIC_IP
```

In **Run a new service**, select **Service**, choose **Git repository**, and set
the build context to `platform/examples/tiny-model-api` with `Dockerfile` as the
Dockerfile. In the resource step, enable **Publish endpoint directly on node IP**
and enter container port `8080`. After the workload is running, open its details
and click **Open endpoint**. Aptar assigns an available public port in the range
`35000-65000` and displays the endpoint as `http://NODE_IP:PORT`.

The direct endpoint is public and bypasses the panel's proxy authentication. Use
application-level authentication before serving sensitive data.
