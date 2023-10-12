import argparse
import httpx
import os
import subprocess
import logging
from starlette import applications, responses, exceptions
from starlette.requests import Request

app = applications.Starlette()
local_server_process = None
logging.basicConfig(level=logging.DEBUG)


def start_local_server(model_filename):
    global local_server_process
    if local_server_process:
        local_server_process.terminate()
        local_server_process.wait()
    cmd = ["python3", "-m", "llama_cpp.server", "--model", model_filename,
           "--n_gpu_layers", "1", "--n_ctx", "4096"]  # TODO: set this more correctly
    logging.debug('Running: %s' % ' '.join(cmd))
    local_server_process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


@app.route('/{path:path}', methods=['GET', 'POST', 'PUT', 'DELETE'])
async def proxy(request: Request):
    path = request.url.path
    url = f"http://localhost:8000{path}"

    data = await request.body()
    headers = dict(request.headers)
    r = None
    async with httpx.AsyncClient() as client:
        try:
            if request.method == 'GET':
                r = await client.get(url, params=request.query_params, headers=headers)
            elif request.method == 'POST':
                r = await client.post(url, data=data, headers=headers, timeout=30)
            elif request.method == 'PUT':
                r = await client.put(url, data=data, headers=headers)
            elif request.method == 'DELETE':
                r = await client.delete(url, headers=headers)
        except httpx.RemoteProtocolError as exc:
            logging.debug(f'Connection closed prematurely: {exc}')
    content = r.content if r else ''
    status_code = r.status_code if r else 204
    headers = dict(r.headers) if r else dict()
    return responses.Response(content=content, status_code=status_code, headers=headers)


@app.exception_handler(404)
async def not_found(request, exc):
    return responses.JSONResponse({"error": "Not found"}, status_code=404)


@app.exception_handler(500)
async def server_error(request, exc):
    return responses.JSONResponse({"error": "Server error"}, status_code=500)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model')
    args = parser.parse_args()
    
    start_local_server(args.model)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)
