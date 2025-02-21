# simple mflux api

bare-bone web API to the [mflux library](https://github.com/filipstrand/mflux).

mflux lib is not really updated on the pip repositories, so it's manually linked to the github repository in `pyproject.toml`.

this is just a simple API wrapper, for a full breakdown of mflux capabilities, check out the [mflux github](https://github.com/filipstrand/mflux).

## install and run

first install the dependencies using [uv](https://github.com/astral-sh/uv?tab=readme-ov-file#installation)

```
uv sync
```

then run the api

```
uv run python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

> [!NOTE]  
> the first time you run mflux, it will download the model weights and cache them. that's around 30gb of space, so it might take a while.

## web frontend

in the `/client` folder, there is a simple frontend to test the api.
just run a web server in the client folder to test it out.

## next steps

- [x] add a simple web frontend to test the api
- [ ] add a queue system (_you are position #4 in the queue_) etc.
- [ ] add error handling
