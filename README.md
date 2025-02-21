# simple mflux api

It uses the majestic mflux library to generate images. The problem is that mflux is not really updated on the pip repositories, so I had to manually install it.

# run

```
uv run python -m uvicorn main:app --host 0.0.0.0 --port 8000
```
