# Deploying the demo to Hugging Face Spaces (free CPU)

1. Create a new Space at huggingface.co/new-space, choose **Docker** as the SDK and the free CPU hardware.
2. Clone the Space, copy this repo's `app/`, `models/`, `samples/` and `Dockerfile` into it.
3. Spaces needs Git LFS for files over 10 MB, so before committing run:
   `git lfs install && git lfs track "*.onnx" && git add .gitattributes`
4. Put this block at the very top of the Space's README.md:

```
---
title: LoomGuard
emoji: 🧵
colorFrom: indigo
colorTo: yellow
sdk: docker
app_port: 7860
---
```

5. Commit and push. The Space builds the Docker image and serves the dashboard at your Space URL.
   Put that URL on your resume and in the GitHub repo's "About" section.
