We have already created an initial project for you to participate in the ITU competition and written the corresponding `.gitlab-ci.yml` file. When you submit code to the `main` branch or create a new pipeline, a training task will be triggered. You can then go to the web interface to check your project status and runtime logs. Please note that you can only submit **3 times per day**; submissions beyond that will not be accepted.

OOur GitLab instance supports logging in with Zero2x platform accounts. However, please note that these accounts cannot be used to push code via Git or pull Docker images directly.

To perform these actions, you must create a personal access token to use as your password. Please navigate to **Avatar > Edit Profile > Access Tokens** to generate your token. For detailed instructions, please refer to the official Jihu GitLab documentation.

Please note the following in `.gitlab-ci.yml`: dataset path, model output address, and launch name can be modified as needed. The paths in your code must be consistent with those in `.gitlab-ci.yml`. Only modify the container mount paths — do **not** change the actual paths to avoid conflicts with other participants.

We also have specific requirements for the naming of your output files. Please follow them strictly, as failure to do so will affect the final scoring of your submitted model.

**Track 1 output:**  
`/your_container_output_path/result.json`

**Track 2 output:**  
`/your_container_output_path/turbidity_result.json`  
`/your_container_output_path/chla_result.json`

**Track 3 output:**  
`/your_container_output_path/result.json`

The base images used for the competition are specified in the `Dockerfile`. If you need to use the base image for local debugging, you can search for the project `itu_docker_images`. The container registry of that project contains the base images used for the competition. You can also pull them with Docker.

```
sudo vim /etc/docker/daemon.json

{
  "insecure-registries": ["gitlab-itu.zero2x.org:5050"]
}

systemctl daemon-reload
systemctl restart docker

docker login http://gitlab-itu.zero2x.org:5050

docker pull gitlab-itu.zero2x.org:5050/itu_images/itu_docker_images:ubuntu22.04-py310.19
docker pull gitlab-itu.zero2x.org:5050/itu_images/itu_docker_images:ubuntu22.04-cuda12.3.2-cudnn9-py310.19
docker pull gitlab-itu.zero2x.org:5050/itu_images/itu_docker_images/competition-base:pytorch2.5.1-cuda12.1-cudnn9
```
