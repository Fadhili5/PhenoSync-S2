我们已经为您创建好参加itu比赛的初始项目，编写好了对应的.gitlab-ci.yml，您提交代码到main分支或是新建流水线将会触发创建训练任务的动作，之后您可到网页端查看您的项目状态和运行日志，请注意每天只可提交3次，超过次数后将不再提交任务。

我们的gitlab支持使用zero2x的平台账号登录，但此账号不支持使用git推送代码或使用docker拉取进行，这需要您创建自己的token作为密码进行操作，请到头像>Edit Profile>Access Tokens创建自己的token，具体操作可查询极狐GitLab官方文档。

您需要注意的是.gitlab-ci.yml中有数据集路径、模型输出地址、启动名称，这些可自行修改，代码中的路径需要与.gitlab-ci.yml中保持一直，请只修改容器挂载路径，实际的路径请勿修改，避免与其他参赛选手冲突。

我们对您的输出结果的文件命名也有要求，请务必遵循，否则会影响到您提交的模型的最终评分
赛道一的输出：

/你的容器输出路径/result.json

赛道二的输出：

/你的容器输出路径/turbidity_result.json

/你的容器输出路径/chla_result.json

赛道三的输出：

/你的容器输出路径/result.json

赛事使用的基础镜像路径编写在了Dockerfile，若您需要使用基础镜像进行本地调试，可搜索*itu_docker_images*项目，此项目的container registry上传了赛事使用的基础镜像，也可使用docker 拉取。

```bash
sudo vim /etc/docker/daemon.json

{
  "insecure-registries": ["gitlab.zero2x.org:5050"]
}

systemctl daemon-reload
systemctl restart docker

docker login http://gitlab.zero2x.org:5050

docker pull gitlab.zero2x.org:5050/itu_images/itu_docker_images:ubuntu22.04-py310.19
docker pull gitlab.zero2x.org:5050/itu_images/itu_docker_images:ubuntu22.04-cuda12.3.2-cudnn9-py310.19
docker pull gitlab-itu.zero2x.org:5050/itu_images/itu_docker_images/competition-base:pytorch2.5.1-cuda12.1-cudnn9
```

