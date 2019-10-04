---
layout: post
title: VSCode使用Git
catalories: 工具
tags：Git VSCode插件安装
author:xueyaiii
---

{:toc}

# 从远程克隆代码到本地电脑

- 输入`ctrl + shift + p` 打开命令面板，输入`Git clone`,回车，输入仓库地址
- 安装`Git history`插件，用来查看整个仓库文件提交历史和修改情况

# 将修改的库提交到远程

### 查看修改，进行提交

- `Ctrl + shift + G` 代开代码管理工具,看修改了多少文件
- 在代码管理窗口，点击`+`，将所有文件提交到暂存区
- 打开`···`，选择：提交已暂存的 ( Commit Staged )，输入消息，按`ctrl + enter`提交
- 最后将所有代码push到云端，输入账号密码

### 解决总是输入用户名和密码

- 打开终端，会看到cmd定位在我们仓库位置，我们只要添加：`git config --global credential.helper store`
- 退出vscode并重启，进行git操作

