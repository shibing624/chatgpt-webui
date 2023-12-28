#!/bin/bash

# 获取脚本所在目录
script_dir=$(dirname "$(readlink -f "$0")")

# 将工作目录更改为脚本所在目录
cd "$script_dir" || exit

# 检查Git仓库是否有更新
git remote update
pwd

if ! git status -uno | grep 'up to date' > /dev/null; then
	# 如果有更新，关闭当前运行的服务器
	pkill -f main.py

	# 拉取最新更改
	git pull

	# 安装依赖
	pip3 install -r requirements.txt

	# 重新启动服务器
	nohup python3 main.py &
fi

# 检查main.py是否在运行
if ! pgrep -f main.py > /dev/null; then
	# 如果没有运行，启动服务器
	nohup python3 main.py &
fi
