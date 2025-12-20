#!/bin/bash

# 安装依赖
pip3 install --no-deps -e .
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-deps "tensordict>=0.8.0,<=0.10.0,!=0.9.0"
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pyvers
python -m pip install -r requirements.txt

# 从 .env 文件读取 WANDB_KEY 并登录 wandb
if [ -f .env ]; then
    WANDB_KEY=$(grep -E "^WANDB_KEY=" .env | cut -d '=' -f2- | tr -d '"' | tr -d "'" | xargs)
    
    if [ -n "$WANDB_KEY" ] && [ "$WANDB_KEY" != "your_wandb_api_key_here" ]; then
        echo "正在使用 .env 中的 WANDB_KEY 登录 wandb..."
        export WANDB_API_KEY="$WANDB_KEY"
        wandb login "$WANDB_KEY" --relogin
        if [ $? -eq 0 ]; then
            echo "✓ Wandb 登录成功"
        else
            echo "✗ Wandb 登录失败，请检查 WANDB_KEY 是否正确"
        fi
    else
        echo "⚠ .env 文件中未找到有效的 WANDB_KEY，跳过 wandb 登录"
    fi
else
    echo "⚠ .env 文件不存在，跳过 wandb 登录"
    echo "提示: 创建 .env 文件并添加 WANDB_KEY=your_api_key 以启用自动登录"
fi


# new_password="12345"
# echo "root:$new_password" | chpasswd
# sleep infinity