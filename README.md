# CS336 Spring 2025 Assignment 1

## 这是Standford CS336 作业一的仓库
内容有
-手写tokenizer训练流程和Tokenizer类（继承nn.Module），训练一个vocabulary=10000的tokenizer，并使用
-手写Transformer大部分组件，遵循torch的格式。
-训练Transformer ,当然用的是小的数据集
## 使用uv来管理虚拟环境
略 -类似python虚拟环境
### 下载数据
略 -就是大量txt，使用TinyStoriesV2-GPT4（2GB），owt12GB太大了

使用```source .venv/bin/activate```来选择虚拟环境.  

使用```uv add ipykernel```
在jupter中使用，

使用
```
uv run pytest tests/test_train_bpe.py
```
运行测试代码，全部pass说明组件符合要求。
![alt text](note/image-1.png)