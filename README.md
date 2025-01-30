# 基于 OminiControl 框架改造

```
./src/train/train_local.py 改成加载本地数据集
./src/data.py 把只能接受RGB输入改成可以接受RGBA
./src/callbacks.py 改成接受4张图片，每两个分别作为condition推理一次
./src/train/model.py 改成加载本地的Flux.1 dev
./src/flux/condition.py 改成根据一次推理接受到的图片条件数，分别分配y方向的position_delta
./src/flux/generate.py 调用condiition类的encode方法时会额外传入一个condition_index，该condition的y轴偏移量是-(condition_index + 1) * self.condition.size[0] // 16
subject_512.yaml 改成加载本地的Flux.1 dev
```
