# 三分类模型与运行资源

`config/config_classic_cn.yaml` 使用 `warrior_v3` 的 OpenVINO `int8_head_fp` 三分类权重：`0=monster`、`1=player`、`2=self`。权重和游戏素材不包含在源码仓库中；请自行取得有权使用的部署包，并将 `monster_detect.openvino_deployment_root` 指向它。示例配置使用 `deployment/warrior_v3_openvino_int8`。

部署目录至少需要：

```text
deployment/warrior_v3_openvino_int8/
  ACTIVE_MODEL.txt
  models/int8_head_fp_openvino_model/
    model.xml
    model.bin
    metadata.yaml
    quantization.json
```

`ACTIVE_MODEL.txt` 的内容应为 `models/int8_head_fp_openvino_model/model.xml`。程序核对三个类别、`float32 NCHW [1,3,736,1280]` 输入、`[1,300,6]` 输出及量化记录中的 XML/BIN 哈希；任一项不匹配都会拒绝启动。

三分类示例采用游戏画面裁剪 `[4,0,5,117]` 和 70% 内容缩放，以对齐该模型的训练输入。这些值依赖窗口捕获尺寸；换客户端或训练流程时应重新核对。模型推理覆盖裁剪后的整个游戏区域，先确认唯一 `self`，再从 `monster` 中选择攻击目标。小地图仅用于巡逻和路线坐标；固定平台模式应先校准完整小地图内容区。

可先用不发送按键的检查命令验证模型加载与推理：

```powershell
python -m tools.fullscreen_perception_check --config config/config_classic_cn.yaml --frames 20
```

使用旧双分类模型时，将 `yolo_variant` 改为 `int8_v2` 并提供对应部署包与名字样本。小地图底图、路线图和界面模板由使用者按目标客户端自行准备，分别存放在 `minimaps/<map_id>/` 和 `misc/`；这些目录默认不纳入 Git。
