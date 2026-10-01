# 运行资源准备说明

本仓库只包含源码。下列目录均被 `.gitignore` 排除，应该只保存在本机。

## 1. OpenVINO 部署包

默认路径为：

```text
deployment/openvino_cpu_2class/
├─ ACTIVE_MODEL.txt
├─ manifest.json
├─ yolo26_openvino/
│  └─ __init__.py
└─ int8_v2/
   └─ models/
      └─ mixed_head_fp/
         ├─ model.xml
         └─ model.bin
```

部署包必须满足以下契约：

- 类别严格为 `{0: "monster", 1: "player"}`；
- 活动版本为 `int8_v2`；
- 活动模型路径为 `int8_v2/models/mixed_head_fp/model.xml`；
- 外部输入为 `uint8`、BGR、NHWC、`[1, 224, 1280, 3]`；
- 输出为 `[1, 300, 6]`，模型内部已执行 NMS；
- `manifest.json` 中的 XML/BIN SHA-256 必须与文件一致；
- `yolo26_openvino.OpenVINOYolo26` 必须接受部署根目录、`confidence`、`max_det` 和 `variant` 参数。

仓库不提供模型下载地址。请只使用你有权使用和分发的模型与运行时。

## 2. 名字定位样本

主界面的名字标定页面也可创建和编辑样本；保存后，配置中的 `nametag.name` 必须对应同一个配置名称。名字模板匹配结果加上标定的 `player_offset` 得到人物脚点。

目录格式：

```text
nametag/classic_cn_player/
├─ profile.yaml
├─ sample_001.png
└─ sample_002.png
```

`profile.yaml` 示例：

```yaml
settings:
  max_score: 0.30
  local_search_radius: 140
  global_refresh_frames: 30
  max_jump: 250
  jump_confirm_frames: 2
  jump_confirm_radius: 25
  max_missed_frames: 5
  edge_weight: 0.65
samples:
  - file: sample_001.png
    player_offset: [40, 75]
    enabled: true
```

建议使用工具生成，不要手写人物偏移：

```powershell
python -m tools.nametagCalibrator --profile classic_cn_player --cfg classic_cn
```

名字截图可能包含个人角色信息，请勿提交到公开仓库。

## 3. 小地图与路线

目录格式：

```text
minimaps/your_map_id/
├─ map.png
├─ route1.png
├─ route2.png
└─ route_rest.png
```

- `map.png` 是完整的小地图底图；
- `route*.png` 使用 `config/config_default.yaml` 中 `route.color_code` 和 `route.color_code_up_down` 定义的像素颜色；
- 普通模式至少需要一个非 `route_rest.png` 的路线文件；
- 新版路线编辑器同时保存 `route*.json` 语义路线和同名 PNG 预览；不应只复制其中一个文件；
- 地图目录名必须与 `bot.map` 完全一致。

创建命令：

```powershell
python -m tools.routeRecorder --new_map your_map_id --cfg classic_cn
```

## 4. UI 识别模板

旧版引擎启动时会读取：

```text
misc/login_button_cn.png
```

请从自己的客户端截图中紧密裁剪登录按钮，保持与程序捕获画面相同的缩放比例。英文客户端对应 `misc/login_button_eng.png`。

其他旧功能如传统符文、数字或怪物模板可能引用 `rune/`、`numbers/`、`monster/`，但本快照的当前 YOLO 主路径不提供这些素材。只有确认自己需要对应旧工具时才创建。

## 5. 不要提交的内容

- 模型 XML/BIN、数据集和第三方运行时副本；
- 人物名字样本和游戏截图；
- 地图、路线、录像、日志和临时配置；
- API 密钥、访问令牌、证书和私人服务器地址。
