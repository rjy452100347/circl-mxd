# 圈圈冒险岛助手

这是一个基于屏幕捕获、计算机视觉和模拟键盘输入的 Windows 桌面辅助程序。

> 使用自动化程序可能违反游戏或平台规则，并可能导致账号受限或封禁。请只在获得明确许可的测试环境中使用。项目不读取游戏内存，不提供反作弊绕过，也不保证适用于任何具体服务器或客户端版本。
## 当前版本展示（仅测试战士，远程职业效果更好）
![主界面演示](docs/output.gif)

![运行演示](docs/output2.gif)

![路线演示](docs/output3.gif)
## 这个版本包含什么

- 简体中文主界面和高级设置；
- `normal`、`aux`、`patrol` 三种运行模式；
- 基于名字样本的人物画面定位；
- OpenVINO 双类别 `monster/player` YOLO 检测；
- 旧版小地图路线执行、攻击、血蓝监控和换线逻辑；
- 名字样本标定器和路线录制器。

## 环境要求

- Windows 10/11；
- Python 3.12；
- 窗口模式运行的目标程序；
- 使用键盘控制时，本程序与目标程序必须处于相同权限级别。如果目标程序以管理员身份运行，本程序也必须从管理员 PowerShell 启动。


## 安装

在项目根目录打开 PowerShell：

```powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如果要运行测试，再安装 pytest：

```powershell
python -m pip install pytest
```

## 首次运行前必须准备的资源

仓库有意排除了所有运行素材。至少需要自行准备：

1. OpenVINO 部署包；
2. 当前角色的名字定位样本；
3. 普通路线模式使用的小地图底图和路线图；
4. 当前客户端的登录按钮模板 `misc/login_button_cn.png`。

完整目录结构、模型契约和制作方法见 [资源准备说明](docs/RESOURCE_SETUP.md)。未准备资源时可以打开主界面，但点击开始或按 F1 会显示缺失资源错误并拒绝启动。

## 配置

默认配置位于 `config/config_default.yaml`，当前客户端示例差异配置位于 `config/config_classic_cn.yaml`。


## 启动主程序

```powershell
.\.venv\Scripts\Activate.ps1
python -m src.main
```

操作方式：

- 在主界面加载配置并确认地图、攻击键和药水键；
- 点击“开始”或按 `F1` 启动；
- 再按 `F1` 暂停并释放控制键；
- `Ctrl+Shift+F12` 请求紧急关闭；
- “窗口监控”和“路线监控”用于观察识别与路线状态。

## 制作名字样本

先打开目标程序并显示角色，然后运行：

```powershell
python -m tools.nametagCalibrator --profile classic_cn_player --cfg classic_cn
```

按工具提示冻结画面、框选唯一名字文字并点击人物脚底。输出会写入 `nametag/classic_cn_player/`，该目录默认被 Git 忽略。

## 录制小地图与路线

```powershell
python -m tools.routeRecorder --new_map your_map_id --cfg classic_cn
```
器常用按键：

- `F1`：开始或暂停录制；
- `F2`：保存截图；
- `F3`：标记路线终点并保存路线；
- `F4`：保存小地图底图；
- `Q`：关闭录制器。

输出位于 `minimaps/your_map_id/`。录制完成后把 `bot.map` 设置为同一个 `your_map_id`。

## 常见错误

- `找不到部署清单`：模型目录缺少 `manifest.json` 或配置路径错误；
- `找不到活动模型声明`：缺少 `ACTIVE_MODEL.txt`；
- `缺少 profile.yaml 或 sample_*.png`：需要先做名字标定；
- `Image not found: minimaps/.../map.png`：地图底图尚未录制或地图名不一致；
- `Image not found: misc/login_button_cn.png`：需要从自己的客户端画面裁剪登录按钮模板；
- 能识别但人物不移动：通常是程序权限低于目标程序，请用管理员 PowerShell 启动；
- OpenVINO 版本错误：本快照要求 Windows 环境安装 `openvino==2026.3.0`。

运行日志写入 `log/`，临时界面配置写入项目工作目录；这些内容均被 Git 忽略。

## 隐私与安全

- 游戏画面捕获和模型推理均在本机完成；


## 许可证与来源

代码按 [MIT License](LICENSE) 开源，保留原作者 Ken Yu 的版权声明。外部模型、权重、游戏素材和第三方运行时不属于本仓库授权范围，使用者必须自行确认其许可和分发权。详见 [第三方声明](THIRD_PARTY_NOTICES.md) 与 [快照来源说明](OPEN_SOURCE_PROVENANCE.md)。
