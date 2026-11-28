# VisuReach for DeepSeek Harness

[![Stable 0.6.0](https://img.shields.io/badge/stable-v0.6.0-2b7a4b)](https://github.com/duahciM/dsh-visureach/releases/tag/v0.6.0)
[![Research Stage 8](https://img.shields.io/badge/research-stage%208-5865F2)](https://github.com/duahciM/dsh-visureach/releases/tag/modules-stage8)
[![Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

VisuReach 是 DeepSeek Harness（DSH）的离线视频观察插件。它让图像模型先定位值得关注的片段，再按需读取原图、比较局部证据并核验观察结论。适合查找短事件、比较前后状态和记录可复查的视觉依据。主分支存放 **0.6.0** 稳定插件。阶段八源码和 AMR 研究配置见预发布与 [Archive 的 DSH 插件专区](https://github.com/duahciM/EvidenWeft-Archive/tree/main/dsh-plugin)。

## 核心能力

- 从任意本地视频建立证据目录，保留 D1–D6 的独立提名、图 1／图 2、变化线索与未观察区间。
- 通过真实源 PTS 读取原尺寸帧，也可指定片段重采样。目录和波形帮助定位，原视频访问可独立进行。
- 关联原图、裁切父链、局部比较、对象候选、状态链和观察修订。几何结果及身份假设分别保留证据范围。
- 可选接入 OCR、ASR 和语义检索。文字、计算和原图分别标注来源。AMR 局部细化在独立研究配置中使用。

## 项目截图

<!-- 在这里添加 DSH 界面、视频时间轴或原图证据查看截图。 -->

## 快速开始

准备 DSH 和图像输入能力的模型路线。建议使用 Node.js 24（包含 `node:sqlite`）或已验证的 Electron 宿主、Python 3.12，以及同一发行包中的 FFmpeg 和 FFprobe。

~~~powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install numpy opencv-python-headless Pillow
ffmpeg -version
ffprobe -version
~~~

1. 从 [v0.6.0 Release](https://github.com/duahciM/dsh-visureach/releases/tag/v0.6.0) 下载 `local-dsh-video-observer-0.6.0.tgz` 并解压到新目录。压缩包里的 `package/` 即插件根目录。
2. 在 DSH 的 **Plugins → Add plugin** 中选该目录，或用宿主的 `plugin_manager` 执行 `install_bundle`。手动配置时使用插件目录的 `index.js` 和 [Cordis 配置模板](video_observer_offline060/cordis.patch.yml)。
3. 若运行时没有自动找到依赖，在插件配置中填写 `pythonPath` 和 `ffmpegPath`，然后重新加载插件。

### 使用示例

可以告诉 Agent “打开这段视频，找出面板首次亮起的证据。对照发生前后原图，并给出可回查的引用”。工具的实际参数可按当前会话结果填写。

~~~javascript
video_open({path: "D:/videos/panel.mp4"})
observation_goal({goal: "核验面板亮起前后的状态变化"})
catalog_search({start: 0, end: 5})
~~~

`catalog_search` 的范围用秒。原图交付的 `pts` 要使用本次索引返回的真实整数 PTS。工具示例和证据解释见 [0.6.0 使用指南](video_observer_offline060/docs/V060.md)，也可通过 `observer_guide` 按需查看。

### 独立 MCP

需要在 DSH 之外使用时，stdio 入口可绑定一个或多个明确授权的本地视频目录。以下命令从源码仓库根运行，`config.json` 可按下一节填写。

~~~powershell
node video_observer_offline060/mcp_stdio.js config.json D:/videos
~~~

## 配置

| 项目 | 用途 |
|---|---|
| `pythonPath` · `ffmpegPath` | 可执行文件的绝对路径。配置 FFmpeg 时，FFprobe 应在同一目录。 |
| `cacheDir` | 索引与观察缓存目录。迁移或对照版本时可显式指定独立目录。 |
| `sampleFps` · `frameLimit` | 基础扫描默认 5 FPS，`frameLimit: 0` 表示基础扫描无配置帧数上限。原片回查有独立采样参数。 |
| `integration` | 设为 `false` 可关闭扩展工具组。 |
| `semanticAssets` · `semanticPythonPaths` | 可选语义模型资产目录和额外 Python 模块路径。 |
| `tesseractPath` · `whisperModel` | 可选 OCR 可执行文件及本地语音模型。ASR 还需要 FFmpeg 的 whisper 滤镜。 |

基本离线功能使用以上插件配置。可选能力的依赖、开关与完整数据契约见 [INTEGRATION.md](video_observer_offline060/docs/INTEGRATION.md)。

~~~json
{
  "pythonPath": "D:/visureach/.venv/Scripts/python.exe",
  "ffmpegPath": "D:/ffmpeg/bin/ffmpeg.exe",
  "cacheDir": "D:/visureach/cache",
  "sampleFps": 5
}
~~~

## 架构与目录

~~~mermaid
flowchart LR
    H["DSH / MCP 客户端"] --> T["Node.js 工具和会话"]
    V["本地视频"] --> P["Python / FFmpeg 索引和比较"]
    T --> P
    P --> C["候选目录、图和波形"]
    V --> O["真实 PTS 原图与裁切"]
    T --> O
    C --> R["证据查询与可修订台账"]
    O --> R
    R --> H
~~~

Node.js 负责工具接口、会话和证据收据，Python 与 FFmpeg 负责索引、回源和局部测量。原图、派生图、计算和文字各自保留来源，使用者可以按观察目的选择查看范围和精度。

| 路径 | 内容 |
|---|---|
| [video_observer_offline060/](video_observer_offline060/) | 稳定离线插件源码、配置模板和指南 |
| [python/](video_observer_offline060/python/) · [vendor/](video_observer_offline060/vendor/) | 视频索引、解码和随插件交付的算法 |

## 测试、引用与参考

安装、原图交付、历史回归及研究复跑的检查方法见 [Archive 测试与复现](https://github.com/duahciM/EvidenWeft-Archive#测试与复现)。引用插件时注明所用 [Release](https://github.com/duahciM/dsh-visureach/releases) 及其证据归档，源码采用 [Apache 2.0](LICENSE)。

[架构和证据契约](video_observer_offline060/docs/DETAIL.md) · [算法仓库](https://github.com/duahciM/EvidenWeft) · [实验档案](https://github.com/duahciM/EvidenWeft-Archive)
