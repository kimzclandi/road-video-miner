# 道路序列片段选择与冗余分析

[![CI](https://github.com/kimzclandi/road-video-miner/actions/workflows/ci.yml/badge.svg)](https://github.com/kimzclandi/road-video-miner/actions/workflows/ci.yml)

在 KITTI 的已解码 PNG 序列中，固定选取两秒片段时，时序信息是否有助于覆盖更多检测漏检轨迹？本项目比较随机、均匀时间、外观去重和时序覆盖选段，再用固定检测器与参考标签评价覆盖和冗余。

实际处理 20 个序列、120 个不重叠的一秒片段，完成 600 帧 CPU 检测。外观特征为 RGB 直方图，时序特征为相邻直方图变化；选择采用贪心覆盖。它不是完整视频编解码或学习型时序理解系统。

## 当前结果

每序列选择 2 片段、2 秒；下表为 12 个评测序列、五个预设配对种子的描述均值。

| 方法 | 参考漏检轨迹覆盖 | 总轨迹覆盖 | 重复轨迹比例↓ |
|---|---:|---:|---:|
| 随机 | 44.40% | 51.97% | 16.03% |
| 均匀时间 | 56.44% | 60.19% | 15.10% |
| 外观最远点去重 | 46.52% | 51.08% | 12.48% |
| 仅外观覆盖消融 | 53.16% | 58.27% | 14.09% |
| 时序覆盖 | 51.35% | 51.46% | 15.40% |

时序相对随机 **+6.95 个百分点**，序列 bootstrap 区间 **[−7.32, +20.89]**，未证明稳定优势；均匀时间的点估计更高。外观冗余降低、失败覆盖增加和检测精度提高是不同问题，**本项目没有下游训练或训练收益证据**。

[实验报告](docs/REPORT.md) · [原始结果](reports/result.json) · [协议与复现](docs/PROTOCOL_AND_REPRODUCE.md)

## 查看与运行

```bash
uv sync --locked --python 3.12
uv run python portable_replay.py
uv run streamlit run dashboard.py --server.address 127.0.0.1
```

无需下载原始数据即可浏览已保存结果。回放检查固定预算、原选择的贪心条件和指标；不是重新推理，也不保证跨平台生成位级相同的选择名单。完整图像/权重下载与原生复算见[运行说明](docs/PROTOCOL_AND_REPRODUCE.md)。

## 实现与贡献范围

- [片段特征与选择](src/miner.py)：帧—片段—序列关系、时间预算、直方图特征与参考轨迹评测。
- [数据准备](scripts/prepare.py)和[实验执行](scripts/experiment.py)：固定清单、下载校验、错误记录与中断恢复。
- 使用上游预训练检测器与 KITTI 参考标签；未训练视频表征或目标跟踪模型。代码、测试与文档使用 AI 辅助开发。

[实现讲解与练习](docs/INTERVIEW.md)为可选附件。

## 主要限制

序列隔离不证明路线或采集日隔离；5 Hz 观测可能遗漏快速事件。参考指标是自定义 Car/Pedestrian 漏检轨迹覆盖，不是 KITTI 官方 AP/HOTA，也不是独立人工裁决的真实驾驶失败。

跨平台浮点近同分可能改变重生成名单；公开回放对历史贪心选择采用 1e-12 容差并重算指标，保留最初 Linux 严格复算失败记录和历史实现。

原始图像、标签与模型权重不进 Git。代码 [MIT](LICENSE)；KITTI 数据遵循非商业教育研究及 CC BY-NC-SA 3.0 要求，见[数据归属](DATA_LICENSE.md)。
