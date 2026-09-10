# Road Video Miner · 视频场景挖掘与时序去重

**真实序列、相同时间预算、参考失败评测；没有下游训练。**

20个KITTI序列 → 120个不重叠一秒片段 → 600帧CPU检测 → 无标签片段选择 → 逐序列参考轨迹评测。

时序覆盖方法相对随机的参考漏检轨迹覆盖差为 **+6.95个百分点**，12序列bootstrap区间 **[−7.32,+20.89]**。**不支持稳定优势**；均匀时间与仅外观覆盖的点估计都更高。保留全部负证据，不为追正结果改规则。

[中文实验报告](docs/REPORT.md) · [冻结设计与复现](docs/PROTOCOL_AND_REPRODUCE.md) · [岗位/面试](docs/INTERVIEW.md) · [原始结果](reports/result.json) · [本地验收](reports/qa.json)

## 实际结果

每序列选择2片段、2秒；下表为12个评测序列、五个预设配对seed上的描述均值。

| 方法 | 参考漏检轨迹覆盖 | 总轨迹覆盖 | 重复轨迹比例↓ |
|---|---:|---:|---:|
| 随机 | 44.40% | 51.97% | 16.03% |
| 均匀时间 | 56.44% | 60.19% | 15.10% |
| 外观最远点去重 | 46.52% | 51.08% | 12.48% |
| 仅外观覆盖消融 | 53.16% | 58.27% | 14.09% |
| 时序覆盖 | 51.35% | 51.46% | 15.40% |

更低外观冗余、更多失败命中、更高模型精度是三个不同命题。当前没有训练实验，不能声称检测精度或标注效率提升。

## 运行

```bash
uv sync --locked --python 3.12
uv run pytest -q
uv run python portable_replay.py
uv run streamlit run dashboard.py --server.address 127.0.0.1
```

无需数据下载即可查看公开结果并重算选择及预算。独立参考标签重算、数据/权重获取和真实推理见[复现说明](docs/PROTOCOL_AND_REPRODUCE.md)。CPU600帧累计约44.7秒，另有下载、初始化与校验成本。独立虚拟环境与依赖锁，无付费API。

## 核心实现

- `src/miner.py`：时间/血缘契约，时序直方图特征，facility-location，参考匹配与覆盖评价。
- `scripts/prepare.py`、`remote_zip.py`：固定名单，HTTP范围读取、ETag/CRC/SHA256校验，缺帧/解码/重复检查。
- `scripts/experiment.py`：冻结身份、逐帧不可覆盖证据、错误保留、恢复、选择先于参考评价。
- `reports/protocol.json`：模型、源码、预处理、数据、种子、主指标和统计单位。

## 边界

KITTI非商业教育研究、CC BY-NC-SA 3.0；[归属及数据许可](DATA_LICENSE.md)。代码MIT不覆盖数据。数据为PNG序列，不是MP4编解码基准。原始图像/标注/权重不进Git。序列隔离不证明路线/采集日隔离；5Hz观测不能覆盖所有快速事件。参考端点为自定义Car/Pedestrian漏检轨迹覆盖，不是KITTI官方AP/HOTA。实现由AI辅助，作者必须亲自理解并通过面试中的解释和修改验证。
