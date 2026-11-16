# agent-memory-map

面向 [Hermes Agent](https://github.com/NousResearch/hermes-agent) 类常驻 Agent 的**记忆地图体系**：把长期记忆从「平铺指针清单」重组为「总入口地图 → 分类枢纽 → 条目」的导航结构，配套 FTS+语义双通道召回、续聊 recap、可达性体检哨兵与两级容量治理。

## 为什么需要它

平铺指针式 L0 记忆有三个结构性病根（本项目开发过程中逐一实测）：

1. **满员休眠**：L0 容量有限，新条目挤占存量，指针被挤出后「想不到去查」；
2. **存了 ≠ 找得回**：条目下沉后 INDEX 漏收、用词漂移，检索链路静默断链；
3. **模糊检索失效**：FTS 全词 AND + 短词盲区，自然语言问法实测 0/20 命中。

本项目的解法：地图导航解决 ①（分类树恒定，条目数与 L0 尺寸解耦）；可达性体检解决 ②（每周哨兵+探针题库）；FTS+语义双通道+通道强提名解决 ③（基准从 0/20 → 20/20，见 docs/BENCHMARK.md）。

## 架构

```
L0 总入口（恒定 ~900 字符）
  = ⚡热区（每轮必须的输出偏好/红线指针）
  + 🗺 地图（8 条分类行 → 分类枢纽文件）
大分类 = 分类枢纽文件（如 channels.md）
小分类 = 枢纽文件内章节（过大再拆子文件）
确切点 = 具体条目/坑/参数

辅助层：
  INDEX.md     关键词平铺索引（检索加速，与地图互补）
  probes.json  探针题库（可达性检测，换说法覆盖）
  体检哨兵     地图一致性 + 探针可达 + 容量分级
  语义召回     state.db 会话库上的 FTS+BM25 RRF 双通道
  续聊 recap   召回命中 → 会话末段再水化（凭据脱敏）
```

容量治理为**两级触发**：env-reminder 便签卡控（≥95% 注入执行令，当轮执行地图治理：合并归组/降级休眠/淘汰）为主触发；体检容量分级（75%/90%）为周度预警。

## 组件

| 脚本 | 作用 |
|------|------|
| `scripts/session_recall.py` | 会话语义召回：双面语料 dense + 双面 BM25 RRF 融合，父子会话同根归并，时间词过滤，通道强提名保位 |
| `scripts/session_resume.py` | 续聊 recap：召回定位 → 防注入头 + 末段原文 + 未决事项，三层凭据脱敏（脱敏先于截断） |
| `scripts/memory_reach_audit.py` | 可达性体检哨兵：A 结构（悬空引用/孤儿文件）+ B 探针抽检 + C 地图一致性 + D 容量分级 |
| `scripts/semantic_recall_pilot.py` | 构建索引与基准评估器（build3 = 生产语料） |
| `scripts/test_semantic_units.py` | 单元测试（时间词快照/脱敏回归/强提名断言） |

## 快速开始

```bash
# 依赖环境（建议独立 venv）
pip install numpy fastembed   # 嵌入模型 bge-small-zh-v1.5 约 24MB，自动下载

# 建索引（语料 = 你的会话库；路径经 SEM_DB/SEM_IDX/EMB_OUT2 环境变量配置）
python3 scripts/semantic_recall_pilot.py build3

# 语义召回
python3 scripts/session_recall.py "上次聊过的某个话题" 5

# 续聊 recap
python3 scripts/session_resume.py --resume "<会话id|搜索词>"

# 可达性体检
python3 scripts/memory_reach_audit.py
```

默认读取路径见各脚本头部常量，均可经环境变量覆盖（`SEM_DB`/`SEM_IDX`/`EMB_OUT2`/`EMB_THREADS`/`EMB_CACHE`）。

## 基准

在 20 题模糊问法基准（换说法/缩写/跨句，避免与语料字面量重合）上的迭代：

| 引擎 | 命中 |
|------|------|
| FTS 关键词（基线） | 0/20 |
| 语义 dense（摘要级） | 8/20 |
| 语义 dense（消息级） | 8/20 |
| + 双面词法 RRF | 11/20 |
| + 父子同根计分 + 别名 + 通道强提名 | **20/20** |

关键设计教训见 docs/BENCHMARK.md。

## 测试

```bash
python3 scripts/test_semantic_units.py   # 23 项单测
```

## License

MIT
