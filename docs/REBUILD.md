# 重建清单（新机部署）

## 1. 依赖环境
```bash
python3 -m venv /opt/data/venvs/emb
pip install numpy fastembed   # 嵌入模型自动下载（国内可设 HF_ENDPOINT=https://hf-mirror.com）
```
模型缓存钉持久目录：`EMB_CACHE=/opt/data/.cache/fastembed`（默认落 TMPDIR 会被定期清理）。

## 2. 路径配置（环境变量，全部有默认值）
| 变量 | 默认 | 说明 |
|------|------|------|
| SEM_DB | `<HERMES_HOME>/state.db` | 会话库（召回语料源） |
| SEM_IDX | `/opt/data/state/semantic_recall_v3_dual.npz` | 语义索引 |
| EMB_OUT2 | 同上 | 构建输出 |
| EMB_THREADS | 2 | onnxruntime 线程钉死（防吃满核） |
| EMB_CACHE | /opt/data/.cache/fastembed | 模型缓存持久目录 |

## 3. 建索引
```bash
python3 scripts/semantic_recall_pilot.py build3
```

## 4. 定时任务（建议错峰）
- 语义索引重建：每日（低峰）——全量重建幂等，量小无需增量
- 可达性体检：每周——异常时告警（脚本退出码 1）

## 5. 探针题库
复制 examples/probes.example.json 为正式题库，按「真实问法+换说法别名+预期承载」持续补充。

## 6. 容量治理触发器
外部监视器（如 Cron、systemd timer）盯 L0 文件尺寸，≥阈值触发治理流程（合并归组/降级休眠/淘汰）。阈值建议 75% 预警 / 90% 强制。
