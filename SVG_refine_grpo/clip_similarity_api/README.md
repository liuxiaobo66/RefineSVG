# CLIP 图像相似度 API（单卡高并发）

位置：`${PROJECT_DIR}/clip_similarity_api`

## 1) 启动服务

```bash
bash ${PROJECT_DIR}/clip_similarity_api/start_clip_similarity_api.sh
```

默认参数：
- 端口：`18080`
- 模型：`${CLIP_MODEL_PATH}`
- 单卡：`CUDA_VISIBLE_DEVICES=0`
- 微批：`MAX_BATCH_SIZE=64`, `BATCH_TIMEOUT_MS=8`

可覆盖示例：

```bash
CUDA_VISIBLE_DEVICES=0 PORT=18081 MAX_BATCH_SIZE=128 BATCH_TIMEOUT_MS=5 \
bash ${PROJECT_DIR}/clip_similarity_api/start_clip_similarity_api.sh
```

停止服务：

```bash
bash ${PROJECT_DIR}/clip_similarity_api/stop_clip_similarity_api.sh
```

## 2) 健康检查

```bash
curl -s http://127.0.0.1:18080/health | python -m json.tool
```

## 3) 单次调用示例

```bash
python3 \
  ${PROJECT_DIR}/clip_similarity_api/example_call.py
```

## 4) 用 source/target 目录做 smoke test

```bash
python3 \
  ${PROJECT_DIR}/clip_similarity_api/smoke_test_pairs.py \
  --limit 32
```

## 5) 并发压测

```bash
python3 \
  ${PROJECT_DIR}/clip_similarity_api/benchmark_concurrency.py \
  --requests 1000 --concurrency 64 --pairs-limit 1024 --pairs-per-request 1
```

如果你在 RL 里能一次传多对样本，推荐把 `--pairs-per-request` 调大（例如 4/8），
可显著提升 pair/s 吞吐。

## 6) API 协议

### POST `/v1/similarity`

请求体：

```json
{
  "pairs": [
    {
      "image_a_path": "/path/to/a.png",
      "image_b_path": "/path/to/b.png"
    }
  ],
  "normalize_to_01": false
}
```

也支持 base64：

```json
{
  "pairs": [
    {
      "image_a_b64": "...",
      "image_b_b64": "..."
    }
  ]
}
```

返回：

```json
{
  "results": [
    {"similarity": 0.98, "error": null}
  ],
  "model_path": "...",
  "device": "cuda:0",
  "queue_size": 0,
  "batch_size_last": 12,
  "request_time_ms": 14.5
}
```

> similarity 是 cosine 值，范围 `[-1, 1]`；`normalize_to_01=true` 时映射到 `[0,1]`。
