# CLIP Similarity API 测试报告

测试时间：2026-03-21 (UTC)

## 环境
- Python: `python3`
- GPU: `CUDA_VISIBLE_DEVICES=0`（H20）
- 模型路径: `${CLIP_MODEL_PATH}`

## 启动命令

```bash
bash ${PROJECT_DIR}/clip_similarity_api/start_clip_similarity_api.sh
```

健康检查：
- `GET /health` 返回 `ok=true`，模型加载成功。

## 功能验证

### 1) 单样本调用（路径 + base64）
脚本：`example_call.py`

- 路径输入：HTTP 200，返回 similarity=0.9345703125
- base64 输入：HTTP 200，返回 similarity=0.9345703125

### 2) 批量调用 smoke test（source/target 同名配对）
脚本：`smoke_test_pairs.py --limit 32`

- HTTP 200
- `pairs=32 success=32 fail=0`
- 示例相似度：0.9346, 0.9238, 0.8638, 0.9941 ...

## 并发压测

数据目录：
- source: `/path/to/test/source_images`
- target: `/path/to/test/target_images`

脚本：`benchmark_concurrency.py`

### 场景A（1 pair / request）
```bash
--requests 1000 --concurrency 128 --pairs-per-request 1
```
结果：
- Success: 1000 / 1000
- Throughput: **90.40 req/s**（≈90.40 pair/s）
- Latency: p50=1275ms, p90=1862ms, p99=2307ms

### 场景B（8 pairs / request）
```bash
--requests 500 --concurrency 64 --pairs-per-request 8
```
结果：
- Success: 500 / 500
- Throughput: **20.01 req/s**（**160.06 pair/s**）
- Latency: p50=3111ms, p90=3204ms, p99=3571ms

## 结论
- API 已可稳定部署并调用，支持高并发请求。
- 在当前配置下：
  - 单对请求场景约 90 pair/s；
  - 多对合并请求可到约 160 pair/s。
- 服务端微批处理生效（batch size 会随并发自动增长）。
