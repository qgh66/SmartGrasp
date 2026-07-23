# SmartGrasp Perception Pipeline

## 配置

### 1. 创建环境（一行搞定）
conda env create -f smartgrasp.full.yml
conda activate smartgrasp

### 2. 安装 SAM2

git clone https://github.com/facebookresearch/sam2.git ~/sam2
cd ~/sam2
mkdir -p checkpoints
wget -O checkpoints/sam2.1_hiera_small.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt

代码自动检测顺序: SAM2_ROOT 环境变量 > ./sam2_repo > ./sam2 > ~/sam2 > ~/Gsam2/Grounded-SAM-2 > ~/Grounded-SAM-2
（把 repo 放在以上任意一个路径即可）

### 3. 设置 API 密钥（如果不用脚本内置的默认值）
export OPENAI_API_KEY=sk-xxx
export OPENAI_BASE_URL=https://your-api-endpoint/v1

### 4. 数据（可选）
export SMARTGRASP_DATA_DIR=/path/to/data
默认 ./data

## 运行

bash perception/run_perception.sh           # 全部场景（默认 perception + reason）
bash perception/run_perception.sh 59        # 单个（默认 perception + reason）
bash perception/run_perception.sh 59 242    # 多个（默认 perception + reason）
RUN_REASON_AFTER_PERCEPTION=0 bash perception/run_perception.sh 59 # 只跑 perception
MODE=gt bash perception/run_perception.sh 59 # GT 模式

输出: data/scene_{id}/perception/ 和 data/scene_{id}/reason/ ; 日志: logs/

## 参数速查

变量                      默认值      说明
------------------------------------------------------
MODE                       vlm        vlm / gt
SAM2_POINTS_PER_SIDE       24         SAM2 采样密度
SAM2_PRED_IOU_THRESH       0.68       候选质量阈值
SAM2_STABILITY_SCORE_THRESH 0.83      稳定性阈值
REVIEW_MODEL_ID            gpt-5.5    VLM 模型
REVIEW_TIMEOUT             300        API 超时(秒)
KERNEL_SIZE                11         遮挡检测核
DEVICE                     auto       cuda / cpu
