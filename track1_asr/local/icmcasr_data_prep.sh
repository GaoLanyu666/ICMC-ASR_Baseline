. ./path.sh || exit 1

stage=0
stop_stage=2
nj=48

. tools/parse_options.sh

data_root=$1
enhanced_data_root=$2
dataset=$3

# AEC+IVA对train，dev，eval三个数据集进行AEC+IVA处理
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "[local/icmcasr_data_prep.sh] stage 0: AEC + IVA Enhancement"
  local/enhancement.sh --nj ${nj} ${data_root} ${enhanced_data_root} ${dataset}
fi

# 根据TextGrid生成txt文件，分割原有近场音频文件，分割增强远场音频文件
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "[local/icmcasr_data_prep.sh] stage 1: Segment ${dataset} wavs"
  python3 local/segment_wavs.py ${data_root} ${enhanced_data_root} ${dataset} ${nj}
fi

# 根据 TextGrid 文件和音频文件生成 wav.scp、text 和 utt2spk 文件，存放在data文件夹下，而不是数据集文件夹
# 对文本进行规范化处理。
# 生成 spk2utt 文件并验证数据目录
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "[local/icmcasr_data_prep.sh] stage 2: Prepare data files"
  python3 local/data_prep.py ${data_root} ${enhanced_data_root} ${dataset}
fi
