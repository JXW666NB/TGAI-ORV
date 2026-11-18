# 在 V100 上探查环境（版本、卡、路径、torch）
$ErrorActionPreference = 'Continue'
$orv = 'D:\orv_pip_0962\Scripts\python.exe'
$pyt = 'C:\Python311\python.exe'

"=== ORV ==="
& $orv -c "import tgai_go; print('ORV', tgai_go.__version__)"
"=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
"=== 路径 ==="
foreach ($p in @('D:\DSH-Workspace\TGAI-对话档案\_tgai_orv',
                 'D:\TGAI-V3-1.5B\03_src\model_v3.py',
                 'D:\orv_src\python\tgai_go\train\v3_csa.py',
                 'D:\orv_src\python\tgai_go\train\v3_pretrain.py')) {
    if (Test-Path -LiteralPath $p) { "  OK   $p" } else { "  缺失 $p" }
}
"=== 已装 tgai_go 里有分块注意力吗 ==="
& $orv -c "from tgai_go.train import v3_csa as C; print('_ATTN_CQ', C._ATTN_CQ, 'chunked', hasattr(C, '_attn_chunked'))"
"=== torch ==="
& $pyt -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
