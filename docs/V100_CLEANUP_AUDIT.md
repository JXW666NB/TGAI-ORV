# V100 系统清理审计（2026-11-14，只读 + 已做备份）

> 工具：`tools/probe_v100_audit.py`（体检，按节跑：`1` 任务 / `2` 自启动 / `3` 进程 / `4` 浏览器）
> 备份：`tools/probe_task_backup.py` → **278 个任务的 XML + tasks.csv 落在 V100 `D:\_task_backup`**
> 这一步只读、只写备份目录，没有删除任何东西。

## 1. 计划任务：278 个（非 Microsoft 路径）

| 前缀 | 数量 | 指向 |
| --- | --- | --- |
| `TGAI*` | 72 | `D:\tgai\_*.bat` 等早期实验 |
| `TAMUC-*` | 56 | `D:\tayauotmuc\_*.bat`（分词/训练实验批） |
| 其余一次性 | ~150 | `tgai_acc2` / `tgai_accept` / `tgai_api` / `TGAI_burn4` / `tgai_chunk2`… |
| OneDrive | 3 | 系统自带 |
| NVIDIA | 1 | 驱动自带 |

**状态**：绝大多数是「就绪」，没有正在运行的（推理服务被停后）。

## 2. 数据目录（真正的空间大头）

| 路径 | 文件数 | 占用 | 性质 |
| --- | --- | --- | --- |
| `D:\tayauotmuc` | 1,150 | **168.7 GB** | 早期分词/训练实验（TAMUC 批处理指向它） |
| `D:\tgai` | 3,618 | **52.1 GB** | 早期实验 + 可能含推理模型（**先甄别再动**） |
| `D:\orv_pip_0962` | 3,554 | 80.4 MB | 当前 ORV venv（**保留**） |
| `D:\orv_src` | 1,318 | 110.9 MB | 代码源（**保留**） |

## 3. 浏览器：**没有发现 2345 劫持**

| 检查项 | 结果 |
| --- | --- |
| 已安装浏览器 | 只有 **Edge**（`C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe`） |
| IE 主页（HKCU\...\Internet Explorer\Main）| `http://go.microsoft.com/fwlink/p/?LinkId=255141`（微软默认） |
| Edge 组策略（HKLM/HKCU Policies\Microsoft\Edge）| 空 |
| Chrome 组策略 | 空；Chrome **未安装** |
| Edge 配置 `homepage` / `startup_urls` | 均为空 |
| 桌面/公共桌面/开始菜单/**任务栏固定项**里的 .lnk | **无任何含 "2345" 的快捷方式** |

⇒ 这台机器当前是干净的；2345 要么之前已清掉，要么不在这一台。

## 4. 建议的清理顺序（**待确认后执行**）

1. **删任务（可回滚）**：`TAMUC-*` 56 个 + 一次性 `tgai_*` 约 150 个。
   回滚方式：`schtasks /create /xml D:\_task_backup\<name>.xml /tn <name>`。
2. **删数据（不可回滚）**：`D:\tayauotmuc`（168.7 GB）。删前先确认里面没有要留的语料/权重。
3. **甄别 `D:\tgai`**：52.1 GB，可能含 `TGAI_NB_V2.7_v4.tg` 等推理模型（推理服务正指向它）
   —— **只删实验产物，不动模型**。
4. 保留：`D:\tgai_orv`、`D:\orv_pip_0962`、`D:\orv_src`、`D:\TGAI-V3-1.5B`（V3 训练资产）。
