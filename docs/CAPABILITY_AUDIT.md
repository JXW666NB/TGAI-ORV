# ORV 能力盘点（脚本生成，可复核）

> 由 `tools/inventory.py` 从**源码与测试**直接提取，不是手写记忆。
> 重新生成：`python tools/inventory.py`

## 汇总

| 项 | 数量 |
| --- | --- |
| Python 模块（不含下划线开头） | 49 |
| 后端接口方法（`rt.*`） | 233 |
| CUDA 内核 | 101 |
| `nn` 命名空间导出 | 16 |
| 核心优化器 | 13 |
| 测试文件 | 116 |
| 测试用例 | 616 |
| `cuda.py` 规模 | 281 KB |

## 模块清单

| 模块 | KB | 公开类 | 公开函数 |
| --- | --- | --- | --- |
| `align.py` | 7 | — | `logsigmoid`, `seq_logprob`, `dpo_pair_loss`, `dpo_batch_loss`, `group_advantages`, `token_logprobs`, `grpo_loss` |
| `amp.py` | 3 | — | `to_bf16_bits`, `from_bf16_bits`, `roundtrip_rel_error`, `amp_requested` |
| `amp_api.py` | 6 | `amp`, `GradScaler` | `amp_on`, `amp_off`, `amp_state`, `amp_invalidate` |
| `amp_mlp.py` | 13 | `GradScaler` | `enable`, `enabled`, `invalidate`, `to_f16`, `silu_mul_f16`, `silu_mul_bwd_f16`, `mlp` |
| `backend.py` | 7 | — | `core_methods`, `fallback_methods`, `report`, `required_methods`, `missing`, `cache_token`, `register`, `available` |
| `bpe.py` | 1 | — | `train_bpe`, `encode_ref` |
| `checkpoints.py` | 23 | `CheckpointError`, `Checkpoint` | `map_gguf_name`, `load_gguf`, `hf_name_to_ours`, `load_safetensors_dir`, `save_npz`, `load_npz`, `fix_linear_layout`, `norm_sanity` |
| `ckpt.py` | 8 | — | `read_tok_blob`, `names_sidecar_path`, `read_names_sidecar`, `iter_named_tensors`, `save_checkpoint`, `load_checkpoint`, `list_checkpoints`, `rotate` |
| `conv.py` | 12 | `Conv2d`, `Conv1d` | `conv2d`, `conv1d` |
| `cuda.py` | 281 | `CudaError`, `GpuBuf`, `CudaRuntime` | `fp16_env_requested`, `bf16_env_requested`, `fp16_split_env_requested`, `attn_nop_source_path`, `runtime` |
| `data.py` | 5 | `DataError` | `read_jsonl`, `normalize`, `from_hf`, `train_eval_split` |
| `dispatch.py` | 7 | `DispatchError` | `register_impl`, `resolve`, `available`, `clear`, `snapshot`, `benchmark_overhead` |
| `dist.py` | 7 | `DataParallel` | — |
| `dist_p2p.py` | 8 | `P2PError`, `P2PAllReduce` | `p2p_requested`, `rendezvous_dir` |
| `doctor.py` | 14 | — | `main` |
| `dtypes.py` | 3 | — | `f32_to_bf16_bits`, `bf16_bits_to_f32`, `to_storage`, `from_storage`, `cast_weights`, `save_weights`, `load_weights` |
| `export.py` | 7 | — | `collect`, `save_tg` |
| `graph_train.py` | 6 | `GraphStepError`, `GraphStep` | — |
| `indexing.py` | 6 | — | `index_select`, `gather_flat`, `scatter_flat` |
| `int8_adamw.py` | 6 | `Int8AdamW`, `Fp32AdamW`, `Fp16AdamW` | — |
| `interop.py` | 9 | `InteropError`, `OrvSequential` | `from_torch`, `to_torch`, `state_dict_from_torch`, `state_dict_to_torch` |
| `layer_graph.py` | 6 | `LayerGraphBackward` | — |
| `layers_extra.py` | 8 | `Dropout`, `MaxPool2d`, `AvgPool2d`, `MaxPool1d`, `AvgPool1d`, `BatchNorm1d` | — |
| `loader.py` | 5 | `Dataset`, `IterableDataset`, `Sampler`, `SequentialSampler`, `RandomSampler`, `BatchSampler` | `default_collate` |
| `nn_ops.py` | 24 | — | `rsqrt`, `pow`, `sigmoid`, `tanh`, `gelu`, `linear`, `rms_norm`, `layer_norm` |
| `norms.py` | 1 | `LayerNorm` | — |
| `ops_bundle.py` | 6 | `BundleError` | `load`, `inspect` |
| `ops_cpu.py` | 6 | — | `find_compiler`, `load_source`, `register_elementwise` |
| `ops_cuda.py` | 6 | — | `load_source`, `load_cubin`, `launch`, `register_elementwise` |
| `ops_native.py` | 4 | — | `supported`, `register`, `status` |
| `ops_plugin.py` | 12 | — | `register`, `register_numpy`, `unregister`, `list_ops`, `registered_names`, `load_plugin_file`, `check_contract`, `manifest` |
| `ops_vk.py` | 4 | — | `available`, `devices`, `load_spirv` |
| `ops_vk_compute.py` | 24 | `VulkanComputeError`, `VulkanCompute` | — |
| `ops_vk_ops.py` | 5 | — | `compile_glsl`, `spv_of`, `device`, `available`, `register_elementwise` |
| `opt_extra.py` | 10 | `SGD`, `Lion`, `Accumulator`, `Adafactor` | `scale_grads` |
| `opt_fp16mv.py` | 9 | `Fp16MomAdamW` | `save_opt_states`, `load_opt_states` |
| `param_group.py` | 7 | `SharedStorageError`, `ParameterGroup` | — |
| `pipeline.py` | 10 | — | `run` |
| `plugin.py` | 18 | `PluginError` | `op`, `resolve_op`, `bind`, `dispatch_overhead`, `op_numpy`, `op_native`, `load_manifest`, `native_info` |
| `plugin_manifest.py` | 4 | — | `parse_args_spec`, `load_manifest` |
| `ppo.py` | 5 | — | `gae`, `discounted_returns`, `ppo_loss` |
| `preset.py` | 11 | `Transformer`, `Tokenizer` | `transformer`, `tokens`, `train_tokenizer`, `load_corpus`, `fit` |
| `qwen.py` | 28 | `Qwen25` | `rope_inv_freq`, `rope_tables`, `load_tokenizer`, `build_sequences`, `sft`, `hf_name_to_ours`, `load_safetensors_weights`, `make_tiny_weights` |
| `redux_extra.py` | 3 | — | `max`, `min`, `var`, `std`, `norm` |
| `rl_trainer.py` | 9 | `DPOTrainer`, `GRPOTrainer`, `PPOTrainer` | — |
| `sched.py` | 4 | `ConstantLR`, `LinearLR`, `StepLR`, `CosineLR`, `LambdaLR` | `make_scheduler` |
| `sft_data.py` | 2 | `SftDataset` | `iter_jsonl_texts` |
| `v27.py` | 3 | — | `cfg`, `v27_names`, `to_orv`, `check_coverage` |
| `v27_train.py` | 17 | `V27Trainer`, `TGState` | — |

## `nn` 命名空间

`Module` `CausalSelfAttention` `MoE` `Parameter` `Linear` `Embedding` `RMSNorm` `Sequential` `ModuleList` `functional` `builder` `ConstantLR` `LinearLR` `CosineLR` `StepLR` `LambdaLR`

## 优化器

`AdamW` `LAMB` `Muon` `Module` `CausalSelfAttention` `MoE` `Parameter` `Linear` `Embedding` `RMSNorm` `Sequential` `ModuleList` `functional`

## CUDA 内核（101 个）

`adamw_update` `adamw_update_multi` `adamw_update_multi_f16` `attn_mask_softmax` `attn_mask_softmax_bwd` `bcast_add4` `bcast_add4_v4` `broadcast_to8` `cast_bf16_to_f32` `cast_f16_to_f32` `cast_f32_to_bf16` `cast_f32_to_f16` `cat_extract` `cat_place` `ce_bwd` `ce_fwd_mean` `ce_fwd_rows` `emb_bwd_fill` `emb_bwd_hist` `emb_bwd_owned` `emb_bwd_plan` `emb_bwd_plan2` `emb_bwd_rows` `emb_bwd_rows4` `emb_gather_bwd` `emb_gather_fwd` `ewise_abs` `ewise_abs_bwd` `ewise_add` `ewise_add_v4` `ewise_clamp` `ewise_clamp_bwd` `ewise_div` `ewise_div_scalar` `ewise_exp` `ewise_exp32` `ewise_exp_v4` `ewise_log` `ewise_log32` `ewise_mul` `ewise_mul_v4` `ewise_pow` `ewise_recip` `ewise_relu` `ewise_relu_bwd` `ewise_scale` `ewise_scale_v4` `ewise_sub_scale` `fused_add_ip` `fused_add_ip_v4` `fused_rope_bwd` `fused_rope_bwd_v4` `fused_rope_ip` `fused_rope_ip_v4` `gather_flat` `gather_rows` `index_gather8` `index_scatter_add8` `log_softmax_lastdim` `log_softmax_lastdim_bwd` `moe_compact` `moe_gather_pairs` `moe_gather_rows` `moe_gather_slots` `moe_pad_rows` `moe_row_dot` `moe_row_scale` `moe_scatter_pairs` `moe_scatter_rows` `moe_scatter_slots` `moe_sigmoid` `moe_topk` `muon_momentum` `permute8` `permute8_v4` `reduce_sum` `repeat_reduce` `rms_dw_det` `rms_rstd` `rmsnorm_bwd` `rmsnorm_bwd_dx` `rmsnorm_fwd` `rope_half_bwd` `rope_half_ip` `scatter_add_flat` `scatter_add_rows` `sgemm_compose` `sgemm_mode` `silu_bwd_da` `silu_fwd_inplace` `silu_mul_bwd` `silu_mul_fwd` `softmax_lastdim` `softmax_lastdim_bwd` `sum_axis0_rows` `sum_axis8` `sumsq_final` `sumsq_partial` `sumsq_sequential` `swap_last2` `transpose2d`

## 后端接口方法（233 个）

`__del__` `_activate` `_alloc` `_arena_alloc` `_attn_flash_launch` `_attn_nop_launch` `_attn_set_dyn_smem` `_bcast4_vec_ok` `_bf16_epoch_bump` `_bf16_fail` `_bf16_state` `_bf16_wanted` `_bind_driver` `_bind_dto_async` `_bind_gemm_ex` `_bind_graph_api` `_bind_nvrtc` `_bind_sym` `_bmm_op` `_broadcast_lead` `_broadcast_shapes` `_capture_keep` `_cast_f16_cached` `_cast_flat_ptr` `_cast_flat_ptr_bf16` `_cat_geom` `_cc_arch` `_chk` `_chk_blas` `_chk_nvrtc` `_collect_adamw` `_compile` `_cublas_path` `_d2d_bytes` `_err_text` `_ewise_grid` `_fn` `_fn_module` `_fp16_fail` `_fp16_state` `_gemm_ex_f16` `_gemm_f16_raw` `_graph_destroy` `_graph_disable` `_graph_node_list` `_grouped_fail_note` `_grouped_ready` `_grouped_sgemm` `_grouped_tn_strided` `_in_arena` `_index_i32` `_index_launch_args` `_init_cublas` `_init_driver` `_init_kernels` `_instantiate_graph` `_left_pad8` `_make_tmpl` `_mark_submitted` `_memset32` `_moe_row_block` `_note_adamw` `_numpy_exact_gemm` `_nvrtc_log` `_nvrtc_path` `_pad4` `_pairs` `_pool_checkout` `_pool_ev_acquire` `_pool_frozen` `_pool_note` `_pool_reclaim_idle` `_pool_release` `_pool_shed_ready` `_pool_shrink_idle` `_pool_take` `_pool_token_of` `_pool_trim` `_pool_trim_to` `_ptr` `_raise_alloc_failed` `_rms_rstd_scratch` `_row_grid` `_row_strides` `_sgemm` `_sgemm_bf16` `_sgemm_f16` `_split16_fail` `_split16_pair` `_split16_state` `_split16_wanted` `_split_forget_ptr` `_strided_ready` `_strides4` `_vec4_grid` `_vec4_ok` `_warp_row_grid` `abs` `abs_bwd` `adamw` `adamw_multi` `adamw_multi_f16` `add` `add_` `add_broadcast` `attn_mask_softmax_` `attn_mask_softmax_bwd` `attn_nop_bwd` `attn_nop_bwd_flash` `attn_nop_fwd` `attn_nop_fwd_stats` `bmm` `broadcast_to` `cast_f16_to_f32` `cast_f32_to_f16` `cast_f32_to_f16_into` `cat_extract` `cat_place` `ce_bwd` `ce_fwd` `clamp` `clamp_bwd` `clone` `d2h` `div` `div_scalar` `emb_bwd` `emb_fwd` `empty` `empty_cache` `ensure` `ensure_attn_nop` `exp` `exp32` `f16_cache_clear` `f16_cache_enable` `f16_cache_stats` `fp16_gemm_on` `free` `free_device_bytes` `full` `fused_rope_` `fused_rope_bwd` `gather_flat` `gather_rows` `gemm_f16_nn_raw` `gemm_f16_nt_raw` `gemm_f16_tn_raw` `gemm_nt` `gemm_tn` `graph_abort` `graph_arm` `graph_available` `graph_begin` `graph_count` `graph_end` `graph_launch` `graph_launch_slot` `graph_release_all` `graph_stats` `grouped_nn` `grouped_nt` `grouped_tn` `h2d` `index_gather` `index_scatter_add` `launch` `log` `log32` `log_softmax_lastdim` `log_softmax_lastdim_bwd` `matmul_general` `moe_compact` `moe_gather_pairs` `moe_gather_rows` `moe_gather_slots` `moe_row_dot` `moe_row_scale` `moe_scatter_pairs` `moe_scatter_rows` `moe_scatter_rows_add` `moe_scatter_slots` `moe_sigmoid` `moe_topk` `mul` `muon_momentum` `permute` `pool_stats` `pow_scalar` `read_scalar` `recip` `release_capture_arena` `relu` `relu_bwd` `repeat_reduce` `reserve_capture_arena` `reshape` `rmsnorm_bwd` `rmsnorm_bwd_det` `rmsnorm_fwd` `rope_half_bwd` `rope_half_ip` `scale` `scale_inplace` `scatter_add_flat` `scatter_add_rows` `sgemm` `sgemm_compose` `sgemm_fp16` `sgemm_fp16_nt` `sgemm_mode` `sgemm_nt_strided_batched` `sgemm_split16` `sgemm_strided_batched` `sgemm_tn_strided_batched` `silu_bwd_da` `silu_fwd_inplace` `silu_mul_bwd` `silu_mul_fwd` `softmax_lastdim` `softmax_lastdim_bwd` `sub_scale_` `sub_view` `sum_all` `sum_axis` `sumsq_parallel` `sumsq_sequential` `swap_last2` `sync` `to_numpy` `transpose2d` `unbroadcast` `zeros`

## 测试文件与用例数

| 文件 | 用例 |
| --- | --- |
| `test_adafactor.py` | 6 |
| `test_adamw_parity.py` | 2 |
| `test_amp_api.py` | 7 |
| `test_amp_bf16.py` | 6 |
| `test_amp_readiness.py` | 3 |
| `test_amp_scaler.py` | 6 |
| `test_api_completeness.py` | 8 |
| `test_api_pretrain.py` | 6 |
| `test_api_train.py` | 9 |
| `test_backend_coverage.py` | 4 |
| `test_backward_parity.py` | 2 |
| `test_beginner_pipeline.py` | 5 |
| `test_bpe.py` | 7 |
| `test_causal_self_attention.py` | 3 |
| `test_checkpoint_cuda.py` | 4 |
| `test_checkpoint_grad.py` | 2 |
| `test_checkpoints.py` | 21 |
| `test_chunked_ce.py` | 4 |
| `test_ckpt.py` | 4 |
| `test_ckpt_names.py` | 3 |
| `test_ckpt_segments.py` | 3 |
| `test_clip_accum.py` | 7 |
| `test_contrib_ops.py` | 8 |
| `test_conv.py` | 4 |
| `test_conv_cuda.py` | 2 |
| `test_cpu.py` | 8 |
| `test_cuda_selfcheck.py` | 2 |
| `test_custom_model_parity.py` | 1 |
| `test_data_layer.py` | 12 |
| `test_dataset_report.py` | 4 |
| `test_deterministic.py` | 1 |
| `test_dist_cpu.py` | 1 |
| `test_dist_p2p.py` | 5 |
| `test_doc_op_coverage.py` | 3 |
| `test_docs_consistency.py` | 2 |
| `test_dpo.py` | 6 |
| `test_dtypes.py` | 7 |
| `test_edge_cases.py` | 7 |
| `test_einsum.py` | 5 |
| `test_embedding_weight.py` | 4 |
| `test_error_messages.py` | 3 |
| `test_export_guard.py` | 4 |
| `test_fp16_thresholds.py` | 2 |
| `test_fused_attn_block.py` | 2 |
| `test_fused_attn_math.py` | 3 |
| `test_gguf_mapping.py` | 7 |
| `test_grad_release.py` | 2 |
| `test_graph_train.py` | 5 |
| `test_grpo.py` | 6 |
| `test_hygiene.py` | 4 |
| `test_indexing.py` | 7 |
| `test_int8_adamw.py` | 7 |
| `test_interop.py` | 7 |
| `test_interop_migration.py` | 5 |
| `test_layer_graph_budget.py` | 1 |
| `test_layernorm.py` | 4 |
| `test_layers_extra.py` | 9 |
| `test_llama_structure.py` | 1 |
| `test_load_stream.py` | 5 |
| `test_loader.py` | 9 |
| `test_masked_fill_cuda.py` | 4 |
| `test_moe_aux_free.py` | 6 |
| `test_moe_batch.py` | 1 |
| `test_moe_capacity.py` | 9 |
| `test_moe_engine_parity.py` | 2 |
| `test_moe_grouped_fallback.py` | 4 |
| `test_moe_shared_experts.py` | 7 |
| `test_moe_z_loss.py` | 6 |
| `test_muon.py` | 9 |
| `test_muon_ns_numerics.py` | 4 |
| `test_nn_ops.py` | 14 |
| `test_nn_ops_batch2.py` | 12 |
| `test_norm_sanity.py` | 5 |
| `test_numpy_backend.py` | 11 |
| `test_op_plugins.py` | 10 |
| `test_ops_bundle.py` | 7 |
| `test_ops_cpu.py` | 4 |
| `test_ops_cuda.py` | 2 |
| `test_ops_native.py` | 4 |
| `test_ops_vk.py` | 4 |
| `test_ops_vk_ops.py` | 4 |
| `test_opt_extra_and_sched.py` | 10 |
| `test_optimizer_parity.py` | 1 |
| `test_optimizers_extra.py` | 5 |
| `test_param_group.py` | 11 |
| `test_plugin_contract.py` | 6 |
| `test_plugin_dispatch.py` | 12 |
| `test_plugin_native.py` | 14 |
| `test_plugin_system.py` | 14 |
| `test_ppo.py` | 9 |
| `test_ppo_trainer.py` | 4 |
| `test_preset_quickstart.py` | 10 |
| `test_qwen25_forward.py` | 2 |
| `test_qwen3_forward.py` | 8 |
| `test_qwen_grad_ckpt.py` | 3 |
| `test_qwen_sft_smoke.py` | 3 |
| `test_reductions.py` | 7 |
| `test_resume.py` | 2 |
| `test_rl_trainer.py` | 5 |
| `test_rope_fullwidth.py` | 1 |
| `test_rope_half.py` | 2 |
| `test_rope_scaling.py` | 4 |
| `test_sdpa_lowmem.py` | 2 |
| `test_sft_data.py` | 3 |
| `test_slice_probe.py` | 2 |
| `test_sweep_report.py` | 7 |
| `test_tg4.py` | 5 |
| `test_tg_audit.py` | 5 |
| `test_tg_bf16.py` | 4 |
| `test_tg_check.py` | 5 |
| `test_tg_layout.py` | 5 |
| `test_tg_names.py` | 3 |
| `test_tgop_e2e.py` | 3 |
| `test_torch_compat_plugin.py` | 7 |
| `test_v27_train.py` | 3 |
| `test_version.py` | 3 |
