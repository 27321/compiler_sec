本仓库是对模型编译阶段的攻击与安全检测实现。


安全检测主要有以下三项：
1. 对FX Graph阶段产物fx_graph_readable.py、fx_graph_runnable.py、fx_graph_transformed.py
进行单文件检测；
2. 对Fusion阶段前后产物ir_pre_fusion.txt、ir_post_fusion.txt进行差分分析；
3. 对Code Generation阶段产物.source文件与.cubin文件反汇编得到的.sass文件进行差分分析。



（1）fusion_injection文件夹是对Fusion阶段的攻击以及对攻击得到的产物进行检测。
修改源文件：/root/miniconda3/envs/hf_env/lib/python3.11/site-packages/torch/_inductor/scheduler.py
scheduler_original.py是原始文件备份，scheduler.py是将其替换的攻击版本文件。
使用以下命令进行替换：
cp /workspace/Documents/pytorch2_wjk/fusion_injection/scheduler.py /root/miniconda3/envs/hf_env/lib/python3.11/site-packages/torch/_inductor/scheduler.py
然后运行fushion_inject_demo.py文件进行一次编译并获得编译产物，
运行以下命令对两目标文件进行差分分析：
python check_ir_anomalies.py \
  --pre torch_compile_debug/run_2026_02_03_02_09_01_374307-
pid_62441/torchinductor/model__0_inference_0.0/ir_pre_fusion.txt \
  --post torch_compile_debug/run_2026_02_03_02_09_01_374307-
pid_62441/torchinductor/model__0_inference_0.0/ir_post_fusion.txt \
  --out ir_anomaly_report.txt
（compile_fx1.py等文件为攻击前期探索产物，由于未达到ir_pre_fusion.txt干净而ir_post_fusion.txt中毒的攻击效果，最终没有采用）
fushion_injection.md是攻击方法文档。



（2）kernel_injection文件夹是对Code Generation阶段的攻击以及对攻击得到的产物进行检测。
修改源文件：/root/miniconda3/envs/hf_env/lib/python3.11/site-packages/triton/backends/nvidia/compiler.py
triton_nvidia_compiler_original.py是原始文件备份，triton_nvidia_compiler.py是将其替换的攻击版本文件。
使用以下命令进行替换：
cp ./triton_nvidia_compiler.py      /root/miniconda3/envs/hf_env/lib/python3.11/site-packages/triton/backends/nvidia/compiler.py
然后运行ptx_inject_demo.py文件进行一次编译并获得编译产物，
反汇编命令：
nvdisasm triton_per_fused_embedding_mean_0.cubin > triton_per_fused_embedding_mean_0.sass
运行以下命令对两目标文件进行差分分析：
python triton_kernel_integrity_checker.py triton_cache/5XEUUDPE6FZ4JL57ZK2L3RSZOCMWNPG26X7P3IMF6WJDYB56JYBQ/triton_per_fused_embedding_mean_0.source triton_cache/5XEUUDPE6FZ4JL57ZK2L3RSZOCMWNPG26X7P3IMF6WJDYB56JYBQ/triton_per_fused_embedding_mean_0.sass
ptx_fusion.md是攻击方法文档，source_sass_check_method.md是检测方法文档。
