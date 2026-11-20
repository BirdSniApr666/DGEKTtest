# DGEKT
The implementation of the paper *DGEKT: A Dual Graph Ensemble Learning Method for Knowlegde Tracing.*

## Abstract
Knowledge tracing, which aims to trace students’ evolving knowledge states based on their exercise-answering sequences, plays a critical role in supporting intelligent educational services. Recently, some graph-based models have been developed to incorporate the relationships between exercises to improve knowledge tracing. Despite the encouraging progress, only a single type of relationship information is generally explored. This paper presents a novel dual graph ensemble learning method for Knowledge Tracing (DGEKT), which establishes a dual graph structure of students’ learning interac tions to capture the heterogeneous exercise-concept associations and interaction transitions by hypergraph and directed graph modeling. Moreover, online knowledge distillation is introduced to form an ensemble teacher model from the dual graph models. In this way, besides the reference to students’ responses to a single exercise at the next step, the ensemble teacher model provides its predictions on all exercises as extra supervision for better modeling ability. In the experiments, we compare DGEKT against eight knowledge tracing baselines on three benchmark datasets, and the results demonstrate that DGEKT achieves state-of-the-art performance.

## Overall Architecture
![图片](https://user-images.githubusercontent.com/77867386/165916825-9c2135cc-d83b-43b4-82bb-c059a49af7e1.png)

## Dataset
We evaluate our method on three benchmark datasets for knowledge tracing, i.e., ASSIST09, ASSIST17, and EdNet.
In addition to the ASSIST17 dataset provided in the code, the ASSIST09 and EdNet datasets which mentioned in the paper are in the Google Drive, which you can download with this [link](https://drive.google.com/file/d/1ItqFv0fH6ibTotmflFNeAX0d7PdMaF7B/view?usp=sharing).
### Try using your own dataset!

You can also use your own data set, but note that besides the four row data set, you also need to build the incidence matrix of question and concept of the dataset and put it in the `/Dataset/H` folder.
## Models

 - `/KnowledgeTracing/model/Model.py`:end-to-end prediction framework;
 -  `/KnowledgeTracing/hgnn_models`:the module of Concept Association HyperGraph(CAHG);
 -  `KnowledgeTracing/DirectedGCN`:the module of Directed Transition Graph (DTG);
 - `/KnowledgeTracing/data/`:reading and processing datasets;
 - `/KnowledgeTracing/evaluation/eval.py`:Calculate losses and performance;

## Setup

To run this code you need the following:

    a machine with GPUs
    python3
    numpy, pandas, scipy, scikit-learn and torch packages:
```
pip3 install torch==1.7.0 numpy==1.21.2 pandas==1.4.1 scipy==1.7.3 scikit-learn==1.0.2 tqdm==4.26.3 
```
## Student-specific transition graph

The indexed pipeline was checked with PyTorch 2.5.1 (CUDA 12.1).
Use `--transition personal --alpha 5` for the prefix-based DTG. Each node is an
(exercise, observed answer) pair. After each response, the student's directed edge
counts are updated before predicting the following response. Counts carry across
chunks for the same student. The first DTG layer uses the normalized training
graph. In the second layer, incoming and outgoing rows are independently mixed
as `(alpha * population_row + prefix_counts) / (alpha + sum(prefix_counts))`.
An empty personal row falls back to its population row. Both graph modes use the
same CAHG, GRUs, heads and loss.

From the project root, with the authors' ASSIST09 train/test files in
`Dataset/assist2009/` and incidence matrix at `Dataset/H/2009.csv`, run:

```bash
python -m KnowledgeTracing.evaluation.run --dataset assist2009 --stage split
python -m KnowledgeTracing.evaluation.run --dataset assist2009 --transition static --stage train --epochs 20 --seed 216
python -m KnowledgeTracing.evaluation.run --dataset assist2009 --transition personal --alpha 5 --stage train --epochs 20 --seed 216
```

Use the same split and training seed for both modes. Choose any other fixed
alpha using validation AUC only. After selecting a configuration, run the test
and export stages with the same flags:

```bash
python -m KnowledgeTracing.evaluation.run --dataset assist2009 --transition personal --alpha 5 --stage test --seed 216
python -m KnowledgeTracing.evaluation.run --dataset assist2009 --transition personal --alpha 5 --stage export --seed 216
```

Use `--dataset assist2017` for the included data and split. On a 4 GB GPU start
with `--batch-size 16` for both configurations. On a 24 GB GPU the default batch
size is 128. Named runs keep separate outputs and checkpoints by dataset, graph,
alpha and seed, e.g. `outputs/assist2009/personal_alpha5.0/seed216/`.
Use `history.csv` for validation AUC and `metrics.json` for test AUC/ACC. The
prediction CSV's `prob_mean` is the probability used for both reported metrics.
Re-evaluate the static control with this version: it excludes padding in
one-interaction tail chunks that older metrics counted as incorrect answers.

## Cloud Studio 训练结束后自动释放算力

训练包装脚本在子进程退出后用端口预览发布状态。本机浏览器扩展读取状态后，会关闭所选应用在当前浏览器中的编辑页和学习页。Cloud Studio 通常在所有此类页面都关闭后约 10 分钟释放算力。[Cloud Studio 机时说明](https://ide.cloud.tencent.com/docs/guide/billing/machine_time/compute-time-introduction/)

1. 在本机 Chrome 打开 `chrome://extensions`，或在 Edge 打开 `edge://extensions`。开启开发者模式，点击「加载已解压的扩展程序」，选择本仓库的 `scripts/cloudstudio-browser` 目录。
2. 将 `scripts/cloudstudio_train.py` 同步到 Cloud Studio 的仓库。在云端 IDE 终端进入仓库根目录，用包装脚本在前台启动原训练命令。下面的示例会等到训练、测试和预测导出都完成。

   ```bash
   python scripts/cloudstudio_train.py -- python -u -m KnowledgeTracing.evaluation.run --dataset assist2017 --stage all --epochs 20
   ```

3. 在云端 IDE 的端口面板找到 `8765`，将访问权限设为「仅自己可见」，然后点「在浏览器中打开」。预览中会显示状态 JSON。[Cloud Studio 端口预览说明](https://ide.cloud.tencent.com/docs/guide/code_editing/productivity_plugin/port-plug/)
4. 在预览标签页点击扩展图标，选中对应应用并点「开始监控」。默认异常退出也会关页，可取消勾选。保存同一应用里的编辑内容并确认没有其他任务还在运行。

训练期间保持本机唤醒、浏览器联网和预览标签页打开。若浏览器启用了内存节省或睡眠标签页，将 Cloud Studio 编辑页和预览页加入保持活动的列表。其他设备或浏览器配置中同一应用的编辑页和学习页也需要关闭。确认释放可看 Buddy Academy 的「应用管理」。如果工作空间设了「定时关机」，确认剩余时间足够跑完整个任务。[Cloud Studio 定时关机说明](https://ide.cloud.tencent.com/docs/guide/code_editing/basic_operations/auto-shutdown/)

训练的模型、指标和项目日志仍写入原路径。包装脚本在训练退出后继续提供状态，便于浏览器读到最终结果。手动结束这个等待时，在云端终端按 `Ctrl+C`。

首次可以把训练命令换成下面的短任务，先验证关页流程：

```bash
python scripts/cloudstudio_train.py -- python -c 'import time; time.sleep(90)'
```

## Hyperparameter Settings

`/KnowledgeTracing/Constant/Constants.py` is specially used to set super parameters, all parameters of the whole model can be set in this file.

> MAX_STEP = 50 
> BATCH_SIZE = 128 
> LR = 0.001 
> EPOCH = 20 
> EMB = 256 
> HIDDEN = 128 
> kd_loss = 5.00E-06

## Save Log

If you need to save the log, create a `log` folder under the evaluation folder.
There are trained models in the model folder, which can be directly run `KTtest()` in  `run.py`  . 
Of course, you can also train a new models, just  run  `KTtrain()` in `run.py`

## Contact us
If you have any questions, please contact iext48@163.com.

