# gitYB

ANN 学习与实验项目入口。

## 先看文档

- `docs/project-management-plan.md`
- `docs/ann-development-guide.md`

## 当前核心目录

- `ANNProject/src/ann_project/`：整理后的 ANN 核心代码包
- `ANNProject/scripts/`：可直接运行的训练或评估入口
- `ANNProject/data/`：按 `raw`、`interim`、`processed` 分层保存数据
- `ANNProject/notebooks/`：学习脚本、Notebook、探索性材料
- `ANNProject/outputs/`：训练结果、图表、指标和模型权重等产物
- `ANNProject/BPNetwork.py`、`ANNProject/Analysis_Tool/`：旧代码导入兼容入口

## Git 原则

- `main` 只放稳定版本
- 每次开发开独立分支
- 训练结果、缓存、IDE 文件默认不提交

## 下一步

当前已完成第一轮目录整理。下一步建议补依赖环境、运行基线脚本，再逐步把 `bp_network.py` 拆成模型、数据、训练和评估模块。
