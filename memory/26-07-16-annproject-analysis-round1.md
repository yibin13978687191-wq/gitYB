# 26-07-16 — ANNProject 全面分析 + 模块化重构

## 基础信息

- 场景分类：**代码**（标识符: `coding`）
- 对话时间：2026-07-16
- 工作区：ANNProject（BP神经网络项目重构）
- 当前版本：24da5f9 → 进行中

## 任务内容

对 ANNProject 进行全量代码审查与模块化重构。重构方向：
1. 创建 `BaseNeuralNetwork` 抽象基类，定义模型标准接口
2. 精简 `bp_network.py`，去除混入类继承、改用组合模式
3. 修复 Dropout/BatchNorm 层构建 Bug
4. 统一导出入口、清理无用导入

## 完成状态

### ✅ 已完成

- [x] 全项目代码审查，整理出15项问题清单
- [x] 修复 `metrics.py` 中 `RegressionMetrics.compute()` 缺少 return 的致命 Bug + 移除死代码
- [x] 创建 `BaseNeuralNetwork` 抽象基类（`src/models/base.py`）
  - 定义 `forward()` 抽象方法 + `get_parameter_num()` 等工具方法
  - 定义可选钩子接口（`record_predictions`, `get_statistics` 等）
- [x] 重写 `bp_network.py`：
  - 继承 `BaseNeuralNetwork`，不再直接继承 `AnalysisMixin` / `VisualMixin`
  - 剥离 Dropout 为独立属性，不再放入 hidden 层列表
  - 修复 `batch_norm` 辅助函数直接索引而非 for 循环
  - 使用组合方式注入评估/可视化引擎
- [x] 修复 `trainer.py`：`loss.sum().backward()` → `loss.backward()`
- [x] 清理无用导入（`AnalysisMixin`, `VisualMixin` 从 trainer.py 移除）
- [x] 更新 `__init__.py` 文件，规范导出接口
- [x] 所有修改文件语法验证通过

### ⏳ 待办/未完成

- [ ] `TrainingMetrics` dataclass 可变默认值问题
- [ ] 统一全项目 `print()` 改用 `logging`
- [ ] 建立测试用例覆盖重构后的模型接口
- [ ] 将评估/可视化从模型钩子完全迁移到 Trainer 组合调用
- [ ] 设计 CNN 架构模板（复用 BaseNeuralNetwork 接口）

## 核心架构变更

```
旧: ANN(nn.Module, AnalysisMixin, VisualMixin)  ← 多重继承，职责耦合
新: ANN(BaseNeuralNetwork)                        ← 单一继承，组合扩展
    ├── self._analyzer   (NeuralNetworkAnalyzer)  ← 可选组合
    └── self._visualizer (NeuralNetworkVisualizer) ← 可选组合
```

## 建议

1. 下一轮重点修复 `TrainingMetrics` dataclass 的可变默认值问题
2. 考虑把 `record_predictions`/`get_statistics` 等钩子从模型剥离到 Trainer
3. 在现有接口上尝试搭建一个简单的 CNN 类以验证抽象设计
