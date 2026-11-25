# 🔬 皮肤镜病变 7 分类挑战（DermaMNIST）

> MedCraft「见微」论坛第二期 · 现场实践　|　⏱ 限时 30 分钟

**一句话**：给一张皮肤镜照片，判断它属于 7 类皮肤病变里的哪一类。

---

## 📌 赛题背景
皮肤镜是皮肤科医生看痣和皮损时用的放大镜头。本题的照片来自公开的皮肤镜图库，每张都由医生给出了诊断类别。你的任务是训练一个模型，自动给没见过的照片分类。

## 🚀 快速开始
1. 下载 `DermaMNIST.zip`（群文件，或本仓库），解压到一个新文件夹。
2. 用 Trae 打开解压出的 `DermaMNIST/` 文件夹，点"信任"。
3. 让 AI 先读这份 README，用大白话讲清：输入是什么、要输出什么、怎么交。

## 🗂 数据
每张图是 28×28 像素的彩色 PNG。

| 文件夹 / 文件 | 内容 | 张数 |
|---|---|---|
| `train/` | 训练用的图 + 标签 `labels.csv` | 7007 |
| `val/` | 自查用的图 + 标签 `labels.csv` | 1003 |
| `test/` | 要你预测的图，**没有标签** | 2005 |
| `sample_submission.csv` | 提交格式样例 | 2005 行 |
| `score.py` | 查提交格式；也能在 `val/` 上算几项常用指标 | |

`labels.csv` 的列：`filename`（图片文件名）、`label`（类别编号）、`name_zh`（中文名）、`name_en`（英文名）。

**7 个类别**

| 编号 | 中文名 | 英文名 |
|---|---|---|
| 0 | 光化性角化病和上皮内癌 | actinic keratoses and intraepithelial carcinoma |
| 1 | 基底细胞癌 | basal cell carcinoma |
| 2 | 良性角化样病变 | benign keratosis-like lesions |
| 3 | 皮肤纤维瘤 | dermatofibroma |
| 4 | 黑色素瘤 | melanoma |
| 5 | 黑素细胞痣 | melanocytic nevi |
| 6 | 血管病变 | vascular lesions |

## 📤 提交格式
交一个 `submission.csv`，和 `sample_submission.csv` 格式一样：

```
id,label
test_0000,5
test_0001,2
...
```

- 两列 `id,label`；`test/` 里每张图一行，共 2005 行；
- `id` 是图片文件名去掉 `.png`（如 `test_0000`），`label` 填 0–6 的类别编号；
- 交之前在 `DermaMNIST/` 文件夹里运行 `python score.py --check submission.csv` 检查格式。

## 📮 怎么交
全程在 GitHub 网页上点，不用配命令行（先在浏览器里登录 GitHub）：
1. 让 AI 在本地准备好一个 `队名/` 文件夹，里面三样东西：`submission.csv`、代码、一个写了几句"我们怎么做的"的 `README.md`（不放模型权重和数据）。
2. 打开本仓库 https://github.com/MedCraft-ShanghaiTech/dermamnist-practice ，点右上角 **Fork**。
3. 在你 fork 出来的仓库里进 `submissions/` → **Add file → Upload files**，把 `队名` 文件夹整个拖进去 → **Commit changes**。
4. 回到你 fork 的首页 → **Contribute → Open pull request → Create pull request**。

30 分钟内可以再交：同样上传，覆盖 `submissions/队名/submission.csv`（文件名不变，不要另存成 v2 之类），PR 会自动更新，不用新开；以最后一次为准。

电脑上 git 已经能推送的，也可以让 AI 推到你的 fork，你再开 PR。不要把 GitHub 的密码或 token 发给 AI。

实在提不了 PR，就把 `submission.csv` 改名成"队名.csv"，微信发给工作组。

## 🏆 评分与榜单
- 限时 30 分钟，到点后由组织者统一打分（测试集答案只在组织者手里）；
- 评分方式打分时揭晓；
- 当场投屏榜单，只显示队名，**不评奖**。

## 📜 规则
- 只用本数据包里的文件，**不要下载 MedMNIST 原版**（原版带测试集答案）；
- 模型、方法、AI 工具都不限。

## 💡 建议
先搞清任务，跑通交一版，再改进；注意时限和效率：训练要多久，跟数据量、网络大小（权重多少）和复杂度有关；省着用积分，一个人的用完了换队友的账号；卡住了先把报错原样发给 AI。

## 📚 数据来源与许可
MedMNIST v2 的 DermaMNIST 子集（原图来自 HAM10000），CC BY-NC 4.0，仅限非商业用途。引用：Yang J, et al. MedMNIST v2. *Scientific Data*, 2023；Tschandl P, et al. The HAM10000 dataset. *Scientific Data* 5, 180161 (2018)。
