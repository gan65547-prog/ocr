# RK3588 OCR 与 Whisper 部署记录及 SOP

整理时间：2026-11-26。设备：gan@192.168.1.108，aarch64。代码仓库： https://github.com/gan65547-prog/ocr 。本文已提交到代码仓库。

## 现有资产与边界

- 板端 OCR：`/home/gan/ocr_test`，包含 `test_ocr_board.py`、检测、识别及方向分类 RKNN 模型、字典和测试图片。脚本实际只执行检测和识别，未调用分类模型。
- 板端语音：`/home/gan/whisper-rknn`，包含 encoder/decoder ONNX、RKNN、转换日志、词表、滤波器、测试音频和 Python 脚本。不要将个人录音或未经许可的数据公开。
- 板端 Whisper 转换环境 `~/whisper-rknn/venv`：rknn-toolkit2 2.3.2，rknn-toolkit-lite2 2.5.0。实测 runtime 2.5.0、驱动 0.9.8。转换日志显示 decoder 部分操作回退 CPU，不能假设全部在 NPU 执行。
- 尚未找到原始模型下载地址、ONNX 导出脚本、OCR 转换脚本和原始测试基准；不能据此声称已完整复现从原始权重到 ONNX 或 OCR RKNN 的转换。模型来源、版本与许可证需进一步核实。

## 1. 准备与备份

```bash
ssh gan@192.168.1.108
uname -m
ls -lh ~/ocr_test ~/whisper-rknn/model
```

保留现有模型、字典、音频和转换日志的备份。仓库只包含恢复的脚本和必要文本资产；不包含 RKNN、ONNX、运行库、音频或图片。新设备需从已核实来源取得对应模型与运行环境。OCR 脚本默认使用 `~/ocr_test` 下模型和字典；Whisper 示例依赖 `whisper/model` 目录的相对路径。

## 2. Whisper ONNX 转 RKNN

已有板端 ONNX 可作为转换输入；该命令使用 toolkit2，FP 模式不使用量化。保持已有文件不被覆盖时指定新输出路径：

```bash
cd ~/whisper-rknn
./venv/bin/python python/convert.py encoder.onnx rk3588 fp encoder-new.rknn
./venv/bin/python python/convert.py decoder.onnx rk3588 fp decoder-new.rknn
```

确认返回成功并检查日志、文件大小；新模型应先与既有模型对比验证，再替换部署文件。量化模式需要校准数据，现有脚本未提供数据集参数，不应直接当成完整量化流程。原始 ONNX 的生成流程尚待恢复。

## 3. Whisper 板端测试

```bash
cd ~/whisper-rknn/python
../venv/bin/python whisper_board.py \
  --encoder_model_path ../model/encoder.rknn \
  --decoder_model_path ../model/decoder.rknn \
  --task zh --audio_path ../model/test_zh.wav
```

`--task` 支持 `zh` 或 `en`，输入音频由脚本转换至单声道 16 kHz；示例以 20 秒为处理长度。2026-11-26 在现有设备执行成功，输出：`对我做了介绍,我想说的是,如果对我的研究感兴趣`。这是单条样本结果，并非准确率或性能基准。运行日志显示静态模型的动态范围查询警告，未阻止本次推理。

## 4. OCR 板端测试

```bash
cd ~/ocr_test
python3 test_ocr_board.py test_cn_en.png
```

脚本使用 `ppocrv4_det_960.rknn`、`ppocrv4_rec_320.rknn` 和 `ppocr_keys_v1.txt`，输出检测框数量、耗时和识别文字。其检测后处理为简化实现，应使用标注样本检查结果，不能视作官方完整 PP-OCR 流程。

2026-11-26 使用系统 Python 复测时，`import cv2` 失败：`libavcodec.so.60: undefined symbol: x264_encoder_open_164`，该次运行未完成 OCR 推理；之后使用现有隔离环境成功运行，结果见第 6.2 节。应先检查 OpenCV、FFmpeg、libx264 的动态库来源及版本，在独立环境修复依赖后重试，避免直接覆盖系统库。方向分类模型虽然存在，但当前脚本未使用。

## 5. 交付检查

确认脚本与模型匹配，分别记录模型来源、哈希、转换命令、环境版本、测试样本授权、实际输出和耗时；对 OCR 检测框及文字、Whisper 中英文样本分别评估。仓库当前为恢复的实验代码，尚需核对上游来源和许可证后再作为正式发行版本使用。

---

## 6. 使用现存资产的恢复与验证（2026-11-26 补充）

本节不要求重新下载模型，以 `gan@192.168.1.108` 的现存文件为源。先确保源设备可访问；如果它和备份都不可用，单靠 GitHub 仓库无法凭空恢复模型。以下命令不会覆盖原模型或修改系统库。

### 6.1 检查源文件并备份

在操作电脑执行（目标目录应有至少 1 GB 可用空间）：

```bash
mkdir -p rk3588-backup/ocr rk3588-backup/whisper/model rk3588-backup/whisper/python
scp gan@192.168.1.108:~/ocr_test/ppocrv4_det_960.rknn rk3588-backup/ocr/
scp gan@192.168.1.108:~/ocr_test/ppocrv4_rec_320.rknn rk3588-backup/ocr/
scp gan@192.168.1.108:~/ocr_test/ppocr_keys_v1.txt rk3588-backup/ocr/
scp gan@192.168.1.108:~/ocr_test/test_ocr_board.py rk3588-backup/ocr/
scp gan@192.168.1.108:~/ocr_test/test_cn_en.png rk3588-backup/ocr/
scp gan@192.168.1.108:~/whisper-rknn/encoder.onnx rk3588-backup/whisper/
scp gan@192.168.1.108:~/whisper-rknn/decoder.onnx rk3588-backup/whisper/
scp gan@192.168.1.108:'~/whisper-rknn/model/*.rknn' rk3588-backup/whisper/model/
scp gan@192.168.1.108:'~/whisper-rknn/model/*.txt' rk3588-backup/whisper/model/
scp gan@192.168.1.108:~/whisper-rknn/model/test_zh.wav rk3588-backup/whisper/model/
scp gan@192.168.1.108:'~/whisper-rknn/python/*.py' rk3588-backup/whisper/python/
```

音频、图片可能包含私人内容，备份后妥善保管，不要上传公开仓库。复制后用 `sha256sum`（macOS 用 `shasum -a 256`）检查；关键文件的 SHA-256：

```text
ppocrv4_det_960.rknn  0d227525ec763e07f8478d7f2b22c4be3998562a1d282bee124b6511048343db
ppocrv4_rec_320.rknn  340134f787916b174cec210b50b62b17ae569f41c711c85ebef4b2f03ab3a28a
encoder.onnx          bb3b84cae0bbecf22d07f3dc1406390edcb4e2efcaa90db996816538294c3e09
decoder.onnx          c7dae5e7685050babf92babeca09cf25e40fb3b394265c34b501afe2270cfd18
encoder.rknn          1ac16a165c156bcd89ebd17ca7697c4822912ed98f2d50a3e8d8af160c535a79
decoder.rknn          5a5ea0049f230f38f28248b5c41ce3299c345083aac33781245214f8e7d66b83
```

请将所有备份文件的哈希与源设备对照。备份还应另行保存设备系统镜像或可恢复的依赖包；Python 虚拟环境不可假定能直接复制到不同路径或系统运行。

### 6.2 原设备 OCR 验证（已实测）

系统 Python 的 FFmpeg 依赖损坏，使用已存在的隔离环境：

```bash
ssh gan@192.168.1.108
cd ~/ocr_test
~/whisper-rknn/venv/bin/python test_ocr_board.py test_cn_en.png
```

2026-11-26 实测检测到 6 个框，检测约 165 ms，单框识别约 14–16 ms；例如识别出“体感小游戏 RK3588测试”和“人民币99.50元折扣8.5折”。英文及部分文本有错误，属于实验结果，不代表准确率保证。`/lib/aarch64-linux-gnu/libx264.so.164` 实际导出的是 `x264_encoder_open_163`，而系统 FFmpeg 需要 `x264_encoder_open_164`；不要通过改系统软链接或覆盖库来修复。

### 6.3 原设备 Whisper 验证（已实测）

执行第 3 节命令。使用备份中的原有 encoder/decoder ONNX 按第 2 节重新转换时，输出放在新文件名，核对推理成功后才考虑切换。Whisper 的 ONNX 文件哈希与本次核对的瑞芯微 Model Zoo 对应 ONNX 一致；这只证明文件相同，不表示要重新下载。

### 6.4 恢复到另一台设备前必须满足的条件

新设备需要兼容 RK3588 NPU 驱动和运行库，Python 3.12 及与板端架构匹配的依赖。原设备已记录：`rknn-toolkit2 2.3.2`、`rknn-toolkit-lite2 2.5.0`、`numpy 1.26.4`、`opencv-python 4.11.0.86`、`onnx 1.16.1`、`onnxruntime 1.26.0`、`torch 2.2.0`、`scipy 1.17.1`、`soundfile 0.14.0`、`setuptools 80.10.2`；请同时保存 `~/whisper-rknn/venv/bin/pip freeze` 结果及对应安装包。将 OCR 备份放至目标用户的 `~/ocr_test`，Whisper 放至 `~/whisper-rknn`，在目标设备重建依赖后执行上面的两项测试。

**验证边界：** 上述 OCR 和 Whisper 是在原设备已有虚拟环境及已有模型上实测通过；尚未在干净系统重建完整依赖并验证，因此不能保证另一台设备仅按当前文档一次成功。现存 OCR 的 960 检测模型对应的 ONNX 和转换过程尚未找到，无法从原始 OCR 模型重建该文件。不同版本模型不能随意替换：官方示例检测 ONNX 输入为 480×480，现有 OCR 脚本固定使用 960×960，直接混用会出错或得到错误结果。
