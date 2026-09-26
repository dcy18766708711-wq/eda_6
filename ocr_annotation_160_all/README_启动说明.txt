1. 首次运行：bash install_cpu.sh
2. 启动标注：bash start_cpu.sh
3. 浏览器打开：http://127.0.0.1:8765
4. 标注结果：result/annotations/
5. 将整个 result 目录发回汇总者。

如果端口被占用：PORT=8767 bash start_cpu.sh
如果只手工输入文字且不需要自动 OCR，可在 start_cpu.sh 的命令末尾增加 --no-ocr-assist。
