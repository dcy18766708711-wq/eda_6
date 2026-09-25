#!/usr/bin/env python3
"""
电路框图分析完整提交脚本
集成Task1（组件检测和连接识别）和Task2（电路问答）
"""

import os
import sys
import json
import torch
import argparse
import warnings
from pathlib import Path
from PIL import Image
from typing import Dict, List, Optional

# 导入依赖库
from ultralytics import YOLO
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import PeftModel

warnings.filterwarnings('ignore')


class YOLODetector:
    """YOLO组件检测器"""

    def __init__(self, model_path: str, conf_threshold: float = 0.25, device: int = 0):
        """
        初始化YOLO检测器

        Args:
            model_path: 模型权重文件路径
            conf_threshold: 置信度阈值
            device: 推理设备 (0 for GPU, -1 for CPU)
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        print(f"[YOLO] 加载模型: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = device
        print("[YOLO] 模型加载完成")

    def predict(self, image_path: str) -> Dict:
        """
        对单张图片进行YOLO推理

        Args:
            image_path: 图片路径

        Returns:
            包含组件信息的字典
        """
        results = self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            imgsz=640,
            save=False,
            device=self.device,
            verbose=False
        )

        result = results[0]
        boxes = result.boxes

        # 获取原始图片尺寸
        image = Image.open(image_path).convert("RGB")
        img_width, img_height = image.size

        output_data = {
            "components": [],
            "image_info": {
                "width": img_width,
                "height": img_height,
                "path": image_path
            }
        }

        if len(boxes) > 0:
            # 提取坐标 (xyxy格式: [x1, y1, x2, y2])
            xyxy_coords = boxes.xyxy.cpu().numpy().astype(int)

            for i, box in enumerate(xyxy_coords):
                component_entry = {
                    "Component": f"Component_{i + 1}",
                    "Pos": box.tolist(),  # [x1, y1, x2, y2]
                    "I_O": {"input": 0, "output": 1},
                    "Connection": {"input": [], "output": []}
                }
                output_data["components"].append(component_entry)

        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return output_data


class CoordinateScaler:
    """坐标缩放器"""

    MAX_DIMENSION = 1008  # 最大维度阈值

    @staticmethod
    def scale_image_and_coords(image_path: str, json_data: Dict) -> tuple:
        """
        缩放图片和坐标，保持宽高比

        Args:
            image_path: 原始图片路径
            json_data: YOLO输出的JSON数据

        Returns:
            缩放后的图片、更新后的JSON数据、缩放因子
        """
        with Image.open(image_path).convert("RGB") as img:
            original_width, original_height = img.size
            max_dim = max(original_width, original_height)

            scale_factor = 1.0
            target_width, target_height = original_width, original_height

            # 如果超过最大维度，进行缩放
            if max_dim > CoordinateScaler.MAX_DIMENSION:
                scale_factor = CoordinateScaler.MAX_DIMENSION / max_dim
                target_width = round(original_width * scale_factor)
                target_height = round(original_height * scale_factor)

                # 缩放图片
                resized_img = img.resize((target_width, target_height), Image.LANCZOS)
                print(f"[Scaler] 图片缩放: {original_width}x{original_height} -> {target_width}x{target_height}")
            else:
                resized_img = img.copy()
                print(f"[Scaler] 图片无需缩放: {original_width}x{original_height}")

            # 更新坐标
            updated_data = json_data.copy()
            updated_data["components"] = []

            for component in json_data["components"]:
                updated_component = component.copy()
                if "Pos" in component and isinstance(component["Pos"], list) and len(component["Pos"]) == 4:
                    # 应用缩放: [x1, y1, x2, y2]
                    new_pos = [
                        round(component["Pos"][0] * scale_factor),
                        round(component["Pos"][1] * scale_factor),
                        round(component["Pos"][2] * scale_factor),
                        round(component["Pos"][3] * scale_factor)
                    ]
                    updated_component["Pos"] = new_pos
                updated_data["components"].append(updated_component)

            # 记录缩放信息
            updated_data["scale_info"] = {
                "original_size": [original_width, original_height],
                "resized_size": [target_width, target_height],
                "scale_factor": float(scale_factor)
            }

            return resized_img, updated_data, scale_factor


class LLMInference:
    """大模型推理器 - 支持LoRA适配器"""

    def __init__(self, model_path: str, lora_checkpoint: str = None):
        """
        初始化大模型推理器

        Args:
            model_path: 基础模型路径
            lora_checkpoint: LoRA微调后的检查点路径（可选）
        """
        print(f"[LLM] 加载基础模型: {model_path}")
        self.model_path = model_path
        self.lora_checkpoint = lora_checkpoint

        # 确定数据类型
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        # 从预训练模型加载基础模型
        # 使用device_map="auto"让transformers自动处理设备分配
        # 由于此时GPU中只有一个模型，不会出现显存不足的问题
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=self.dtype,
            device_map="auto",
            attn_implementation="eager"
        )

        # 如果提供了LoRA检查点，加载LoRA适配器
        if lora_checkpoint:
            print(f"[LLM] 加载LoRA适配器: {lora_checkpoint}")

            # 检查LoRA检查点目录是否存在
            if not os.path.isdir(lora_checkpoint):
                raise FileNotFoundError(f"LoRA检查点目录不存在: {lora_checkpoint}")

            # 检查必要的LoRA文件
            required_files = ['adapter_config.json', 'adapter_model.safetensors']
            for file in required_files:
                file_path = os.path.join(lora_checkpoint, file)
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"LoRA文件缺失: {file_path}")

            try:
                # 加载LoRA权重 - 直接从checkpoint读取adapter_config.json
                # 这与infer_task2_simple.py的加载方式一致，确保推理结果相同
                self.model = PeftModel.from_pretrained(
                    self.model,
                    model_id=lora_checkpoint
                )

                # 验证LoRA是否成功加载
                if not hasattr(self.model, 'active_adapters'):
                    raise RuntimeError("LoRA适配器加载失败：模型缺少active_adapters属性")

                # 检查是否有活跃的适配器
                active_adapters = self.model.active_adapters() if callable(self.model.active_adapters) else self.model.active_adapters
                if not active_adapters:
                    raise RuntimeError(f"LoRA适配器未被正确激活。活跃适配器: {active_adapters}")

            except Exception as e:
                print(f"[LLM] ✗ LoRA适配器加载失败: {e}")
                raise
        else:
            print("[LLM] ℹ 未指定LoRA检查点，使用基础模型")

        # 加载处理器
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model.eval()

        print("[LLM] 模型加载完成")

    def infer_task1(self, image: Image.Image, user_prompt: str) -> str:
        """
        Task1推理：识别连接关系

        Args:
            image: PIL图像对象
            user_prompt: 用户提示词

        Returns:
            模型输出的JSON字符串
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": user_prompt},
                ],
            }
        ]

        # 准备输入
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")

        # 执行推理
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=4096, temperature=0.1, top_p=0.9)

        # 解码输出
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        response = output_text[0] if isinstance(output_text, list) else str(output_text)
        return response

    def infer_task2(self, image: Image.Image, question: str, options: List[str] = None,
                    max_new_tokens: int = 128, temperature: float = 0.7, top_p: float = 0.9) -> str:
        """
        Task2推理：电路问答

        Args:
            image: PIL图像对象
            question: 问题文本
            options: 选项列表（单选题时使用）
            max_new_tokens: 最大生成令牌数
            temperature: 生成温度
            top_p: nucleus采样参数

        Returns:
            生成的答案
        """
        # 构造用户提示
        if options:
            user_content = f"{question}\n直接给出选项结果\n" + "\n".join(options)
        else:
            user_content = question

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": user_content},
                ],
            }
        ]

        # 准备输入
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")

        # 执行推理
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True
            )

        # 解码输出
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        answer = response[0].strip() if isinstance(response, list) else str(response).strip()
        return answer


class CircuitAnalysisPipeline:
    """完整的电路分析pipeline"""

    def __init__(self, yolo_model_path: str, llm_model_path: str, task1_lora_checkpoint: str = None,
                 task2_lora_checkpoint: str = None, conf_threshold: float = 0.5):
        """
        初始化pipeline

        Args:
            yolo_model_path: YOLO模型路径
            llm_model_path: 大模型路径（基础模型）
            task1_lora_checkpoint: Task1 LoRA微调后的检查点路径（可选）
            task2_lora_checkpoint: Task2 LoRA微调后的检查点路径（可选）
            conf_threshold: YOLO置信度阈值
        """
        self.yolo_detector = YOLODetector(yolo_model_path, conf_threshold=conf_threshold)
        self.task1_lora_checkpoint = task1_lora_checkpoint
        self.task2_lora_checkpoint = task2_lora_checkpoint
        self.llm_model_path = llm_model_path
        self.scaler = CoordinateScaler()

        # 初始化时先只加载Task2推理器（不加载Task1以保证Task2的纯净状态）
        print("[Init] 初始化Task2推理器...")
        self.llm_task2 = LLMInference(llm_model_path, lora_checkpoint=task2_lora_checkpoint)
        print("[Init] Task2推理器初始化完成")

        # Task1推理器延迟加载标志
        self.llm_task1 = None
        self.task1_llm_loaded = False

    def _build_user_prompt_task1(self, scaled_json: Dict) -> str:
        """
        构建Task1的用户提示词

        Args:
            scaled_json: 缩放后的YOLO检测结果

        Returns:
            用户提示词字符串
        """
        # 构建组件坐标映射部分
        components_mapping = "组件坐标位置映射（格式：[左上X, 左上Y, 右下X, 右下Y]）：\n"
        for component in scaled_json["components"]:
            comp_name = component["Component"]
            pos = component["Pos"]
            components_mapping += f"- {comp_name}: {pos}\n"

        # 构建完整的用户提示词
        user_prompt = f"""请分析下面的电路框图，根据提供的组件坐标位置和图像中的连接关系，提取所有组件的输出连接关系。

{components_mapping}

分析步骤：
1. 在图像中根据坐标定位出每个组件的位置信息
2. 观察从每个组件出发的箭头或连线
3. 沿着连线追踪到组件
4. 如果一个组件输出到多个目标，列出所有目标

输出格式（JSON数组）：
[
  {{"组件名称1": "输出组件1名称,输出组件2名称,..."}},
  {{"组件名称2": "输出组件名称,..."}},
  ...
]

"""
        return user_prompt

    def process_image(self, image_path: str, task2_questions: Dict = None) -> Dict:
        """
        处理单张图片，完整pipeline

        Args:
            image_path: 输入图片路径
            task2_questions: Task2的问题JSON

        Returns:
            包含完整分析结果的字典
        """
        print(f"\n{'='*60}")
        print(f"处理图片: {os.path.basename(image_path)}")
        print(f"{'='*60}")

        # Step 1: YOLO检测组件
        print("\n[Step 1] YOLO检测组件...")
        yolo_result = self.yolo_detector.predict(image_path)
        print(f"[YOLO] 检测到 {len(yolo_result['components'])} 个组件")

        # 保存原始YOLO坐标（用于最终JSON输出）
        original_yolo_result = {
            "components": [comp.copy() for comp in yolo_result["components"]],
            "image_info": yolo_result["image_info"].copy()
        }

        # Step 2: 坐标缩放
        print("\n[Step 2] 缩放坐标...")
        resized_image, scaled_json, scale_factor = self.scaler.scale_image_and_coords(
            image_path, yolo_result
        )

        # Step 3: Task1 - 识别连接关系
        print("\n[Step 3] Task1 - 识别连接关系...")
        user_prompt = self._build_user_prompt_task1(scaled_json)
        print("[LLM] 正在执行Task1推理...")

        # 在加载Task1之前，清空Task2推理器以释放GPU显存
        print("[Clean] 清空Task2推理器以释放显存...")
        del self.llm_task2
        self.llm_task2 = None
        torch.cuda.empty_cache()

        # 加载Task1推理器
        print("[Init] 加载Task1推理器...")
        self.llm_task1 = LLMInference(self.llm_model_path, lora_checkpoint=self.task1_lora_checkpoint)
        self.task1_llm_loaded = True
        print("[Init] Task1推理器加载完成")

        # 执行Task1推理
        llm_response_task1 = self.llm_task1.infer_task1(resized_image, user_prompt)

        # 解析Task1结果
        print("[LLM] 解析Task1输出...")
        task1_result = []
        try:
            # 尝试提取JSON数组部分
            json_start = llm_response_task1.find('[')
            json_end = llm_response_task1.rfind(']') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = llm_response_task1[json_start:json_end]
                connection_list = json.loads(json_str)

                # 融合YOLO结果和LLM识别的连接
                connection_dict = {}
                if isinstance(connection_list, list):
                    for item in connection_list:
                        if isinstance(item, dict):
                            connection_dict.update(item)

                # 构建最终的Task1结果（使用原始YOLO坐标）
                for component in original_yolo_result["components"]:
                    comp_name = component["Component"]
                    task1_entry = component.copy()

                    # 从LLM结果中获取连接关系
                    if comp_name in connection_dict:
                        output_str = connection_dict[comp_name]
                        output_list = [x.strip() for x in output_str.split(',') if x.strip()]
                        task1_entry["Connection"]["output"] = output_list

                    task1_result.append(task1_entry)

                print("[LLM] 成功解析Task1输出")
        except Exception as e:
            print(f"[Warning] Task1解析异常: {e}，使用原始YOLO结果")
            task1_result = original_yolo_result["components"]

        # Step 4: Task2 - 处理问题
        print("\n[Step 4] Task2 - 电路问答...")

        # 清空Task1推理器，重新加载Task2推理器以获得纯净的模型状态
        print("[Clean] 清空Task1推理器...")
        del self.llm_task1
        self.llm_task1 = None
        self.task1_llm_loaded = False
        torch.cuda.empty_cache()

        print("[Reload] 重新加载Task2推理器...")
        self.llm_task2 = LLMInference(self.llm_model_path, lora_checkpoint=self.task2_lora_checkpoint)

        task2_result = []

        if task2_questions:
            for question_item in task2_questions:
                print(f"[Task2] 处理问题: {question_item['question'][:50]}...")

                options = question_item.get('options', None)

                # 清理选项中的LaTeX公式，简化为更简单的格式
                # 这是为了确保和infer_task2_simple.py的行为一致
                if options and isinstance(options, list):
                    cleaned_options = []
                    for opt in options:
                        # 只保留选项的前缀（A/B/C/D. 部分）
                        if opt and len(opt) > 0:
                            # 如果是纯文本选项，保持原样
                            # 如果是LaTeX选项，也保持原样（因为这是题目要求）
                            cleaned_options.append(opt)
                    options = cleaned_options if cleaned_options else None

                answer = self.llm_task2.infer_task2(
                    resized_image,
                    question_item['question'],
                    options=options,
                    max_new_tokens=128,
                    temperature=0.7,
                    top_p=0.9
                )

                # 构建Task2结果
                result_item = {
                    "type": question_item["type"],
                    "question": question_item["question"],
                    "answer": answer
                }

                if options:
                    result_item["options"] = options

                task2_result.append(result_item)
                print(f"[Task2] 答案: {answer}")

        # Step 5: 融合最终结果
        final_result = {
            "task1": task1_result,
            "task2": task2_result
        }

        print(f"{'='*60}\n")

        return final_result


def get_model_paths(base_dir: str = None) -> Dict[str, str]:
    """
    获取模型路径，优先使用评测系统路径

    Args:
        base_dir: 基础目录路径

    Returns:
        包含模型路径的字典
    """
    if base_dir is None:
        # 默认使用当前脚本所在目录
        base_dir = os.path.dirname(os.path.abspath(__file__))

    return {
        "yolo_model": os.path.join(base_dir, "models", "yolo_best.pt"),
        "task1_lora_model": os.path.join(base_dir, "models", "task1_model"),
        "task2_lora_model": os.path.join(base_dir, "models", "task2_model"),
        "base_model": os.path.join(base_dir, "large_model"),  # 使用本地large_model目录
    }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="电路框图分析和问答系统")
    parser.add_argument("--image_path", required=True, help="输入图像文件夹路径")
    parser.add_argument("--task2_question_path", required=True, help="task2问题文件夹路径")
    parser.add_argument("--output_path", required=True, help="输出结果目录路径")

    args = parser.parse_args()

    # 检查输入路径
    if not os.path.isdir(args.image_path):
        print(f"错误：输入图像文件夹不存在 -> {args.image_path}")
        return

    if not os.path.isdir(args.task2_question_path):
        print(f"错误：task2问题文件夹不存在 -> {args.task2_question_path}")
        return

    # 创建输出目录
    os.makedirs(args.output_path, exist_ok=True)

    # 获取模型路径
    model_paths = get_model_paths()

    # 检查模型文件是否存在
    if not os.path.exists(model_paths["yolo_model"]):
        print(f"错误：YOLO模型不存在 -> {model_paths['yolo_model']}")
        return

    if not os.path.isdir(model_paths["task1_lora_model"]):
        print(f"错误：Task1 LoRA模型不存在 -> {model_paths['task1_lora_model']}")
        return

    if not os.path.isdir(model_paths["task2_lora_model"]):
        print(f"错误：Task2 LoRA模型不存在 -> {model_paths['task2_lora_model']}")
        return

    print("[Init] 初始化Pipeline...")
    pipeline = CircuitAnalysisPipeline(
        yolo_model_path=model_paths["yolo_model"],
        llm_model_path=model_paths["base_model"],
        task1_lora_checkpoint=model_paths["task1_lora_model"],
        task2_lora_checkpoint=model_paths["task2_lora_model"],
        conf_threshold=0.4
    )

    # 获取所有图片
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = sorted([
        f for f in os.listdir(args.image_path)
        if f.lower().endswith(image_extensions)
    ])

    if not image_files:
        print(f"错误：在 {args.image_path} 中未找到图片文件")
        return

    print(f"\n[Info] 找到 {len(image_files)} 张图片\n")

    # 处理每张图片
    for image_file in image_files:
        image_path = os.path.join(args.image_path, image_file)

        # 获取对应的task2问题
        base_name = os.path.splitext(image_file)[0]
        task2_question_file = os.path.join(args.task2_question_path, f"{base_name}.json")

        task2_questions = None
        if os.path.exists(task2_question_file):
            try:
                with open(task2_question_file, 'r', encoding='utf-8') as f:
                    task2_data = json.load(f)
                    task2_questions = task2_data.get("task2", [])
            except Exception as e:
                print(f"[Warning] 读取Task2问题文件失败 {task2_question_file}: {e}")

        # 处理图片
        try:
            result = pipeline.process_image(image_path, task2_questions=task2_questions)

            # 保存结果
            output_file = os.path.join(args.output_path, f"{base_name}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            print(f"[Success] 结果已保存: {output_file}")

        except Exception as e:
            print(f"[Error] 处理 {image_file} 失败: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n[Complete] 处理完成！结果保存在: {args.output_path}")


if __name__ == "__main__":
    main()
