from collections import defaultdict

from llmc.utils.registry_factory import MODEL_REGISTRY
from transformers import AutoConfig, SmolVLMForConditionalGeneration
from loguru import logger
from .base_model import BaseModel
import torch
from transformers.models.llama.modeling_llama import LlamaRMSNorm
# from .smolvlm_model import SmolVLMAutoModelForCausalLM
from llmc.compression.quantization.module_utils import (
    _LLMC_LINEAR_TYPES_, _LLMC_LN_TYPES_, _TRANSFORMERS_LINEAR_TYPES_,
    _TRANSFORMERS_LN_TYPES_, LlmcFp8Linear)

@MODEL_REGISTRY
class SmolVLM2(BaseModel):
    def __init__(self, config, device_map=None, use_cache=False):
        super().__init__(config, device_map, use_cache)
        self.vision_prefix = "model.vision_model"
        self.text_prefix = "model.text_model"

        self._init_modality_specific_params()
        # 添加兼容性属性
        self.linear_blocks = []  # 用于兼容旧式索引访问
        self.block_modality_map = {}  # 记录块所属模态

    def _init_modality_specific_params(self):
        """初始化多模态专用参数"""
        self.blocks = {
            "vision": [],
            "text": []
        }
        self.vision_embeds = []
        self.text_embeds = []
        self.block_name_prefix = {}
        self.pairs = {}

    def _init_mm_specific_params(self):
        """多模态专用参数初始化"""
        self.vision_model = None
        self.text_model = None
        self.connector = None
        self._device = 'cpu'  # 初始设备状态跟踪
    def add_layernorms_class(self):
        """重写方法：兼容多模态LayerNorm注册"""
        ln_class_list = []

        # 遍历所有模态的第一个块
        for modality, blocks in self._blocks_dict.items():
            if not blocks:
                continue

            first_block = blocks[0]
            ln_dict = self.get_layernorms_in_block(first_block)

            for ln_name in ln_dict:
                ln_class = ln_dict[ln_name].__class__
                if ln_class not in ln_class_list:
                    ln_class_list.append(ln_class)

        # 注册新发现的LayerNorm类型
        for ln_class in ln_class_list:
            if ln_class not in _TRANSFORMERS_LN_TYPES_:
                _TRANSFORMERS_LN_TYPES_.append(ln_class)

        # logger.info(f'Registered LayerNorm types: {_TRANSFORMERS_LN_TYPES_}')
    # ----------------- 核心抽象方法实现 -----------------
    # 修改build_model方法
    def build_model(self):
        self.model_config = AutoConfig.from_pretrained(
            self.model_path,
            trust_remote_code=True,  # 必须启用
            model_type="smolvlm",  # 显式指定类型
            torch_dtype=torch.bfloat16  # 强制指定配置类型
        )

        # 使用自定义加载器
        self.model = SmolVLMForConditionalGeneration.from_pretrained(
            self.model_path,
            config=self.model_config,
            device_map=self.device_map,
            trust_remote_code=True,  # 关键参数
            torch_dtype=torch.bfloat16,  # 统一加载类型
            low_cpu_mem_usage=True,
        )
        # 修正lm_head数据类型
        if self.model.lm_head.weight.dtype != torch.bfloat16:
            self.model.lm_head = self.model.lm_head.to(torch.bfloat16)

        logger.info(f"lm_head dtype: {self.model.lm_head.weight.dtype}")
        # 验证类型一致性
        text_emb = self.model.model.text_model.embed_tokens
        assert text_emb.weight.dtype == torch.bfloat16, "文本嵌入层类型错误"
        assert self.model.lm_head.weight.dtype == torch.bfloat16, "输出头类型错误"
        # 初始化组件引用
        self.vision_model = self.model.model.vision_model
        self.text_model = self.model.model.text_model
        self.connector = self.model.model.connector

        # 统一设备初始化
        self._sync_device()

    def _sync_device(self):
        """统一设备名称格式"""
        # 确保使用带索引的CUDA名称
        target_device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
        self._device = target_device  # 存储为torch.device对象

        # 转移视觉模型（保持BF16）
        self.vision_model = self.vision_model.to(target_device, dtype=torch.bfloat16)

        # 转移文本模型
        self.text_model = self.text_model.to(target_device)

        # 转移连接器（显式指定设备）
        self.connector = self.connector.to(target_device)

        # 特殊处理连接器投影层
        proj_layer = self.connector.modality_projection.proj
        proj_layer = proj_layer.to(target_device)

        # 验证设备格式统一
        vision_param = next(self.vision_model.parameters())
        text_param = next(self.text_model.parameters())
        connector_param = next(proj_layer.parameters())

        assert str(vision_param.device) == str(target_device), "视觉模型设备未同步"
        assert str(text_param.device) == str(target_device), "文本模型设备未同步"
        assert str(connector_param.device) == str(target_device), "连接器设备未同步"

    def _validate_connector_device(self):
        """修正设备验证方式"""
        proj_layer = self.connector.modality_projection.proj
        proj_param = next(proj_layer.parameters())

        # 统一转换为字符串比较
        current_device_str = str(self._device)
        connector_device_str = str(proj_param.device)

        assert connector_device_str.startswith(current_device_str.split(':')[0]), \
            f"设备类型不匹配: {connector_device_str} vs {current_device_str}"
        assert connector_device_str == current_device_str, \
            f"设备索引不匹配: {connector_device_str} vs {current_device_str}"

    def collect_first_block_input(self, calib_data, padding_mask=None):
        """增强版数据收集方法"""
        # 强制启用CUDA设备
        self._device = torch.device("cuda")
        self._sync_device()
        self._validate_connector_device()  # 新增验证

        first_block_input = defaultdict(list)
        Catcher = self._create_mm_catcher(first_block_input)

        # 替换第一个块并保持梯度
        original_block = self.blocks[0]
        self.blocks[0] = Catcher(original_block).to(self._device).requires_grad_(False)

        # 增强数据预处理
        processed_data = self._preprocess_mm_data(calib_data)

        # 调试信息
        print(f"校准数据样本数: {len(processed_data)}")
        print(f"第一个样本键值: {processed_data[0].keys()}")

        # 执行前向传播（带梯度跟踪）
        with torch.enable_grad():
            self._run_mm_forward(processed_data)

        # 验证捕获结果
        # 修改后的断言
        assert len(first_block_input) > 0, (
            f"数据捕获失败！设备链验证：\n"
            f"Vision: {next(self.vision_model.parameters()).device}\n"
            f"Text: {next(self.text_model.parameters()).device}\n"
            f"Connector: {next(self.connector.modality_projection.proj.parameters()).device}"
        )

        self.first_block_input = first_block_input
        return

    def _create_mm_catcher(self, buffer):
        """改进的捕获器实现"""

        class MMCatcher(torch.nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module
                self.hook_handle = None

            def forward_hook(self, module, inputs, outputs):
                # 直接捕获第一个块的输入
                hidden_states = inputs[0] if isinstance(inputs, tuple) else inputs
                buffer['data'].append(hidden_states.detach().cpu())

            def forward(self, hidden_states, **kwargs):
                # 注册前向钩子
                self.hook_handle = self.module.register_forward_hook(self.forward_hook)
                return self.module(hidden_states, **kwargs)

        return MMCatcher

    def _preprocess_mm_data(self, calib_data):
        """增强数据类型转换"""
        processed = []
        for data in calib_data:
            # 生成视觉输入（确保BF16类型）
            pixel = data.get('pixel_values', torch.randn(1, 3, 224, 224))
            pixel = pixel.to(dtype=torch.bfloat16)  # 新增类型转换

            # 添加num_images维度
            pixel = pixel.unsqueeze(1)  # [1,1,3,224,224]

            mm_data = {
                'input_ids': data.get('input_ids', torch.zeros(1, 512, dtype=torch.long)),
                'pixel_values': pixel,
                'attention_mask': data.get('attention_mask', torch.ones(1, 512, dtype=torch.long))
            }

            # 设备转移并验证
            mm_data = {k: v.to(self._device).contiguous() if isinstance(v, torch.Tensor) else v
                       for k, v in mm_data.items()}

            # 类型验证
            assert mm_data['pixel_values'].dtype == torch.bfloat16, \
                f"视觉输入类型错误！应为bfloat16，实际为{mm_data['pixel_values'].dtype}"

            processed.append(mm_data)
        return processed


    def _run_mm_forward(self, data):
        """增强前向传播"""
        with torch.no_grad():
            for sample in data:
                # 显式构造输入字典
                inputs = {
                    'input_ids': sample['input_ids'],
                    'pixel_values': sample['pixel_values'],
                    'attention_mask': sample['attention_mask'],
                    'output_hidden_states': True,
                    'return_dict': True
                }


                # 强制使用完整前向传播
                outputs = self.model(**inputs)
                # 触发梯度计算（即使不更新参数）
                outputs.loss.backward() if hasattr(outputs, 'loss') else None


    @property
    def blocks(self):
        """兼容属性：返回合并后的块列表"""
        return self.linear_blocks

    @blocks.setter
    def blocks(self, value):
        """设置器：保持多模态结构的同时维护线性列表"""
        if isinstance(value, dict):
            # 多模态结构
            self._blocks_dict = value
            self.linear_blocks = []
            idx = 0
            for modality, blocks in value.items():
                self.linear_blocks.extend(blocks)
                for _ in blocks:
                    self.block_modality_map[idx] = modality
                    idx += 1
        else:
            # 单模态结构（兼容旧代码）
            self.linear_blocks = value
    def find_blocks(self):
        """改造后的块定位方法"""
        # 实际存储为字典结构
        self._blocks_dict = {
            "vision": self.model.model.vision_model.encoder.layers,
            "text": self.model.model.text_model.layers
        }
        # 自动维护线性索引
        self.blocks = self._blocks_dict  # 触发setter

    def find_embed_layers(self):
        """定位所有嵌入层（修正路径）"""
        # 视觉嵌入
        self.vision_embeds = [
            self.model.model.vision_model.embeddings.patch_embedding,  # 增加.model
            self.model.model.vision_model.embeddings.position_embedding  # 增加.model
        ]

        # 文本嵌入
        self.text_embeds = [self.model.model.text_model.embed_tokens]  # 增加.model

    def get_embed_layers(self):
        """获取全部嵌入层"""
        return self.vision_embeds + self.text_embeds

    def get_layers_except_blocks(self):
        """获取非块结构层（修正路径）"""
        return [
            *self.vision_embeds,
            self.model.model.vision_model.post_layernorm,  # 增加.model
            *self.text_embeds,
            self.model.model.text_model.norm,  # 增加.model
            self.model.model.connector.modality_projection.proj,  # 增加.model
            self.model.lm_head
        ]

    def get_subsets_in_block(self, block):
        """分模态处理量化子集"""
        if self._is_vision_block(block):
            return self._vision_subsets(block)
        return self._text_subsets(block)

    def skip_layer_name(self):
        """指定跳过量化的层"""
        return [
            'connector.modality_projection',  # 连接器投影
            'post_layernorm',  # 视觉后归一化
            'rotary_emb',  # RoPE位置编码
            'lm_head'  # 输出头（可选）
        ]

    def has_bias(self):
        """视觉部分有bias，文本部分无bias"""
        return True

    # ----------------- 模态特定方法 -----------------
    def find_block_name(self):
        """定义块名前缀映射"""
        self.block_name_prefix = {
            'vision': 'model.vision_model.encoder.layers',
            'text': 'model.text_model.layers'
        }
        self.pairs = {
            'vision': {'q_proj': 'qkv', 'out_proj': 'out', 'fc1': 'fc1'},
            'text': {'q_proj': 'qkv', 'o_proj': 'out', 'up_proj': 'fc1'}
        }

    def get_head_layers(self):
        """获取输出头层"""
        return [self.model.lm_head]

    def get_pre_head_layernorm_layers(self):
        """获取头前归一化层（修正路径）"""
        return [
            self.model.model.vision_model.post_layernorm,
            self.model.model.text_model.norm
        ]

    # ----------------- 私有辅助方法 -----------------
    def _is_vision_block(self, block):
        """通过视觉块特有结构判断"""
        return hasattr(block, 'layer_norm1') and \
            hasattr(block, 'layer_norm2') and \
            hasattr(block.mlp, 'fc1')
    def _vision_subsets(self, block):
        """视觉编码器量化子集构造"""
        subsets = []

        # 注意力层
        subsets.extend([
            {
                'layers': {
                    'self_attn.q_proj': block.self_attn.q_proj,
                    'self_attn.k_proj': block.self_attn.k_proj,
                    'self_attn.v_proj': block.self_attn.v_proj,
                },
                'prev_op': [block.layer_norm1],
                'input': ['self_attn.q_proj'],
                'inspect': block.self_attn,
                'has_kwargs': True
            },
            {
                'layers': {'self_attn.out_proj': block.self_attn.out_proj},
                'prev_op': [block.self_attn.v_proj],
                'input': ['self_attn.out_proj'],
                'inspect': block.self_attn.out_proj,
                'has_kwargs': False
            }
        ])

        # MLP层
        subsets.append({
            'layers': {
                'mlp.fc1': block.mlp.fc1,
                'mlp.fc2': block.mlp.fc2,
            },
            'prev_op': [block.layer_norm2],
            'input': ['mlp.fc1'],
            'inspect': block.mlp,
            'has_kwargs': False,
            'is_mlp': True
        })

        return subsets

    def get_layernorms_in_block(self, block):
        """多模态归一化层识别（修正版）"""
        if self._is_vision_block(block):
            # 视觉编码器的LayerNorm
            return {
                'layer_norm1': block.layer_norm1,
                'layer_norm2': block.layer_norm2
            }
        else:
            # 文本解码器的RMSNorm
            return {
                'input_layernorm': block.input_layernorm,
                'post_attention_layernorm': block.post_attention_layernorm
            }

    def _text_subsets(self, block):
        """文本解码器量化子集构造"""
        subsets = []

        # 注意力层
        subsets.extend([
            {
                'layers': {
                    'self_attn.q_proj': block.self_attn.q_proj,
                    'self_attn.k_proj': block.self_attn.k_proj,
                    'self_attn.v_proj': block.self_attn.v_proj,
                },
                'prev_op': [block.input_layernorm],
                'input': ['self_attn.q_proj'],
                'inspect': block.self_attn,
                'has_kwargs': True
            },
            {
                'layers': {'self_attn.o_proj': block.self_attn.o_proj},
                'prev_op': [block.self_attn.v_proj],
                'input': ['self_attn.o_proj'],
                'inspect': block.self_attn.o_proj,
                'has_kwargs': False
            }
        ])

        # MLP层
        subsets.extend([
            {
                'layers': {
                    'mlp.gate_proj': block.mlp.gate_proj,
                    'mlp.up_proj': block.mlp.up_proj,
                },
                'prev_op': [block.post_attention_layernorm],
                'input': ['mlp.gate_proj'],
                'inspect': block.mlp,
                'has_kwargs': False,
                'is_mlp': True
            },
            {
                'layers': {'mlp.down_proj': block.mlp.down_proj},
                'prev_op': [block.mlp.up_proj],
                'input': ['mlp.down_proj'],
                'inspect': block.mlp.down_proj,
                'has_kwargs': False,
                'is_mlp': True
            }
        ])

        return subsets

    def move_embed_to_device(self, device):
        """统一管理多模态组件设备"""
        # 视觉组件
        self.model.model.vision_model.to(device)
        # 文本组件
        self.model.model.text_model.to(device)
        # 连接器
        self.model.model.connector.to(device)
        # 输出头
        self.model.lm_head.to(device)

        # 确保嵌入层在目标设备
        for embed in self.vision_embeds + self.text_embeds:
            embed.to(device)
    # ----------------- 分布式训练支持 -----------------
    def parallel_quantize(self, dataset, **kwargs):
        """并行量化多模态组件"""
        # 量化视觉部分
        self._quant_modality('vision', dataset, **kwargs)

        # 量化文本部分
        self._quant_modality('text', dataset, **kwargs)

        # 量化连接器
        self._quant_connector(**kwargs)

    def _quant_modality(self, modality, dataset, **kwargs):
        """量化单个模态"""
        # 实现细节根据实际量化工具调整
        pass

    def _quant_connector(self, **kwargs):
        """量化连接器投影"""
        connector_layer = self.model.connector.modality_projection.proj
        # 实现量化逻辑
