import torch
from llmc.utils.registry_factory import MODEL_REGISTRY
from .base_model import BaseModel
from transformers import AutoConfig, SmolVLMForConditionalGeneration
from loguru import logger
from accelerate import Accelerator, DistributedType
from typing import Optional, Union
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
        # smol_VLMForConditionalGeneration=self.model
        # self.model=self.model.model
        # 修正lm_head数据类型
        if self.model.lm_head.weight.dtype != torch.bfloat16:
            self.model.lm_head = self.model.lm_head.to(torch.bfloat16)
        logger.info(f"lm_head dtype: {self.model.lm_head.weight.dtype}")
        # 初始化组件引用
        self.vision_model = self.model.model.vision_model
        self.text_model = self.model.model.text_model
        self.connector = self.model.model.connector
        # 验证类型一致性
        text_emb = self.text_model.embed_tokens
        assert text_emb.weight.dtype == torch.bfloat16, "文本嵌入层类型错误"
        assert self.model.lm_head.weight.dtype == torch.bfloat16, "输出头类型错误"


        # 统一设备初始化
        # self._sync_device()
    def find_blocks(self):
        # 文本模型的块（LlamaDecoderLayer）作为主要处理块
        self.blocks = self.text_model.layers
        # 视觉模型的块单独存储（可选，根据需求）
        self.vision_blocks = self.vision_model.encoder.layers

    def find_embed_layers(self):
        # 视觉嵌入层：patch embedding（ Conv2d）和位置嵌入（Embedding）
        self.vision_patch_embed = self.vision_model.embeddings.patch_embedding
        self.vision_pos_embed = self.vision_model.embeddings.position_embedding
        # 文本嵌入层
        self.text_embed_tokens = self.text_model.embed_tokens

    def get_embed_layers(self):
        # 返回所有嵌入层（视觉和文本）
        return [self.vision_patch_embed, self.vision_pos_embed, self.text_embed_tokens]

    def get_head_layers(self):
        # 生成头
        return [self.model.lm_head]

    def get_pre_head_layernorm_layers(self):
        # 文本模型的最终层归一化
        return [self.text_model.norm]

    def get_layers_except_blocks(self):
        # 除块外的层：视觉嵌入、视觉后归一化、文本嵌入、文本最终归一化、生成头
        return [
            self.vision_patch_embed,
            self.vision_pos_embed,
            self.vision_model.post_layernorm,
            self.text_embed_tokens,
            self.text_model.norm,
            self.model.lm_head
        ]

    def skip_layer_name(self):
        # 跳过生成头（与原始LLaMA逻辑一致）
        return ['lm_head']

    def has_bias(self):
        # 视觉模块的线性层有偏置（q_proj/k_proj/v_proj/out_proj均bias=True），文本模块无偏置
        return True

    def get_layernorms_in_block(self, block):
        # 处理文本块的层归一化（与LLaMA一致）
        return {
            'input_layernorm': block.input_layernorm,
            'post_attention_layernorm': block.post_attention_layernorm,
        }

    def get_subsets_in_block(self, block):
        # 文本块的子集结构（与LLaMA一致）
        return [
            {
                'layers': {
                    'self_attn.q_proj': block.self_attn.q_proj,
                    'self_attn.k_proj': block.self_attn.k_proj,
                    'self_attn.v_proj': block.self_attn.v_proj,
                },
                'prev_op': [block.input_layernorm],
                'input': ['self_attn.q_proj'],
                'inspect': block.self_attn,
                'has_kwargs': True,
            },
            {
                'layers': {'self_attn.o_proj': block.self_attn.o_proj},
                'prev_op': [block.self_attn.v_proj],
                'input': ['self_attn.o_proj'],
                'inspect': block.self_attn.o_proj,
                'has_kwargs': False,
            },
            {
                'layers': {
                    'mlp.gate_proj': block.mlp.gate_proj,
                    'mlp.up_proj': block.mlp.up_proj,
                },
                'prev_op': [block.post_attention_layernorm],
                'input': ['mlp.gate_proj'],
                'inspect': block.mlp,
                'has_kwargs': False,
                'is_mlp': True,
            },
            {
                'layers': {'mlp.down_proj': block.mlp.down_proj},
                'prev_op': [block.mlp.up_proj],
                'input': ['mlp.down_proj'],
                'inspect': block.mlp.down_proj,
                'has_kwargs': False,
                'is_mlp': True,
            },
        ]

    # 以下为可选扩展（若需处理视觉块，可添加额外方法，但BaseModel未强制要求）
    def find_block_name(self):
        # 定义文本块的命名前缀（与LLaMA一致）
        self.block_name_prefix = 'text_model.layers'
        self.pairs = {'q_proj': 'qkv', 'o_proj': 'out', 'up_proj': 'fc1'}
    # 保持与BaseModel接口兼容的其他方法（如需可补充视觉处理逻辑）
