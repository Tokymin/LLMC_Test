from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel, SmolVLMConfig, \
    SmolVLMForConditionalGeneration
from transformers.utils import logging

logger = logging.get_logger(__name__)

# 注册配置类
AutoConfig.register("smolvlm", SmolVLMConfig)


# 自定义模型加载器
class SmolVLMAutoModelForCausalLM(AutoModelForCausalLM):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # 强制指定配置类
        config = kwargs.pop("config", None) or AutoConfig.from_pretrained(pretrained_model_name_or_path)

        if config.model_type == "smolvlm":
            return SmolVLMForConditionalGeneration.from_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                config=config,
                **kwargs
            )

        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)


# 注册到Auto系统
AutoModelForCausalLM.register(SmolVLMConfig, SmolVLMForConditionalGeneration)
