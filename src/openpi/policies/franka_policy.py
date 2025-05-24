import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_franka_example() -> dict:
    """Creates a random input example for the Franka policy."""
    return {
        "joints": np.random.rand(7),
        "gripper": np.random.rand(1),
        "image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),  # 1. channel is the last dim
        "wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)  # 2. the image data is finally 255
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")  # 1. channel is the last dim
    return image


@dataclasses.dataclass(frozen=True)
class FrankaInputs(transforms.DataTransformFn):
    """
    该类被用于把输入转成model期待的格式. 该类被同时用于训练 和 推理.

    对于你自己的数据集,你可以复制这个类并修改其中的keys 基于下面的注释
    """

    # 不要修改 pi0 model (不是 pi0-fast) 的 action dimension
    action_dim: int

    # 决定使用哪一个 model
    # 别修改
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        # 只针对pi0_model 做掩码，不是 pi0-fast
        mask_padding = self.model_type == _model.ModelType.PI0

        # 把 机器人本体感知输入 pad 到 action_dim
        # 如果 本体感知输入不是 "observation/state"，修改下面的键
        state = np.concatenate(
            [data["joints"], np.array([data["gripper"]])]  # 将标量转为1维数组
        )
        state = transforms.pad_to_dim(state, self.action_dim)

        # 可能需要 parse images 到 uint8 (H,W,C) 因为 LeRobot 自动存储 float32 (C,H,W)
        # 在 inference 阶段跳过
        # 如果存储的key不是 "observation/image" 或者 "observation/wrist_image", 修改
        # PI0 喜欢 一个第三视角 和 两个腕部视角，如果腕部视角不齐就零填充
        base_image = _parse_image(data["base_rgb"])
        wrist_image = _parse_image(data["wrist_rgb"])

        # 创建模型输入字典，不要修改 keys 哦
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_ if mask_padding else np.True_,
            },
        }

        # 把 actions pad 到 model 的 action dimension
        # actions 只在 training 时 available
        if "actions" in data:
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class FrankaOutputs(transforms.DataTransformFn):
    """
    该类被用于 将 模型输出 转换成 数据集的特定格式。只被用于 inference
    """

    def __call__(self, data: dict) -> dict:
        # 只返回最初的 N actions, 7 joint + 1 gripper
        return {"actions": np.asarray(data["actions"][:, :8])}
