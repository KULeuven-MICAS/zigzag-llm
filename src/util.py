from enum import StrEnum
from typing import TypeAlias
from typing import Any, TypeVar
import numpy as np
import pickle

from src.config import LLMConfig, QuantConfig


LAYERS_TO_PLOT = ["key_proj", "mul_qk_t", "mul_logits", "feedforward_expand", "feedforward_contract"]
GROUPS = ["Linear proj.", "Attention", "FFN"]

CME_T = TypeVar("CME_T", Any, Any)  # CME type not available here
ARRAY_T: TypeAlias = np.ndarray[Any, Any]


class Stage(StrEnum):
    PREFILL = "prefill"
    DECODE = "decode"


def generalize_layer_name(layer: str):
    """Give the layer name a prettier format, and generalize single layers to full LLM. e.g. key projection -> all
    linear projections"""
    if "key_proj" in layer:
        return "linear projection"
    elif "mul_qk_t" in layer:
        return "mul K*Q^T"
    elif "mul_logits" in layer:
        return "mul attn*V"
    elif "feedforward_expand" in layer:
        return "MLP layer 1"
    elif "feedforward_contract" in layer:
        return "MLP layer 2"
    else:
        return layer


def get_cmes_to_plot(cmes: list[CME_T]):
    """Return CMEs in order of `LAYERS_TO_PLOT"""
    result: list[CME_T] = []
    for name in LAYERS_TO_PLOT:
        cme = next(filter(lambda x: name in x.layer.name, cmes), None)
        if cme is not None:
            result.append(cme)
    return result


def get_cmes_full_model(cmes: list[CME_T], model: LLMConfig, stage: Stage = Stage.PREFILL):
    """Generalize the zigzag results (for single layers) to a full LLM
    @param prefill: whether the results are from a prefill or decode phase simulation"""
    assert len(cmes) == 5, "These are not the `LAYERS_TO_PLOT`"
    number_of_runs = 1 if stage == Stage.PREFILL else model.decode_size
    return [cme * model.get_post_simulation_multiplier(cme.layer.name) * number_of_runs for cme in cmes]


def get_cmes_full_model_from_pickle(pickle_file: str, model: LLMConfig, stage: Stage) -> list[CME_T]:
    with open(pickle_file, "rb") as fp:
        cmes: list[CME_T] = pickle.load(fp)

    cmes = get_cmes_to_plot(cmes)
    cmes = get_cmes_full_model(cmes, model, stage)
    return cmes


def get_experiment_id(model: LLMConfig, stage: Stage, quant: QuantConfig, accelerator_name: str):
    """Generate the name of the experiment"""
    return f"{model.parameterized_name}_{quant.name}_{stage}_{accelerator_name}"


def get_onnx_path(model: LLMConfig, stage: Stage, quant: QuantConfig):
    ONNX_DIR = "outputs/onnx"
    return f"{ONNX_DIR}/{model.parameterized_name}_{quant.name}_{stage}.onnx"


def get_accelerator_path(accelerator: str):
    return f"inputs/hardware/{accelerator}.yaml"
