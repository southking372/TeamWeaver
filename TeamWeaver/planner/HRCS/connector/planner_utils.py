from typing import Dict, List, Any, Optional, Union, TYPE_CHECKING
import json
import numpy as np

if TYPE_CHECKING:
    from habitat_llm.planner.centralized_llm_planner import ScenarioConfigTask

def extract_json_from_text(text: str, target_type: type = dict) -> Optional[str]:
    """从可能包含额外文本的LLM响应中提取JSON片段。"""
    if target_type == dict:
        start_char, end_char = '{', '}'
    elif target_type == list:
        start_char, end_char = '[', ']'
    else:
        return None

    try:
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start != -1 and end > start:
            return text[start:end+1]
    except Exception:
        pass
    return None

def get_llm_config() -> Dict[str, Any]:
    """获取LLM配置，模拟llm_planner中的配置格式"""
    return {
        'system_tag': '[SYSTEM]',
        'user_tag': '[USER]', 
        'assistant_tag': '[ASSISTANT]',
        'eot_tag': '[EOT]'
    }

def update_param_value(scenario_config: Union[Dict[str, Any], "ScenarioConfigTask"], key: str, value: Any) -> None:
    """
    更新参数值，同时支持字典类型和ScenarioConfigTask类型。
    """
    if isinstance(scenario_config, dict):
        scenario_config[key] = value
    else:
        # 对于ScenarioConfigTask实例，需要更新scenario_params字典
        try:
            if hasattr(scenario_config, 'scenario_params') and key in ['A', 'T', 'ws', 'Hs']:
                # 直接更新scenario_params中的矩阵
                scenario_config.scenario_params[key] = value
                print(f"DEBUG: Updated scenario_params['{key}'] in ScenarioConfigTask")
            else:
                # 尝试使用全局任务变量更新方法
                scenario_config.update_global_task_var(key, value)
        except AttributeError:
            print(f"Error: scenario_config 既不是字典也没有适当的更新方法")

def find_target_position(target_name: str, world_state: Dict[str, Any]) -> Optional[List[float]]:
    """查找目标在世界状态中的位置"""
    if not target_name:
        return None
        
    # 首先在家具中查找
    furniture_pos = world_state.get('furniture_positions', {}).get(target_name)
    if furniture_pos and 'position' in furniture_pos:
        return furniture_pos['position']
    
    # 然后在物体中查找
    object_pos = world_state.get('object_positions', {}).get(target_name)
    if object_pos and 'position' in object_pos:
        return object_pos['position']
    
    return None
