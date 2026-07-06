from typing import Dict, List, Any, Optional, Union, TYPE_CHECKING
import json
import numpy as np

if TYPE_CHECKING:
    from habitat_llm.planner.centralized_llm_planner import ScenarioConfigTask

def extract_json_from_text(text: str, target_type: type = dict) -> Optional[str]:
    """from which may contain additional textLLMExtract from responseJSONfragment."""
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
    """getLLMconfiguration, simulationllm_plannerConfiguration format in"""
    return {
        'system_tag': '[SYSTEM]',
        'user_tag': '[USER]', 
        'assistant_tag': '[ASSISTANT]',
        'eot_tag': '[EOT]'
    }

def update_param_value(scenario_config: Union[Dict[str, Any], "ScenarioConfigTask"], key: str, value: Any) -> None:
    """
Update parameter values, supporting both dictionary type andScenarioConfigTasktype.
    """
    if isinstance(scenario_config, dict):
        scenario_config[key] = value
    else:
        #forScenarioConfigTaskInstance needs to be updatedscenario_paramsdictionary
        try:
            if hasattr(scenario_config, 'scenario_params') and key in ['A', 'T', 'ws', 'Hs']:
                #direct updatescenario_paramsmatrix in
                scenario_config.scenario_params[key] = value
                print(f"DEBUG: Updated scenario_params['{key}'] in ScenarioConfigTask")
            else:
                #Try using global task variable update method
                scenario_config.update_global_task_var(key, value)
        except AttributeError:
            print(f"Error: scenario_configNeither a dictionary nor a proper update method")

def find_target_position(target_name: str, world_state: Dict[str, Any]) -> Optional[List[float]]:
    """Find target in world stateposition"""
    if not target_name:
        return None
        
    #Find it first in Furniture
    furniture_pos = world_state.get('furniture_positions', {}).get(target_name)
    if furniture_pos and 'position' in furniture_pos:
        return furniture_pos['position']
    
    #Then search in the object
    object_pos = world_state.get('object_positions', {}).get(target_name)
    if object_pos and 'position' in object_pos:
        return object_pos['position']
    
    return None
