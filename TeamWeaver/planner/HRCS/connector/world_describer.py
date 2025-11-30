from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from habitat_llm.agent.env import EnvironmentInterface

class WorldDescriber:
    """
    Generates detailed world descriptions for LLM prompts.
    Decouples the logic of querying the WorldGraph and formatting it into
    human-readable text from the PerceptionConnector.
    """
    def __init__(self):
        pass

    def get_world_description(self, env_interface: "EnvironmentInterface", world_state: dict, is_partial_obs: bool = False) -> str:
        """
        Generates a detailed world description string for the LLM prompt,
        providing complete layout information and dynamic agent status.
        """
        # === 1. Generate static environment layout description ===
        layout_description = self._get_layout_description(env_interface, is_partial_obs)
        
        # === 2. Generate dynamic agent status description ===
        agent_description = self._get_agent_status_description(world_state)
        
        # === 3. Combine into a single, comprehensive description ===
        return f"{layout_description}\n\n{agent_description}"

    def _get_layout_description(self, env_interface: "EnvironmentInterface", is_partial_obs: bool = False) -> str:
        """
        Generates a detailed world layout description string for the LLM prompt,
        providing complete layout information even in partial observation mode.
        """
        # === 1. Always use full graph for static layout information (rooms & furniture) ===
        full_graph = env_interface.full_world_graph
        all_rooms = full_graph.get_all_rooms()
        all_furniture = full_graph.get_all_furnitures()
        
        # === 2. For objects, respect the observation mode ===
        if is_partial_obs:
            observed_objects = []
            # **FIX**: Use explicit 'is not None' and 'isinstance' checks to avoid implicit len() call on DynamicWorldGraph
            if (env_interface.world_graph is not None and 
                isinstance(env_interface.world_graph, dict) and
                any(g is not None and len(g) > 0 for g in env_interface.world_graph.values())):
                # Collect observed objects from all agent perspectives
                observed_objects = list({
                    node.name: node 
                    for ag in env_interface.world_graph.values() 
                    if ag is not None
                    for node in ag.get_all_objects()
                }.values())
        else:
            # Full observability - use all objects
            observed_objects = full_graph.get_all_objects()

        # === 3. Build comprehensive description ===
        
        # Start with layout information (always complete)
        description_lines = [
            "Environment Layout Information:",
            "The following rooms and furniture are known in this house:"
        ]
        
        # Build room-furniture mapping
        room_contents = {room.name: [] for room in all_rooms}
        unplaced_furniture = []

        for furniture in all_furniture:
            parent_room = full_graph.find_room_for_furniture(furniture)
            if parent_room and parent_room.name in room_contents:
                room_contents[parent_room.name].append(furniture.name)
            else:
                unplaced_furniture.append(furniture.name)

        # Add room and furniture details
        if not room_contents:
            description_lines.append("- No room layout information available.")
        else:
            for room_name, furniture_list in sorted(room_contents.items()):
                if furniture_list:
                    description_lines.append(f"- {room_name}: {', '.join(sorted(furniture_list))}")
                else:
                    description_lines.append(f"- {room_name}: (no furniture)")
        
        if unplaced_furniture:
            description_lines.append(f"- Additional furniture: {', '.join(sorted(unplaced_furniture))}")
        
        description_lines.append("")  # Separator
        
        # Add object information with observation status
        if is_partial_obs:
            if not observed_objects:
                description_lines.extend([
                    "Object Information (Partial Observation):",
                    "- No objects have been observed yet.",
                    "- Objects must be found through exploration before they can be manipulated.",
                    "- Use 'Explore[room_name]' to discover objects in specific rooms."
                ])
            else:
                description_lines.extend([
                    "Object Information (Partial Observation):",
                    "The following objects have been observed (others may exist but are not yet known):"
                ])
                
                for obj in sorted(observed_objects, key=lambda o: o.name):
                    parent = full_graph.find_furniture_for_object(obj)
                    
                    # Add state information if available
                    state_info_parts = []
                    try:
                        if obj.get_property('powered') is not None:
                            state_info_parts.append('powered on' if obj.get_property('powered') else 'powered off')
                        if obj.get_property('is_clean') is not None:
                            state_info_parts.append('clean' if obj.get_property('is_clean') else 'dirty')
                    except (KeyError, AttributeError):
                        pass
                    
                    state_str = f" ({', '.join(state_info_parts)})" if state_info_parts else ""

                    if parent:
                        description_lines.append(f"- '{obj.name}'{state_str} is on/in '{parent.name}'")
                    else:
                        description_lines.append(f"- '{obj.name}'{state_str} location unknown")
        else:
            # Full observation mode
            description_lines.extend([
                "Object Information (Full Observation):",
                "All objects in the environment:"
            ])
            
            if not observed_objects:
                description_lines.append("- No objects exist in this environment.")
            else:
                for obj in sorted(observed_objects, key=lambda o: o.name):
                    parent = full_graph.find_furniture_for_object(obj)
                    
                    # Add state information
                    state_info_parts = []
                    try:
                        if obj.get_property('powered') is not None:
                            state_info_parts.append('powered on' if obj.get_property('powered') else 'powered off')
                        if obj.get_property('is_clean') is not None:
                            state_info_parts.append('clean' if obj.get_property('is_clean') else 'dirty')
                    except (KeyError, AttributeError):
                        pass
                    
                    state_str = f" ({', '.join(state_info_parts)})" if state_info_parts else ""

                    if parent:
                        description_lines.append(f"- '{obj.name}'{state_str} is on/in '{parent.name}'")
                    else:
                        description_lines.append(f"- '{obj.name}'{state_str} location unknown")

        return "\n".join(description_lines)

    def _get_agent_status_description(self, world_state: dict) -> str:
        """Formats the agent status from the world state for the LLM prompt."""
        if not world_state or 'agent_poses' not in world_state:
            return "Agent Status: Not available."
            
        agent_status_lines = ["Agent Status:"]
        agent_poses = world_state.get('agent_poses', {})
        
        # Determine which objects are held by which agent
        held_objects_by_agent = {agent_name: [] for agent_name in agent_poses}
        for obj_name, obj_info in world_state.get('object_positions', {}).items():
            if obj_info and obj_info.get('parent') in agent_poses:
                held_objects_by_agent[obj_info['parent']].append(obj_name)
        
        # Format the description for each agent
        for agent_name, pose_info in sorted(agent_poses.items()):
            pos_str = "position unknown"
            if pose_info and 'position' in pose_info:
                pos = pose_info['position']
                pos_str = f"is at [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]"
            
            held_items = held_objects_by_agent.get(agent_name, [])
            held_str = f"and holding: {', '.join(held_items)}" if held_items else "with hands free"
            
            agent_status_lines.append(f"- Agent '{agent_name}' {pos_str} {held_str}.")
        
        return "\n".join(agent_status_lines) if len(agent_status_lines) > 1 else "Agent Status: No agents found." 