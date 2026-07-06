# SYSTEMS AND METHODS FOR TEAMWEAVER
# Copyright © 2025 HKUST(GZ).
# Developed by Yapeng Liu and SIIE Lab.
# HKUST(GZ) SIIE Lab Reference Number XXXX.
#
# Licensed under the Non-Commercial Open Source Software License.
# You may not use this file except in compliance with the License.
# A copy of the License is included in the root of this repository.

from typing import List, Dict, Any, Tuple
from collections import deque, defaultdict

class TaskDependencyEnhancer:
    """
Based on task semantics and topology, the original sub-task list is converted into a phased execution plan with dependencies.
This class removes all relevant dependency enhancement and task organization logic fromPerceptionConnectordecoupled from it.
    """

    def structure_and_phase(
        self,
        structured_subtasks: List[Dict[str, Any]],
        max_agents: int
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, List[str]]]:
        """
The main entrance method receives the original sub-task list and returns the enhanced and staged task plan.

        Args:
            structured_subtasks:fromLLMGet the original subtask list.
            max_agents:Maximum number of agents.

        Returns:
            A tuple containing:
            - enhanced_subtasks:Complete task list after semantic enhancement and dependency cleanup.
            - execution_phases:A task plan organized by stages that can be executed in parallel.
            - task_dependency_graph:Optimized task dependency graph.
        """
        #1. First perform semantic enhancement and add implicit dependencies(For examplePick -> Navigate -> Place)
        semantically_enhanced_tasks = self._enhance_semantic_dependencies(structured_subtasks)

        #2. Build and optimize task dependency graph
        task_dependency_graph = self._build_dependency_graph(semantically_enhanced_tasks)

        #Prerequisites for updating tasks based on optimized graphs
        task_dict = {task['task_id']: task for task in semantically_enhanced_tasks}
        for task_id, prereqs in task_dependency_graph.items():
            if task_id in task_dict:
                task_dict[task_id]['prerequisites'] = prereqs
        
        updated_tasks = list(task_dict.values())
        
        #3. Build a dependency graph and organize it into execution phases
        execution_phases = self._organize_tasks_into_phases(updated_tasks, max_agents)

        #4. Extract a complete, organized task list from the final stage
        final_task_list = [task for phase in execution_phases for task in phase['tasks']]

        return final_task_list, execution_phases, task_dependency_graph

    def _enhance_semantic_dependencies(self, structured_subtasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhance dependencies based on task semantics, especially handling compound actions (such asPick-Navigate-Place）。"""
        enhanced_tasks = [task.copy() for task in structured_subtasks]
        
        pick_tasks: List[Tuple[int, Dict[str, Any]]] = []
        navigate_tasks: List[Tuple[int, Dict[str, Any]]] = []
        place_tasks: List[Tuple[int, Dict[str, Any]]] = []
        
        for i, task in enumerate(enhanced_tasks):
            task_type = task.get('task_type', '')
            if task_type == 'Pick': pick_tasks.append((i, task))
            elif task_type == 'Navigate': navigate_tasks.append((i, task))
            elif task_type == 'Place': place_tasks.append((i, task))
        
        for nav_idx, nav_task in navigate_tasks:
            if self._is_carrying_navigation_task(nav_task, pick_tasks, place_tasks):
                required_picks = self._find_required_pick_tasks(nav_task, pick_tasks)
                current_prereqs = set(nav_task.get('prerequisites', []))
                for _, pick_task in required_picks:
                    current_prereqs.add(pick_task['task_id'])
                enhanced_tasks[nav_idx]['prerequisites'] = list(current_prereqs)
                enhanced_tasks[nav_idx]['is_carrying_navigation'] = True
        
        for place_idx, place_task in place_tasks:
            required_picks = self._find_pick_tasks_for_place(place_task, pick_tasks)
            required_navs = self._find_carrying_navigation_for_place(place_task, navigate_tasks)
            current_prereqs = set(place_task.get('prerequisites', []))
            for _, pick_task in required_picks:
                current_prereqs.add(pick_task['task_id'])
            for _, nav_task in required_navs:
                current_prereqs.add(nav_task['task_id'])
            enhanced_tasks[place_idx]['prerequisites'] = list(current_prereqs)
        
        self._validate_enhanced_dependencies(enhanced_tasks)
        return enhanced_tasks

    def _organize_tasks_into_phases(
        self,
        structured_subtasks: List[Dict[str, Any]],
        max_agents: int
    ) -> List[Dict[str, Any]]:
        """Organize tasks into phases that can be executed in parallel"""
        sorted_tasks = self._topological_sort_tasks(structured_subtasks)
        
        phases = []
        remaining_tasks = sorted_tasks.copy()
        completed_task_ids = set()
        safety_counter = 0
        max_iterations = len(sorted_tasks) * 2

        while remaining_tasks and safety_counter < max_iterations:
            safety_counter += 1
            current_phase_tasks = []
            
            #Find all currently executable tasks
            runnable_tasks = [
                task for task in remaining_tasks
                if all(prereq in completed_task_ids for prereq in task.get('prerequisites', []))
            ]

            #Sort by parallelism capabilities and priority
            runnable_tasks.sort(key=lambda t: (not t.get('can_parallel', False), -t.get('priority', 3)))

            #Populate current stage
            for task in runnable_tasks:
                if len(current_phase_tasks) >= max_agents:
                    break
                current_phase_tasks.append(task)
                if not task.get('can_parallel', False):
                    break
            
            if not current_phase_tasks and remaining_tasks:
                current_phase_tasks.append(remaining_tasks[0])

            #update status
            for task in current_phase_tasks:
                remaining_tasks.remove(task)
                completed_task_ids.add(task['task_id'])
            
            if current_phase_tasks:
                phases.append({
                    'phase_id': len(phases),
                    'tasks': current_phase_tasks,
                    'max_parallel_tasks': len(current_phase_tasks),
                    'estimated_duration': max((task.get('estimated_duration', 5.0) for task in current_phase_tasks), default=5.0),
                    'required_agents': min(len(current_phase_tasks), max_agents)
                })

        if remaining_tasks:
            if phases:
                phases[-1]['tasks'].extend(remaining_tasks)
            else:
                phases.append({
                    'phase_id': 0, 'tasks': remaining_tasks, 'max_parallel_tasks': len(remaining_tasks),
                    'estimated_duration': max((task.get('estimated_duration', 5.0) for task in remaining_tasks), default=5.0),
                    'required_agents': min(len(remaining_tasks), max_agents)
                })
        return phases

    def _topological_sort_tasks(self, structured_subtasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Topologically sort tasks and automatically handle circular dependencies"""
        cleaned_tasks = self._clean_task_dependencies(structured_subtasks)
        task_dict = {task['task_id']: task for task in cleaned_tasks}
        
        in_degree = {task_id: 0 for task_id in task_dict}
        graph = {task_id: [] for task_id in task_dict}
        
        for task_id, task in task_dict.items():
            for prereq in task.get('prerequisites', []):
                if prereq in graph:
                    graph[prereq].append(task_id)
                    in_degree[task_id] += 1
        
        queue = deque([task_id for task_id, degree in in_degree.items() if degree == 0])
        sorted_task_ids = []
        
        while queue:
            current = queue.popleft()
            sorted_task_ids.append(current)
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        if len(sorted_task_ids) != len(task_dict):
            #Handle circular dependencies
            remaining_ids = set(task_dict.keys()) - set(sorted_task_ids)
            remaining_tasks = [task_dict[task_id] for task_id in remaining_ids]
            remaining_tasks.sort(key=lambda x: x.get('priority', 3), reverse=True)
            sorted_task_ids.extend([task['task_id'] for task in remaining_tasks])
            
        return [task_dict[task_id] for task_id in sorted_task_ids]

    def _build_dependency_graph(self, structured_subtasks: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Build task dependency graphs and detect and fix circular dependencies"""
        dependency_graph = {}
        task_ids = {task['task_id'] for task in structured_subtasks}
        
        for task in structured_subtasks:
            task_id = task['task_id']
            prerequisites = task.get('prerequisites', [])
            
            cleaned_prerequisites = [prereq for prereq in prerequisites if prereq != task_id and prereq in task_ids]
            dependency_graph[task_id] = cleaned_prerequisites
        
        cycles = self._detect_dependency_cycles(dependency_graph)
        if cycles:
            print(f"WARNING: Detected {len(cycles)} dependency cycles:")
            for i, cycle in enumerate(cycles):
                print(f"  Cycle {i+1}: {' -> '.join(cycle + [cycle[0]])}")
            
            dependency_graph = self._break_dependency_cycles(dependency_graph, cycles)
            
        return dependency_graph

    def _detect_dependency_cycles(self, dependency_graph: Dict[str, List[str]]) -> List[List[str]]:
        """Detecting cycles in dependency graphs"""
        cycles = []
        visited = set()
        rec_stack = set()
        
        def dfs(node, path):
            if node in rec_stack:
                try:
                    cycle_start_index = path.index(node)
                    cycles.append(path[cycle_start_index:])
                except ValueError:
                    pass
                return
            if node in visited:
                return

            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in dependency_graph.get(node, []):
                if neighbor in dependency_graph:
                    dfs(neighbor, path + [neighbor])
            
            rec_stack.remove(node)

        for task_id in dependency_graph:
            if task_id not in visited:
                dfs(task_id, [task_id])
        
        #Remove duplicates
        unique_cycles = []
        seen_cycles = set()
        for cycle in cycles:
            sorted_cycle = tuple(sorted(cycle))
            if sorted_cycle not in seen_cycles:
                unique_cycles.append(cycle)
                seen_cycles.add(sorted_cycle)
        
        return unique_cycles

    def _break_dependency_cycles(self, dependency_graph: Dict[str, List[str]], cycles: List[List[str]]) -> Dict[str, List[str]]:
        """Break a dependency loop by removing an edge in the loop"""
        cleaned_graph = {k: list(v) for k, v in dependency_graph.items()}
        for cycle in cycles:
            if len(cycle) < 2:
                continue

            #Try to remove the dependency from the last node to the first node
            u, v = cycle[-1], cycle[0]
            if v in cleaned_graph.get(u, []):
                cleaned_graph[u].remove(v)
                print(f"INFO: Broke cycle by removing dependency: {u} -> {v}")
        return cleaned_graph

    def _inject_navigate_tasks(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
As a safety net, check eachPick/OpenTask, if missing the precedingNavigate, then one is forced to be injected.
        """
        newly_added_tasks = []
        task_dict = {task['task_id']: task for task in tasks}
        
        #Build a dependency graph containing all parent-child relationships for easy search
        full_dependency_graph = {tid: set(p.get('prerequisites', [])) for tid, p in task_dict.items()}
        for _ in range(len(task_dict)): #Iterate to propagate dependencies
            for tid, prereqs in full_dependency_graph.items():
                for p in list(prereqs):
                    if p in full_dependency_graph:
                        full_dependency_graph[tid].update(full_dependency_graph[p])

        interaction_tasks = [t for t in tasks if t.get('task_type') in ['Pick', 'Open']]

        for i, task in enumerate(interaction_tasks):
            has_navigate_prereq = False
            #Check all direct and indirect prerequisites
            prereq_ids = full_dependency_graph.get(task['task_id'], set())
            
            for prereq_id in prereq_ids:
                if prereq_id in task_dict and task_dict[prereq_id].get('task_type') == 'Navigate':
                    #Check if the navigation target is related to the interaction target
                    nav_target = task_dict[prereq_id].get('target', '')
                    interaction_target = task.get('target', '')
                    if self._are_targets_related(nav_target, interaction_target):
                        has_navigate_prereq = True
                        break
            
            if not has_navigate_prereq:
                #Need to inject a newNavigateTask
                #hypothesistargetyes 'object_name', we need to navigate to its parent object
                #This information is usually inworld_stateHere, we can only make a reasonable guess
                #We assumePlaceThe target of the task is the container, andPickThe target of the task is the object
                #The navigation target should be the name of the parent container of this object, but we cannot get it directly here
                # LLMThis information should be provided during decomposition. Here we can only create a general navigation
                #For example, navigate to the name of the object itself and let the executor resolve it
                
                nav_target = task.get('target', '')
                nav_task_id = f"injected_nav_{i}_{task['task_id']}"
                
                new_nav_task = {
                    'task_id': nav_task_id,
                    'task_type': 'Navigate',
                    'target': nav_target,
                    'description': f"Auto-injected: Navigate to {nav_target} before {task['task_type']}.",
                    'priority': task.get('priority', 3) + 1, #Navigation should have higher priority
                    'estimated_duration': 10.0,
                    'prerequisites': [],
                    'can_parallel': False, #Normally navigation cannot be parallelized
                    'phase_group': task.get('phase_group', 'execution')
                }
                
                print(f"INFO: Injecting missing Navigate task for {task['task_type']}->{task['target']}")
                
                #Set the new navigation task as a prerequisite for the original task
                current_prereqs = task.get('prerequisites', [])
                current_prereqs.append(nav_task_id)
                task['prerequisites'] = current_prereqs
                
                newly_added_tasks.append(new_nav_task)

        return tasks + newly_added_tasks
        
    def _clean_task_dependencies(self, structured_subtasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Clean task dependencies; remove self-loops and invalid references"""
        task_ids = {task['task_id'] for task in structured_subtasks}
        cleaned_tasks = []
        for task in structured_subtasks:
            cleaned_task = task.copy()
            prereqs = task.get('prerequisites', [])
            cleaned_task['prerequisites'] = [p for p in prereqs if p in task_ids and p != task['task_id']]
            cleaned_tasks.append(cleaned_task)
        return cleaned_tasks

    def _is_carrying_navigation_task(self, nav_task: Dict[str, Any], pick_tasks: List, place_tasks: List) -> bool:
        """Determine whether it is a navigation task carrying items"""
        nav_target = nav_task.get('target', '').lower()
        nav_description = nav_task.get('description', '').lower()
        place_targets = {p_task.get('target', '').lower() for _, p_task in place_tasks}
        if nav_target in place_targets: return True
        
        carrying_keywords = ['carry', 'carrying', 'bring', 'take', 'transport', 'move', 'deliver']
        if any(keyword in nav_description for keyword in carrying_keywords): return True
        
        try:
            nav_order = int(nav_task.get('task_id', 'task_0').split('_')[1])
            has_pick_before = any(int(p_task.get('task_id', 'task_99').split('_')[1]) < nav_order for _, p_task in pick_tasks)
            has_place_after = any(int(p_task.get('task_id', 'task_0').split('_')[1]) > nav_order for _, p_task in place_tasks)
            return has_pick_before and has_place_after
        except (ValueError, IndexError):
            return False

    def _find_required_pick_tasks(self, nav_task: Dict[str, Any], pick_tasks: List) -> List[Tuple[int, Dict[str, Any]]]:
        """Find all prerequisites required for navigation tasksPickTask"""
        try:
            nav_order = int(nav_task.get('task_id', 'task_0').split('_')[1])
            return [(idx, p_task) for idx, p_task in pick_tasks if int(p_task.get('task_id', 'task_99').split('_')[1]) < nav_order]
        except (ValueError, IndexError):
            return []

    def _find_pick_tasks_for_place(self, place_task: Dict[str, Any], pick_tasks: List) -> List[Tuple[int, Dict[str, Any]]]:
        """turn upPlacecorresponding to the taskPickTask"""
        desc = place_task.get('description', '').lower()
        return [(idx, p_task) for idx, p_task in pick_tasks if p_task.get('target', '').lower() in desc]

    def _find_carrying_navigation_for_place(self, place_task: Dict[str, Any], navigate_tasks: List) -> List[Tuple[int, Dict[str, Any]]]:
        """turn upPlaceThe carrying navigation task corresponding to the task"""
        place_target = place_task.get('target', '').lower()
        related_navs = []
        for nav_idx, nav_task in navigate_tasks:
            if nav_task.get('is_carrying_navigation') and self._are_targets_related(nav_task.get('target', '').lower(), place_target):
                related_navs.append((nav_idx, nav_task))
        return related_navs

    def _are_targets_related(self, target1: str, target2: str) -> bool:
        """Determine whether two goals are related"""
        if not target1 or not target2: return False
        if target1 == target2: return True
        words1 = set(target1.replace('_', ' ').split())
        words2 = set(target2.replace('_', ' ').split())
        return len(words1.intersection(words2)) > 0

    def _validate_enhanced_dependencies(self, enhanced_tasks: List[Dict[str, Any]]) -> None:
        """Verify enhanced dependencies to prevent circular dependencies"""
        task_ids = {task['task_id'] for task in enhanced_tasks}
        for task in enhanced_tasks:
            prereqs = task.get('prerequisites', [])
            task['prerequisites'] = [p for p in prereqs if p in task_ids and p != task['task_id']] 
