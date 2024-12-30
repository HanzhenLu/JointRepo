import os
import ast
import random
import pandas as pd
from typing import Dict, List, Union, Set, Tuple
from tqdm import tqdm
from collections import deque
from colorama import Back

from utils.util import label_line

def read_dir(dir:str) -> Dict[str, str]:
    content_map = {}
    for dirpath, _, filenames in os.walk(dir, topdown=False):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            try:
                with open(file_path, 'r') as f:
                    code = f.read()
                if code.strip() != "":
                    content_map[file_path] = code
            except:
                print(file_path)
            
    
    return content_map
    
def extract_imports(file_path:str, content: str) -> List:
    tree = ast.parse(content, filename=file_path)
    
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    
    return imports

class File:
    def __init__(self, path:str, content:str) -> None:
        self.path = path
        self.content = content
        self.dep_in = 0
        self.dep_out_path: set = set()
        self.dep_in_lib: set = set()
        self.dep_in_path: set = set()
        self.subgraph_idx: int = -1
    
    # 方便打印出来查看内容
    def __str__(self) -> str:
        return "path:{}\ndep_in:{}\ndep_out_path:{}".format(self.path, self.dep_in, str(self.dep_out_path))
    
    # 用于set
    def __hash__(self) -> int:
        return str(self).__hash__()

def TopologicalSort(project:Dict[str, str]) -> Union[List[List[File]], None]:
    graph: Dict[str, File] = {}
    for path, content in project.items():
        cur_file = File(path, content)
        try:
            imports = extract_imports(path, content)
        except Exception as e:
            # print(Back.RED + f"{path} : {e}")
            # print(Back.BLACK + "")
            return None
            
        for imp in imports:
            cur_file.dep_in_lib.add(imp)
        graph[path] = cur_file
    
    # 根据依赖关系连接图中的各个节点
    for key, value in graph.items():
        for imp in value.dep_in_lib:
            imp = imp.replace('.', '/')
            for file in graph.keys():
                if file.endswith(imp+"/__init__.py") or file.endswith(imp+".py"):
                    graph[file].dep_out_path.add(key)
                    graph[key].dep_in_path.add(file)
                    graph[key].dep_in += 1
    # 将整个图分为互不连通的子图
    subgraphs: List[Set[File]] = []
    for _, node in graph.items():
        if node.subgraph_idx == -1:
            subgraph = set()
            subgraph.add(node)
            idx = len(subgraphs)
            node.subgraph_idx = idx
            queue = deque()
            
            for file_path in node.dep_out_path:
                queue.append(file_path)
            for file_path in node.dep_in_path:
                queue.append(file_path)
            while queue:
                element = queue.popleft()
                element = graph[element]
                if element.subgraph_idx == -1:
                    subgraph.add(element)
                    element.subgraph_idx = idx
                    for file_path in node.dep_out_path:
                        queue.append(file_path)
                    for file_path in node.dep_in_path:
                        queue.append(file_path)
            subgraphs.append(subgraph)
            
    def find_min_in_node(subgraph: Set[File], results: List[File]) -> File:
        min_in = 0
        min_node = None
        for node in subgraph:
            if node not in results:
                if min_node is None or node.dep_in < min_in:
                    min_node = node
                    min_in = node.dep_in
        
        return min_node
    
    # 将每个子图转化为一个字符串
    all_results = []
    for subgraph in subgraphs:
        results = []
        while len(results) != len(subgraph):
            min_node = find_min_in_node(subgraph, results)
            for out_file in min_node.dep_out_path:
                graph[out_file].dep_in -= 1
            results.append(min_node)
        
        all_results.append(results)
        
    return all_results

def process_project(project_dict:Dict[str, str]) -> Union[List[List[File]], None]:
    res = TopologicalSort(project_dict)
    return res

def construct_data(example:List[Tuple[str, str]]) -> Tuple[str, str, str, str, List[Dict[str, str]]]:
    outer_try_count = 0
    while outer_try_count < min(10, len(example) - 1):
        # example:List[Tuple[str, str]]
        # the later tuple relies on the former tuple
        # so we shouldn't choose the first tuple
        selected_file = random.choice(example[1:])
        line_index_labels = label_line(selected_file[1])
        # if the selected file contains little content, we should drop it
        if len(line_index_labels) < 10:
            outer_try_count += 1
            continue
        raw_lines = selected_file[1].split('\n')
        raw_lines = [line+"\n" if i != len(raw_lines) - 1 else line for i, line in enumerate(raw_lines)]
        valid_index = [r for r, v in line_index_labels if v]
        
        if len(valid_index) == 0:
            outer_try_count += 1
            continue
        
        inner_try_count = 0
        
        selected_line = None
        while (selected_line is None or len(selected_line) > 200) and inner_try_count < 10:
            selected_index = random.choice(valid_index[1:]) if len(valid_index) > 1 else valid_index[0]
            selected_line = "".join([raw_lines[i] for i in selected_index])
            inner_try_count += 1
        
        if inner_try_count == 10:
            outer_try_count += 1
            continue
        else:
            break
    
    if outer_try_count == min(10, len(example) - 1):
        return None
    
    prefix = "".join([raw_lines[i] for i in range(0, selected_index[0])]) if selected_index[0] > 0 else ""
    suffix = "".join([raw_lines[i] for i in range(selected_index[-1] + 1, len(raw_lines))]) if selected_index[-1] < len(raw_lines) else ""
    if random.random() > 0.9:
        middle = selected_line
    else:
        split_point = random.randint(0, len(selected_line) - 2)
        prefix += selected_line[:split_point]
        middle = selected_line[split_point:]
    
    related_files = [{"path":x[0], "text":x[1]} for x in example if x[0] != selected_file[0]]
    
    return selected_file[0], prefix, suffix, middle, related_files

if __name__ == "__main__":
    root = "github_projects"
    repo_names = os.listdir(root)
    repo_dict = {}
    for repo_name in repo_names:
        repo_dict[repo_name] = read_dir(os.path.join(root, repo_name))

    samples = []
    for repo_name, file_content_map in tqdm(repo_dict.items()):
        # print(Back.GREEN + f"begin {repo_name}", end="")
        # print(Back.BLACK + "\n")
        result = process_project(file_content_map)
        if result is not None:
            samples.append(result)
        
    dataframe = {"task_id":[], "path":[], "left_context":[], "right_context":[], "crossfile_context":[], "groundtruth":[]}
    for repo_sample in samples:
        for cluster in repo_sample:
            cluster:List[File]
            example = []
            for file in cluster:
                example.append((file.path, file.content))
                
            item = construct_data(example)
            if item is not None:
                dataframe["task_id"].append(item[0])
                dataframe["path"].append(item[0])
                dataframe["left_context"].append(item[1])
                dataframe["right_context"].append(item[2])
                dataframe["crossfile_context"].append(item[4])
                dataframe["groundtruth"].append(item[3])
    
    dataframe = pd.DataFrame.from_dict(dataframe)
    if not os.path.exists("data/github_projects/python"):
        os.makedirs("data/github_projects/python")
    dataframe.to_parquet("data/github_projects/python/train.parquet")