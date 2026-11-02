import re
import argparse
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import os
import json
import shutil
from itertools import combinations
from typing import Tuple, Dict

def parse_arguments() -> dict:

    parser = argparse.ArgumentParser(description='Gets ''simion'' clusters of solutions based on their similarity to the original snippet.')
    
    parser.add_argument(
        "--level", type=str, help="The level of the task (easy, medium, hard)", required=True
    )

    parser.add_argument(
        '--leetcode_description', type=str, help='Prompt to send to the model', required=True
    )
    parser.add_argument(
        '--temperature', help='Temperature for the response generation', required=True
    )
    parser.add_argument(
        '--task', type=str, help='Task to perform (python2java, java2python, python, etc)', required=True  
    )
    parser.add_argument(
        '--sampling', type=str, help='Sampling method (twenty queries vs one query)', required=True
    )
    parser.add_argument(
        '--model', type=str, help='Model to use for the response generation', required=True
    )
    parser.add_argument(
        '--no_exceptions', action="store_true", help='Remove behavior where the output is an error', required=False, default=False
    )
    args = parser.parse_args()

    return args

def collect_solution_behavior(directory: str, cpp_behavior: dict, java_behavior: dict, python_behavior: dict, task: str, sampling: str) -> dict:
    solutions_dir = os.path.join(directory, 'preprocessed')
    all_solution_behaviors = {}

    try:
        for solution in os.listdir(solutions_dir):
            solution_dir = os.path.join(solutions_dir, solution)

            # --- Get the solution's behavior --- #
            try:
                with open(f'{solution_dir}/behavior.json', 'r') as file:
                    solution_behavior = json.load(file)
                    if solution_behavior is None or solution_behavior == {}:
                        logging.info(f"Solution {solution} is empty.")
                        continue
            except:
                logging.info(f"Solution {solution} has no behavior.")
                continue

            all_solution_behaviors.update({f'{solution}': solution_behavior})
    
    except Exception as e:
        logging.error(f"no preprocessed directory for {task}/{sampling}")
        all_solution_behaviors = {}
    # Add the reference solutions to the behavior dictionary if the dictionary is not empty
    if (task == "java" or task == "java_regular"):
        all_solution_behaviors.update({'java_reference_solution': java_behavior})
    elif task == "cppToJava":
        all_solution_behaviors.update({'cpp_reference_solution': cpp_behavior})
    elif task == "pythonToJava":
        all_solution_behaviors.update({'python_reference_solution': python_behavior})

    return all_solution_behaviors
# return three dictionaries: cpp, java, python
def filter_out_exceptions_output(solutions_behavior: dict) -> dict:
    filtered_solutions_behavior = {}        
    cpp_error_signatures = [
        "error:",
        "invalid pointer",
        "segmentation fault",
        "corrupted",
        "assertion failed",
        "double free or corruption",
        "malloc():",
        "free():",
        "munmap_chunk():",
        "error",
        "exception",
        "terminate called",
        "timeout"
        ]
    java_error_signatures = [
        "exception in thread",
        "java.lang.",
        "at ",
        "caused by: ",
        "error",
        "timeout"
    ]
    python_error_signatures = [
        "traceback (most recent call last):",
        "exception",
        "error",
        "timeout"
    ]

    for solution, behavior in solutions_behavior.items():
        filtered_solutions_behavior[solution] = {}
        for input, output in behavior.items():

            if "cpp_reference_solution" in solutions_behavior:
                if not any(error in output.lower() for error in cpp_error_signatures):
                    filtered_solutions_behavior[solution].update({input: output})
                        
            elif "java_reference_solution" in solutions_behavior:
                if not any(error in output.lower() for error in java_error_signatures):
                    filtered_solutions_behavior[solution].update({input: output})
                        
            elif "python_reference_solution" in solutions_behavior:
                if not any(error in output.lower() for error in python_error_signatures):
                    filtered_solutions_behavior[solution].update({input: output})
                        
    return filtered_solutions_behavior
            
class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}
    
    def find(self, item):
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]
    
    def union(self, set1, set2):
        root1 = self.find(set1)
        root2 = self.find(set2)
        if root1 != root2:
            if self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            else:
                self.parent[root1] = root2
                if self.rank[root1] == self.rank[root2]:
                    self.rank[root2] += 1
    
    def make_set(self, item):
        self.parent[item] = item
        self.rank[item] = 0

def calculate_similarity(solution_a, solution_b):
    common_keys = set(solution_a.keys()) & set(solution_b.keys())
    if not common_keys:
        return 1 if solution_a == {} and solution_b == {} else 0
    
    matching_pairs = sum(1 for key in common_keys if solution_a[key] == solution_b[key])
    total_comparable_pairs = len(common_keys)
    
    return matching_pairs / total_comparable_pairs

def cluster_solutions(solutions_behavior: dict, similarity_threshold: float) -> dict:
    uf = UnionFind()
    
    # Initialize union-find structure for each solution
    for solution in solutions_behavior.keys():
        uf.make_set(solution)  

    # Calculate similarity and union sets if similarity meets threshold
    for (solution_a, data_a), (solution_b, data_b) in combinations(solutions_behavior.items(), 2):
        similarity = calculate_similarity(data_a, data_b)
        if similarity >= similarity_threshold:
            uf.union(solution_a, solution_b)
    
    # Collect initial clusters from union-find structure
    temp_clusters = {}
    for solution in solutions_behavior.keys():
        root = uf.find(solution)
        if root not in temp_clusters:
            temp_clusters[root] = []
        temp_clusters[root].append(solution)
    
    # Recheck and refine clusters to ensure all elements meet the similarity threshold
    final_clusters = []
    for cluster in temp_clusters.values():
        new_clusters = []
        for solution in cluster:
            found_cluster = False
            # Attempt to place the solution in an existing new cluster
            for new_cluster in new_clusters:
                # Check if solution is similar enough to all members of this new cluster
                if all(calculate_similarity(solutions_behavior[solution], solutions_behavior[other]) >= similarity_threshold for other in new_cluster):
                    new_cluster.append(solution)
                    found_cluster = True
                    break
            if not found_cluster:
                # If no existing cluster is suitable, start a new cluster with this solution
                new_clusters.append([solution])

        # Append refined new clusters to the final list of clusters
        final_clusters.extend(new_clusters)
            
    # Number the clusters as key-value pairs and convert to dictionary
    numbered_clusters = {i+1: cluster for i, cluster in enumerate(final_clusters)}

    return numbered_clusters

def main():
    # -- Get the command line arguments -- #
    args = parse_arguments()
    task = args.task
    model = args.model
    leetcode = args.leetcode_description
    temperature = args.temperature
    sampling = args.sampling
    no_ex_flag = args.no_exceptions
    level = args.level

    logging.info(f"Beginning clustering for {level}/{leetcode}/{model}/{task}/{sampling}")

    if no_ex_flag:
        print()
        logging.info(f"Clustering without those inputs that cause errors...\n")
    
    # --- Create Results Directory --- #
    directory = f"../../data/{level}/{leetcode}/{temperature}/{model}/{task}/{sampling}"
    results_directory = f"{directory}/results"
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)
        logging.info(f"Created directory {results_directory}")
    else:
        logging.info(f"Results directory already exists. Deleting previous clusters for task...")
        if no_ex_flag:
            if os.path.exists(f"{results_directory}/clusters*no_ex.json"):
                os.system(f"rm {results_directory}/clusters*no_ex.json")
        else:
            if os.path.exists(f"{results_directory}/clusters*.json"):
                os.system(f"rm {results_directory}/clusters*.json")

    # -- Fix temperature bug -- #
    if temperature == "0" or temperature == "1" or temperature == "2":

        temperature = int(temperature)
    else:
        temperature = float(temperature)

    # --- Get the behavior dictionaries in the directory --- #
    with open(f'../../data/{level}/{leetcode}/output_corpus_cpp_256.json', 'r') as cpp_file:
        cpp_behavior = json.load(cpp_file)

    with open(f'../../data/{level}/{leetcode}/output_corpus_java_256.json', 'r') as java_file:
        java_behavior = json.load(java_file)

    with open(f'../../data/{level}/{leetcode}/output_corpus_py_256.json', 'r') as python_file:
        python_behavior = json.load(python_file)
        
    solutions_behavior = collect_solution_behavior(directory, cpp_behavior, java_behavior, python_behavior, task, sampling)
    
    if no_ex_flag:
        no_ex_solutions_behavior = filter_out_exceptions_output(solutions_behavior)
    
    if no_ex_flag:
        with open(f"{results_directory}/solutions_behavior_no_ex.json", "w") as file:
            json.dump(no_ex_solutions_behavior, file, indent=4)
    else:
        with open(f"{results_directory}/solutions_behavior.json", "w") as file:
            json.dump(solutions_behavior, file, indent=4)
        
    
    # --- Get Clusters --- #

    clustering_percentages = {
        0.99: "99%",
        0.95: "95%",
        0.75: "75%",
    }

    sampling_types = {
        "once": "One Query",
        "repeated": "Twenty Queries"
    }

    cluster_info = {
        "temperature": temperature,
        "model": model,
        "task": task,
        "sampling": sampling_types[sampling],
        "clusters": {}
    }


    for degree in clustering_percentages.keys():
        clusters = cluster_solutions(solutions_behavior, degree)
        
        if no_ex_flag:
            clusters = cluster_solutions(no_ex_solutions_behavior, degree)
        
        if no_ex_flag:
            with open(f"{results_directory}/clusters_{degree}_no_ex.json", "w") as file:
                json.dump(clusters, file, indent=4)
        else:
            with open(f"{results_directory}/clusters_{degree}.json", "w") as file:
                json.dump(clusters, file, indent=4)


        cluster_info["clusters"].update({clustering_percentages[degree]: len(clusters)})
    
    
    
    # --- Write the cluster information to JSON file --- #
    iteration_key = f'{temperature}/{model}/{task}/{sampling}'

    if no_ex_flag:
        # create the file path
        no_ex_clusters_dir = f"cluster_results/no_ex_clusters/{level}/json"
        no_ex_clusters_filepath = f"{no_ex_clusters_dir}/{leetcode}.json"

        # if the directory does not exist, create it
        if not os.path.exists(no_ex_clusters_dir):
            os.makedirs(no_ex_clusters_dir, exist_ok=True)
        
        # if the file does not exist, create it
        if not os.path.exists(no_ex_clusters_filepath):
            with open(f'{no_ex_clusters_dir}/{leetcode}.json', "w") as file:
                json.dump({}, file, indent=4)
        
        # load the file
        with open(f'{no_ex_clusters_dir}/{leetcode}.json', "r") as file:
            all_no_ex_clusters = json.load(file)

        # update the file
        all_no_ex_clusters.update({iteration_key: cluster_info})

        # write the file
        with open(no_ex_clusters_filepath, "w") as file:
            json.dump(all_no_ex_clusters, file, indent=4)

            
    else:

        # create the file path
        clusters_dir = f"cluster_results/clusters/json"
        clusters_filepath = f"{clusters_dir}/{leetcode}.json"

        # if the file does not exist, create it
        if not os.path.exists(clusters_dir):
            os.makedirs(clusters_dir, exist_ok=True)

        if not os.path.exists(clusters_filepath):
            # and the file
            with open(f'{clusters_dir}/{leetcode}.json', "w") as file:
                json.dump({}, file, indent=4)
        
        # load the file
        with open(f'{clusters_dir}/{leetcode}.json', "r") as file:
            all_clusters = json.load(file)

        # update the file
        all_clusters.update({iteration_key: cluster_info})

        # write the file
        with open(clusters_filepath, "w") as file:
            json.dump(all_clusters, file, indent=4)

    logging.info(f"Clustered for {level}/{leetcode}/{model}/{task}/{sampling} \n")
          
    



if __name__ == "__main__":
    main()