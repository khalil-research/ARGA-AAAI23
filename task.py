import json
import os
import time
import copy
from inspect import signature
from itertools import product, combinations
from queue import PriorityQueue

import rules
from priority_item import PriorityItem
from utils import *
from image import Image
from ARCGraph import ARCGraph

tabu_cool_down = 0


class Task:
    all_possible_abstractions = Image.abstractions
    all_possible_transformations = ARCGraph.transformation_ops

    def __init__(self, filepath):
        """
        contains all information related to an ARC task
        """

        # get task id from filepath
        self.task_id = filepath.split("/")[-1].split(".")[0]

        # input output images given
        self.train_input = []
        self.train_output = []
        self.test_input = []
        self.test_output = []

        # abstracted graphs from input output images
        self.input_abstracted_graphs = dict()  # a dictionary of ARCGraphs, where the keys are the abstraction name and
        self.output_abstracted_graphs = dict()  # values are lists of ARCGraphs with the abs name for all inputs/outputs
        self.input_abstracted_graphs_original = dict()  # a dictionary of ARCGraphs, where the keys are the abstraction name and
        self.output_abstracted_graphs_original = dict()

        # meta data to be kept track of
        self.total_nodes_explored = 0
        self.total_unique_frontier_nodes = 0
        self.frontier_nodes_expanded = 0

        # attributes used for search
        self.shared_frontier = None  # a priority queue of frontier nodes to be expanded
        self.do_constraint_acquisition = None  # whether to do constraint acquisition or not
        self.time_limit = None  # time limit for search
        self.abstraction = None  # which type of abstraction the search is currently working with
        self.static_objects_for_insertion = dict()  # static objects used for the "insert" transformation
        self.object_sizes = dict()  # object sizes to use for filters
        self.object_degrees = dict()  # object degrees to use for filters
        self.skip_abstractions = set()  # a set of abstractions to be skipped in search
        self.transformation_ops = dict()  # a dictionary of transformation operations to be used in search
        self.frontier_hash = dict()  # used for checking if a resulting image is already found by other transformation, one set per abstraction
        self.tabu_list = {}  # used for temporarily disabling expanding frontier for a specific abstraction
        self.tabu_list_waiting = {}  # list of nodes to be added back to frontier once tabu list expired
        self.current_best_scores = {}  # used for tracking the current best score for each abstraction
        self.solution_apply_call = None  # the apply call that produces the best solution
        self.solution_train_error = float("inf")  # the train error of the best solution
        self.current_best_score = float("inf")  # the current best score
        self.current_best_apply_call = None  # the apply call that produces the current best solution
        self.current_best_abstraction = None  # the abstraction that produced the current best solution

        self.load_task_from_file(filepath)
        self.img_dir = "images/" + self.task_id
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)

    def load_task_from_file(self, filepath):
        """
        loads the task from a json file
        """
        with open(filepath) as f:
            data = json.load(f)
        for i, data_pair in enumerate(data["train"]):
            self.train_input.append(
                Image(self, grid=data_pair["input"], name=self.task_id + "_" + str(i + 1) + "_train_in"))
            self.train_output.append(
                Image(self, grid=data_pair["output"], name=self.task_id + "_" + str(i + 1) + "_train_out"))
        for i, data_pair in enumerate(data["test"]):
            self.test_input.append(
                Image(self, grid=data_pair["input"], name=self.task_id + "_" + str(i + 1) + "_test_in"))
            self.test_output.append(
                Image(self, grid=data_pair["output"], name=self.task_id + "_" + str(i + 1) + "_test_out"))

    def solve(self, shared_frontier=True, time_limit=1800, do_constraint_acquisition=True, save_images=False):
        """
        solve for a solution
        :param save_images: whether to save visualization images of the search process.
        :param shared_frontier: whether the search uses a shared frontier between abstractions.
        :param time_limit: maximum time allowed for search in seconds.
        :param do_constraint_acquisition: whether constraint acquisition is used.
        :return:
        """
        self.shared_frontier = shared_frontier
        self.do_constraint_acquisition = do_constraint_acquisition
        self.time_limit = time_limit
        if shared_frontier:
            self.frontier = PriorityQueue()  # frontier for search, each item is a PriorityItem object
        else:
            self.frontier = dict()  # maintain a separate frontier for each abstraction

        print("Running task.solve() for #{}".format(self.task_id), flush=True)
        if save_images:
            for input in self.train_input:
                input.arc_graph.plot(save_fig=True)

        self.start_time = time.time()

        # initialize frontier
        stop_search = self.initialize_frontier()

        # main search loop
        while not stop_search:
            if self.shared_frontier:
                stop_search = self.search_shared_frontier()
            else:
                stop_search = self.search_separate_frontier()
        solving_time = time.time() - self.start_time

        # plot reconstructed train images
        if save_images:
            for i, g in enumerate(self.input_abstracted_graphs_original[self.abstraction]):

                g.plot(save_fig=True)
                for j, call in enumerate(self.solution_apply_call):
                    g.apply(**call)
                    g.plot(save_fig=True, file_name=g.name + "_{}".format(j))

                reconstructed = self.train_input[i].undo_abstraction(g)
                reconstructed.plot(save_fig=True)
                self.train_output[i].arc_graph.plot(save_fig=True)

        # apply to test image
        test_input = self.test_input[0]
        abstracted_graph = getattr(test_input, Image.abstraction_ops[self.abstraction])()
        abstracted_graph.plot(save_fig=True)
        for j, call in enumerate(self.solution_apply_call):
            abstracted_graph.apply(**call)
            abstracted_graph.plot(save_fig=True, file_name=abstracted_graph.name + "_{}".format(j))
        reconstructed = test_input.undo_abstraction(abstracted_graph)
        if save_images:
            test_input.arc_graph.plot(save_fig=True)
            reconstructed.plot(save_fig=True)
            self.test_output[0].arc_graph.plot(save_fig=True)

        # check if the solution found the correct test output
        error = 0
        for node, data in self.test_output[0].graph.nodes(data=True):
            if data["color"] != reconstructed.graph.nodes[node]["color"]:
                error += 1
        if error == 0:
            print("The solution found produced the correct test output!")
        else:
            print("The solution found predicted {} out of {} pixels incorrectly".format(error, len(
                self.test_output[0].graph.nodes())))

        nodes_explored = {"total_nodes_explored": self.total_nodes_explored,
                          "total_unique_frontier_nodes": self.total_unique_frontier_nodes,
                          "frontier_nodes_expanded": self.frontier_nodes_expanded}

        return self.abstraction, self.solution_apply_call, error / len(
            self.test_output[0].graph.nodes()), self.solution_train_error, solving_time, nodes_explored

    def initialize_frontier(self):
        """
        initializes frontier
        :return: True if a solution is found during initialization or time limit has been reached, False otherwise
        """
        print("Initializing Frontier")

        existing_init_abstracted_graphs = {}  # keep track of existing abstracted graphs to check for duplication

        for abstraction in self.all_possible_abstractions:
            # specify the abstraction currently working with
            self.abstraction = abstraction

            # initialize individual frontiers if abstractions do not share one
            if not self.shared_frontier:
                self.frontier[abstraction] = PriorityQueue()

            # initialize additional attributes used in search
            self.current_best_scores[abstraction] = float("inf")
            self.tabu_list[abstraction] = 0
            self.tabu_list_waiting[abstraction] = []
            self.frontier_hash[abstraction] = set()

            # first, produce the abstracted graphs for input output images using the current abstraction
            # these are the 'original' abstracted graphs that will not be updated
            self.input_abstracted_graphs_original[abstraction] = \
                [getattr(input, Image.abstraction_ops[abstraction])() for input in self.train_input]
            self.output_abstracted_graphs_original[abstraction] = \
                [getattr(output, Image.abstraction_ops[abstraction])() for output in self.train_output]

            # skip abstraction if it result in the same set of abstracted graphs as a previous abstraction,
            # for example: nbccg and ccgbr result in the same graphs if there are no enclosed black pixels
            found_match = False
            if len(existing_init_abstracted_graphs) != 0:
                for abs, existing_abs_graphs in existing_init_abstracted_graphs.items():
                    for instance, existing_abs_graph in enumerate(existing_abs_graphs):
                        existing_set = set()
                        new_set = set()
                        for n, subnodes1 in self.input_abstracted_graphs_original[abstraction][instance].graph.nodes(
                                data="nodes"):
                            existing_set.add(frozenset(subnodes1))
                        for m, subnodes2 in existing_abs_graph.graph.nodes(data="nodes"):
                            new_set.add(frozenset(subnodes2))
                        if existing_set != new_set:
                            break  # break if did not match for this instance
                    else:  # did not break, found match for all instances
                        found_match = True
                        break
                if found_match:  # found matching node for all nodes in all abstractions
                    print("Skipping abstraction {} as it is the same as abstraction {}".format(abstraction, abs))
                    self.skip_abstractions.add(self.abstraction)
                    continue
            existing_init_abstracted_graphs[abstraction] = self.input_abstracted_graphs_original[abstraction]

            # get the list of object sizes and degrees
            self.get_static_object_attributes(abstraction)

            # keep a list of transformation ops that we modify based on constraint acquisition results
            self.transformation_ops[abstraction] = self.all_possible_transformations[self.abstraction]

            # constraint acquisition (global)
            if self.do_constraint_acquisition:
                self.constraints_acquisition_global()

            # look for static objects to insert if insert transformation is not pruned by constraint acquisition
            self.static_objects_for_insertion[abstraction] = []
            if len(set(self.transformation_ops[abstraction]) & set(ARCGraph.insertion_transformation_ops)) > 0:
                self.get_static_inserted_objects()

            # initiate frontier with dummy node and expand it (representing doing nothing to the input image)
            frontier_node = PriorityItem([], abstraction, float("inf"), float("inf"))
            self.expand_frontier(frontier_node)

            if self.shared_frontier:
                if len(self.frontier.queue) == 0:  # the current abstraction generated no valid results
                    self.skip_abstractions.add(self.abstraction)
                    continue
                frontier_score = self.frontier.queue[0].priority
            else:
                if len(self.frontier[self.abstraction].queue) == 0:
                    self.skip_abstractions.add(self.abstraction)
                    continue
                frontier_score = self.frontier[self.abstraction].queue[0].priority

            # check if solution exists in the newly expanded frontier
            if frontier_score == 0:  # if priority is 0, the goal is reached
                if self.shared_frontier:
                    frontier_node = self.frontier.get(False)
                else:
                    frontier_node = self.frontier[self.abstraction].get(False)
                self.solution_apply_call = frontier_node.data
                self.solution_train_error = frontier_node.priority
                print("Solution Found! Abstraction used: {}, Apply Call = ".format(self.abstraction))
                print(frontier_node.data)
                print("Runtime till solution: {}".format(time.time() - self.start_time))
                return True

            if time.time() - self.start_time > self.time_limit:  # timeout
                self.solution_apply_call = frontier_node.data
                self.solution_train_error = frontier_node.priority
                self.abstraction = frontier_node.abstraction
                print("Solution Not Found! Best Solution has cost of {}, Abstraction used: {}, Apply Call = ".format(
                    frontier_node.priority, self.abstraction))
                print(self.solution_apply_call)
                print("Runtime till solution: {}".format(time.time() - self.start_time))
                return True

        return False

    def search_shared_frontier(self):
        """
        perform one iteration of search for a solution using a shared frontier
        :return: True if a solution is found or time limit has been reached, False otherwise
        """

        if self.frontier.empty():  # exhausted search space
            self.solution_apply_call = self.current_best_apply_call
            self.solution_train_error = self.current_best_score
            self.abstraction = self.current_best_abstraction
            print("Solution Not Found due to empty search space! Best Solution has cost of {}, "
                  "Abstraction used: {}, Apply Call = ".format(self.current_best_score, self.abstraction))
            print(self.current_best_apply_call)
            print("Runtime till solution: {}".format(time.time() - self.start_time))
            return True

        frontier_node = self.frontier.get(False)

        # if this abstraction is on tabu list, explore something else
        if self.tabu_list[frontier_node.abstraction] > 0:
            # print("abstraction {} is in the tabu list with cool down = {}".format(frontier_node.abstraction, self.tabu_list[frontier_node.abstraction]))
            self.tabu_list_waiting[frontier_node.abstraction].append(frontier_node)
            return False
        # if this abstraction is not on tabu list, but has a worse score than before,
        # explore it and put it on tabu list
        elif frontier_node.priority >= self.current_best_scores[frontier_node.abstraction]:
            self.tabu_list[frontier_node.abstraction] = tabu_cool_down + 1
        else:
            self.current_best_scores[frontier_node.abstraction] = frontier_node.priority

        apply_calls = frontier_node.data
        self.abstraction = frontier_node.abstraction

        # check for solution
        if frontier_node.priority == 0:  # if priority is 0, the goal is reached
            self.solution_apply_call = apply_calls
            self.solution_train_error = 0
            print("Solution Found! Abstraction used: {}, Apply Call = ".format(self.abstraction))
            print(apply_calls)
            print("Runtime till solution: {}".format(time.time() - self.start_time))
            return True
        else:
            if frontier_node.priority < self.current_best_score:
                self.current_best_score = frontier_node.priority
                self.current_best_apply_call = apply_calls
                self.current_best_abstraction = self.abstraction

        print("Exploring frontier node with score {} at depth {} with abstraction {} and apply calls:".format(
            frontier_node.priority, len(apply_calls), self.abstraction))
        print(apply_calls)
        self.expand_frontier(frontier_node)

        all_on_tabu = all(tabu > 0 for tabu in self.tabu_list.values())
        for abs, tabu in self.tabu_list.items():
            if all_on_tabu:
                self.tabu_list[abs] = 0
                for node in self.tabu_list_waiting[abs]:  # put the nodes in waiting list back into frontier
                    self.frontier.put(node)
            elif tabu > 0:
                self.tabu_list[abs] = tabu - 1
                if tabu - 1 == 0:
                    for node in self.tabu_list_waiting[abs]:  # put the nodes in waiting list back into frontier
                        self.frontier.put(node)

        if time.time() - self.start_time > self.time_limit:  # timeout
            self.solution_apply_call = self.current_best_apply_call
            self.solution_train_error = self.current_best_score
            self.abstraction = self.current_best_abstraction
            print("Solution Not Found due to time limit reached! Best Solution has cost of {}, "
                  "Abstraction used: {}, Apply Call = ".format(self.current_best_score, self.abstraction))
            print(self.current_best_apply_call)
            print("Runtime till solution: {}".format(time.time() - self.start_time))
            return True
        return False

    def search_separate_frontier(self):
        """
        perform one iteration of search for a solution using a multiple frontiers
        :return: True if a solution is found or time limit has been reached, False otherwise
        """

        for abstraction in Image.abstractions:
            self.abstraction = abstraction

            if self.abstraction in self.skip_abstractions:
                continue

            # if this abstraction is on tabu list, explore something else
            if self.tabu_list[self.abstraction] > 0:
                self.tabu_list[self.abstraction] = self.tabu_list[self.abstraction] - 1
                continue

            frontier_node = self.frontier[self.abstraction].get()
            apply_calls = frontier_node.data

            # if this abstraction is not on tabu list, but has a worse score than before,
            # explore it and put it on tabu list
            if frontier_node.priority >= self.current_best_scores[self.abstraction]:
                # print("abstraction {} is put on the tabu list".format(frontier_node.abstraction))
                self.tabu_list[self.abstraction] = tabu_cool_down + 1
            else:
                self.current_best_scores[self.abstraction] = frontier_node.priority

            # check for solution
            if frontier_node.priority == 0:  # if priority is 0, the goal is reached
                self.solution_apply_call = apply_calls
                self.solution_train_error = 0
                print("Solution Found! Abstraction used: {}, Apply Call = ".format(self.abstraction))
                print(apply_calls)
                print("Runtime till solution: {}".format(time.time() - self.start_time))
                return True
            else:
                if frontier_node.priority < self.current_best_score:
                    self.current_best_score = frontier_node.priority
                    self.current_best_apply_call = apply_calls
                    self.current_best_abstraction = self.abstraction

            print(
                "Exploring frontier node with score {} at depth {} with abstraction {} and apply calls:".format(
                    frontier_node.priority, len(apply_calls), self.abstraction))
            print(apply_calls)
            self.expand_frontier(frontier_node)

            if time.time() - self.start_time > self.time_limit:  # timeout
                self.solution_apply_call = self.current_best_apply_call
                self.solution_train_error = self.current_best_score
                self.abstraction = self.current_best_abstraction
                print(
                    "Solution Not Found! Best Solution has cost of {}, Abstraction used: {}, Apply Call = ".format(
                        self.current_best_score, self.abstraction))
                print(self.current_best_apply_call)
                print("Runtime till solution: {}".format(time.time() - self.start_time))
                return True
        return False

    def expand_frontier(self, frontier_node):
        """
        expand one frontier node
        """
        self.frontier_nodes_expanded += 1
        print("Expanding frontier node with abstraction {}".format(self.abstraction))
        self.input_abstracted_graphs[self.abstraction] = []  # up to date abstracted graphs
        for input_abstracted_graph in self.input_abstracted_graphs_original[self.abstraction]:
            input_abstracted = input_abstracted_graph.copy()
            for apply_call in frontier_node.data:
                input_abstracted.apply(**apply_call)  # apply the transformation to the abstracted graph
            self.input_abstracted_graphs[self.abstraction].append(input_abstracted)

        filters = self.get_candidate_filters()
        if (time.time() - self.start_time) > self.time_limit:
            return
        apply_calls = self.get_candidate_transformations(filters)
        print("Number of New Candidate Nodes = {}".format(len(apply_calls)))
        added_nodes = 0
        # for apply_call in tqdm(apply_calls):
        for apply_call in apply_calls:
            self.total_nodes_explored += 1
            cumulated_apply_calls = frontier_node.data.copy()
            cumulated_apply_calls.append(apply_call)
            apply_call_score, results_token = self.calculate_score(cumulated_apply_calls)
            if apply_call_score == -1:
                continue
            if results_token in self.frontier_hash[self.abstraction]:
                if (time.time() - self.start_time) > self.time_limit:
                    break
                continue
            else:
                added_nodes += 1
                self.frontier_hash[self.abstraction].add(results_token)
                secondary_score = len(cumulated_apply_calls)
                priority_item = PriorityItem(cumulated_apply_calls, self.abstraction, apply_call_score, secondary_score)
                if self.shared_frontier:
                    self.frontier.put(priority_item)
                else:
                    self.frontier[self.abstraction].put(priority_item)

                # stop if solution is found or time is up
                if apply_call_score == 0:
                    break
                if (time.time() - self.start_time) > self.time_limit:
                    break
        print("Number of New Nodes Added to Frontier = {}".format(added_nodes))
        self.total_unique_frontier_nodes += added_nodes

    def get_candidate_filters(self):
        """
        return list of candidate filters
        """
        ret_apply_filter_calls = []  # final list of filter calls
        filtered_nodes_all = []  # use this list to avoid filters that return the same set of nodes

        for filter_op in ARCGraph.filter_ops:
            # first, we generate all possible values for each parameter
            sig = signature(getattr(ARCGraph, filter_op))
            generated_params = []
            for param in sig.parameters:
                param_name = sig.parameters[param].name
                param_type = sig.parameters[param].annotation
                param_default = sig.parameters[param].default
                if param_name == "self" or param_name == "node":
                    continue
                if param_name == "color":
                    generated_params.append([c for c in range(10)] + ["most", "least"])
                elif param_name == "size":
                    generated_params.append([w for w in self.object_sizes[self.abstraction]] + ["min", "max", "odd"])
                elif param_name == "degree":
                    generated_params.append([d for d in self.object_degrees[self.abstraction]] + ["min", "max", "odd"])
                elif param_type == bool:
                    generated_params.append([True, False])
                elif issubclass(param_type, Enum):
                    generated_params.append([value for value in param_type])

            # then, we combine all generated values to get all possible combinations of parameters
            for item in product(*generated_params):

                # generate dictionary, keys are the parameter names, values are the corresponding values
                param_vals = {}
                for i, param in enumerate(list(sig.parameters)[2:]):  # skip "self", "node"
                    param_vals[sig.parameters[param].name] = item[i]
                candidate_filter = {"filters": [filter_op], "filter_params": [param_vals]}

                #  do not include if the filter result in empty set of nodes (this will be the majority of filters)
                filtered_nodes = []
                applicable_to_all = True
                for input_abstracted_graph in self.input_abstracted_graphs[self.abstraction]:
                    filtered_nodes_i = []
                    for node in input_abstracted_graph.graph.nodes():
                        if input_abstracted_graph.apply_filters(node, **candidate_filter):
                            filtered_nodes_i.append(node)
                    if len(filtered_nodes_i) == 0:
                        applicable_to_all = False
                    filtered_nodes.extend(filtered_nodes_i)
                filtered_nodes.sort()
                # does not result in empty or duplicate set of nodes
                if applicable_to_all and filtered_nodes not in filtered_nodes_all:
                    ret_apply_filter_calls.append(candidate_filter)
                    filtered_nodes_all.append(filtered_nodes)

        # generate filter calls with two filters
        single_filter_calls = [d.copy() for d in ret_apply_filter_calls]
        for filter_i, (first_filter_call, second_filter_call) in enumerate(combinations(single_filter_calls, 2)):
            if filter_i % 1000 == 0:
                if (time.time() - self.start_time) > self.time_limit:
                    break

            candidate_filter = copy.deepcopy(first_filter_call)
            candidate_filter["filters"].extend(second_filter_call["filters"])
            candidate_filter["filter_params"].extend(second_filter_call["filter_params"])

            filtered_nodes = []
            applicable_to_all = True
            for input_abstracted_graph in self.input_abstracted_graphs[self.abstraction]:
                filtered_nodes_i = []
                for node in input_abstracted_graph.graph.nodes():
                    if input_abstracted_graph.apply_filters(node, **candidate_filter):
                        filtered_nodes_i.append(node)
                if len(filtered_nodes_i) == 0:
                    applicable_to_all = False
                filtered_nodes.extend(filtered_nodes_i)
            filtered_nodes.sort()
            # does not result in empty or duplicate set of nodes
            if applicable_to_all and filtered_nodes not in filtered_nodes_all:
                ret_apply_filter_calls.append(candidate_filter)
                filtered_nodes_all.append(filtered_nodes)

        print("Found {} Applicable Filters".format(len(ret_apply_filter_calls)))
        return ret_apply_filter_calls

    def get_candidate_transformations(self, apply_filters_calls):
        """
        generate candidate transformations, return list of full operations candidates
        """

        ret_apply_calls = []
        for apply_filters_call in apply_filters_calls:
            if time.time() - self.start_time > self.time_limit:
                break
            if self.do_constraint_acquisition:
                constraints = self.constraints_acquisition_local(apply_filters_call)
                transformation_ops = self.prune_transformations(constraints)
            else:
                transformation_ops = self.transformation_ops[self.abstraction]
            for transform_op in transformation_ops:
                sig = signature(getattr(ARCGraph, transform_op))
                generated_params = self.parameters_generation(apply_filters_call, sig)
                for item in product(*generated_params):
                    param_vals = {}
                    for i, param in enumerate(list(sig.parameters)[2:]):  # skip "self", "node"
                        param_vals[sig.parameters[param].name] = item[i]
                    ret_apply_call = apply_filters_call.copy()  # dont need deep copy here since we are not modifying existing entries
                    ret_apply_call["transformation"] = [transform_op]
                    ret_apply_call["transformation_params"] = [param_vals]
                    ret_apply_calls.append(ret_apply_call)
        return ret_apply_calls

    def parameters_generation(self, apply_filters_call, transform_sig):
        """
        given filter nodes and a transformation, generate parameters to be passed to the transformation
        example: given filters for red nodes and move_node_max,
            return [up, down, left, right, get_relative_pos(red nodes, blue neighbors of red nodes), ...]
        :param apply_filters_call: the specific apply filter call to get the nodes to apply transformations to
        :param all_calls: all apply filter calls, this is used to generate the dynamic parameters
        :param transform_sig: signature for a transformation
        :return: parameters to be passed to the transformation
        """
        generated_params = []
        for param in transform_sig.parameters:
            param_name = transform_sig.parameters[param].name
            param_type = transform_sig.parameters[param].annotation
            param_default = transform_sig.parameters[param].default
            if param_name == "self" or param_name == "node":  # nodes are already generated using the filters
                continue

            # first we generate the static values
            if param_name == "color":
                all_possible_values = [c for c in range(10)] + ["most", "least"]
            elif param_name == "fill_color" or param_name == "border_color":
                all_possible_values = [c for c in range(10)]
            elif param_name == "object_id":
                all_possible_values = [id for id in range(len(self.static_objects_for_insertion[self.abstraction]))] + [
                    -1]
            elif param_name == "point":  # for insertion, could be ImagePoints or a coordinate on image (tuple)
                all_possible_values = [value for value in ImagePoints]
            elif issubclass(param_type, Enum):
                all_possible_values = [value for value in param_type]
            elif param_type == bool:
                all_possible_values = [True, False]
            elif param_default is None:
                all_possible_values = [None]
            else:
                all_possible_values = []

            # then we add dynamic values for parameters
            if param_name in ARCGraph.dynamic_parameters:
                filtered_nodes_all = []
                # the filters that defines the dynamic parameter values, has their own parameters generated_filter_params
                for param_binding_op in ARCGraph.param_binding_ops:
                    sig = signature(getattr(ARCGraph, param_binding_op))
                    generated_filter_params = []
                    for param in sig.parameters:
                        filter_param_name = sig.parameters[param].name
                        filter_param_type = sig.parameters[param].annotation
                        if filter_param_name == "self" or filter_param_name == "node":
                            continue
                        if filter_param_name == "color":
                            # generated_params[param_name] = [c for c in range(10)]
                            generated_filter_params.append([c for c in range(10)] + ["most", "least"])
                        elif filter_param_name == "size":
                            generated_filter_params.append(
                                [w for w in self.object_sizes[self.abstraction]] + ["min", "max"])
                        elif filter_param_type == bool:
                            # generated_params[param_name] = [True, False]
                            generated_filter_params.append([True, False])
                        elif issubclass(filter_param_type, Enum):
                            generated_filter_params.append([value for value in filter_param_type])

                    for item in product(*generated_filter_params):
                        param_vals = {}
                        for i, param in enumerate(list(sig.parameters)[2:]):  # skip "self", "node"
                            param_vals[sig.parameters[param].name] = item[i]
                        applicable_to_all = True
                        param_bind_nodes = []
                        for input_abstracted_graph in self.input_abstracted_graphs[self.abstraction]:
                            param_bind_nodes_i = []
                            for filtered_node in input_abstracted_graph.graph.nodes():
                                if input_abstracted_graph.apply_filters(filtered_node, **apply_filters_call):
                                    param_binded_node = getattr(input_abstracted_graph, param_binding_op)(filtered_node,
                                                                                                          **param_vals)
                                    if param_binded_node is None:
                                        # unable to find node for filtered node to bind parameter to
                                        applicable_to_all = False
                                        break
                                    param_bind_nodes_i.append(param_binded_node)

                            param_bind_nodes.append(param_bind_nodes_i)
                            if len(param_bind_nodes_i) == 0:
                                applicable_to_all = False
                        if applicable_to_all and param_bind_nodes not in filtered_nodes_all:
                            all_possible_values.append({"filters": [param_binding_op], "filter_params": [param_vals]})
                            filtered_nodes_all.append(param_bind_nodes)
            generated_params.append(all_possible_values)
        return generated_params

    def calculate_score(self, apply_call):
        """
        calculate the total score across all training examples for a given apply call.
        hash the apply call by converting the results to a string and use it as a key to the cache.
        return -1, -1 if the apply call is invalid
        """

        input_abstracted_graphs = [input_abstracted.copy() for input_abstracted in
                                   self.input_abstracted_graphs_original[self.abstraction]]
        try:
            for input_abstracted_graph in input_abstracted_graphs:
                for call in apply_call:
                    input_abstracted_graph.apply(**call)
        except:
            return -1, -1

        token_string = ''

        score = 0
        for i, output in enumerate(self.train_output):
            reconstructed = self.train_input[i].undo_abstraction(input_abstracted_graphs[i])

            # hashing
            for r in range(output.height):
                for c in range(output.width):
                    token_string = token_string + str(reconstructed.graph.nodes[(r, c)]["color"])
            for node, data in input_abstracted_graphs[i].graph.nodes(data=True):
                for j, pixel in enumerate(data["nodes"]):
                    if input_abstracted_graphs[i].is_multicolor:
                        token_string = token_string + str(data["color"][j])
                    else:
                        token_string = token_string + str(data["color"])

            # scoring
            for node, data in output.graph.nodes(data=True):
                if data["color"] != reconstructed.graph.nodes[node]["color"]:
                    if data["color"] == output.background_color or reconstructed.graph.nodes[node][
                        "color"] == output.background_color:
                        # incorrectly identified object/background
                        score += 2
                    else:  # correctly identified object/background but got the color wrong
                        score += 1

        if token_string == "":  # special case when: flatten abstraction resulting in an empty abstracted graph
            token_string = -1

        return score, int(token_string)

    # --------------------------------------Constraint Acquisition-----------------------------------
    def constraints_acquisition_global(self):
        """
        find the constraints that all nodes in the instance must follow
        """
        no_movements = True
        for i, input in enumerate(self.train_input):
            for node, data in input.graph.nodes(data=True):
                if (data["color"] != input.background_color and self.train_output[i].graph.nodes[node][
                    "color"] == input.background_color) \
                        or (data["color"] == input.background_color and self.train_output[i].graph.nodes[node][
                    "color"] != input.background_color):
                    no_movements = False
        no_new_objects = True
        for i, output_abstracted_graph in enumerate(self.output_abstracted_graphs_original[self.abstraction]):
            input_abstracted_nodes = self.input_abstracted_graphs_original[self.abstraction][i].graph.nodes()
            for abstracted_node, data in output_abstracted_graph.graph.nodes(data=True):
                if abstracted_node not in input_abstracted_nodes:
                    no_new_objects = False
                    break
        if no_movements:
            pruned_transformations = ["move_node", "extend_node", "move_node_max", "fill_rectangle", "add_border",
                                      "insert"]
            self.transformation_ops[self.abstraction] = [t for t in self.transformation_ops[self.abstraction] if
                                                         t not in pruned_transformations]
        elif no_new_objects:
            pruned_transformations = ["insert"]
            self.transformation_ops[self.abstraction] = [t for t in self.transformation_ops[self.abstraction] if
                                                         t not in pruned_transformations]

    def constraints_acquisition_local(self, apply_filter_call):
        """
        given an apply_filter_call, find the set of constraints that
        the nodes returned by the apply_filter_call must satisfy.
        these are called local constraints as they apply to only the nodes
        that satisfies the filter.
        """
        found_constraints = []
        for rule in rules.list_of_rules:
            if self.apply_constraint(rule, apply_filter_call):
                found_constraints.append(rule)
        return found_constraints

    def apply_constraint(self, rule, apply_filter_call):
        """
        check if the given rule holds for all training instances for the given apply_filter_call
        """
        satisfied = True
        for index in range(len(self.train_input)):
            params = self.constraints_param_generation(apply_filter_call, rule, index)
            satisfied = satisfied and getattr(rules, rule)(*params)
        return satisfied

    def constraints_param_generation(self, condition, rule, training_index):
        """
        given condition and rule, first generate the sequence using the condition
        then transform the sequence into the expected format for the constraint
        :param condition: {'filters': ['filter_nodes_by_color'],
          'filter_params': [{'color': 0, 'exclude': True}]}
        :param rule: "rule_name"
        :param training_index: training instance index
        """

        input_abs = self.input_abstracted_graphs[self.abstraction][training_index]
        output_abs = self.output_abstracted_graphs_original[self.abstraction][training_index]

        input_nodes = []
        for node in input_abs.graph.nodes():
            if input_abs.apply_filters(node, **condition):
                input_nodes.append(node)

        output_nodes = []
        for node in output_abs.graph.nodes():
            if output_abs.apply_filters(node, **condition):
                output_nodes.append(node)

        if rule == "color_equal":
            input_sequence = [input_abs.graph.nodes[node]["color"] for node in input_nodes]
            output_sequence = [output_abs.graph.nodes[node]["color"] for node in output_nodes]
            input_sequence.sort()
            output_sequence.sort()
            args = [input_sequence, output_sequence]

        elif rule == "position_equal":
            input_sequence = []
            output_sequence = []
            for node in input_nodes:
                input_sequence.extend([subnode for subnode in input_abs.graph.nodes[node]["nodes"]])
            for node in output_nodes:
                output_sequence.extend([subnode for subnode in output_abs.graph.nodes[node]["nodes"]])
            input_sequence.sort()
            output_sequence.sort()
            args = [input_sequence, output_sequence]

        elif rule == "size_equal":
            input_sequence = [input_abs.graph.nodes[node]["size"] for node in input_nodes]
            output_sequence = [output_abs.graph.nodes[node]["size"] for node in output_nodes]
            input_sequence.sort()
            output_sequence.sort()
            args = [input_sequence, output_sequence]
        return args

    def prune_transformations(self, constraints):
        """
        given a set of constraints that must be satisfied, return a set of transformations that do not violate them
        """
        transformations = self.transformation_ops[self.abstraction]
        for constraint in constraints:
            if constraint == "color_equal":
                pruned_transformations = ["update_color"]
            elif constraint == "position_equal":
                pruned_transformations = ["move_node", "extend_node", "move_node_max"]
            elif constraint == "size_equal":
                pruned_transformations = ["extend_node"]
            transformations = [t for t in transformations if t not in pruned_transformations]
        return transformations

    # --------------------------------- Utility Functions ---------------------------------
    def get_static_inserted_objects(self):
        """
        populate self.static_objects_for_insertion, which contains all static objects detected in the images.
        """
        self.static_objects_for_insertion[self.abstraction] = []
        existing_objects = []

        for i, output_abstracted_graph in enumerate(self.output_abstracted_graphs_original[self.abstraction]):
            # difference_image = self.train_output[i].copy()
            input_abstracted_nodes = self.input_abstracted_graphs_original[self.abstraction][i].graph.nodes()
            for abstracted_node, data in output_abstracted_graph.graph.nodes(data=True):
                if abstracted_node not in input_abstracted_nodes:
                    new_object = data.copy()
                    min_x = min([subnode[1] for subnode in new_object["nodes"]])
                    min_y = min([subnode[0] for subnode in new_object["nodes"]])
                    adjusted_subnodes = []
                    for subnode in new_object["nodes"]:
                        adjusted_subnodes.append((subnode[0] - min_y, subnode[1] - min_x))
                    adjusted_subnodes.sort()
                    if adjusted_subnodes not in existing_objects:
                        existing_objects.append(adjusted_subnodes)
                        self.static_objects_for_insertion[self.abstraction].append(new_object)

    def get_static_object_attributes(self, abstraction):
        """
        populate self.object_sizes and self.object_degrees, which contains all sizes and degrees existing objects
        """
        self.object_sizes[abstraction] = set()
        self.object_degrees[abstraction] = set()
        for abs_graph in self.input_abstracted_graphs_original[abstraction]:
            for node, size in abs_graph.graph.nodes(data="size"):
                self.object_sizes[abstraction].add(size)
            for node, degree in abs_graph.graph.degree():
                self.object_degrees[abstraction].add(degree)

    def apply_solution(self, apply_call, abstraction, save_images=False):
        """
        apply solution abstraction and apply_call to test image
        """
        self.abstraction = abstraction
        self.input_abstracted_graphs_original[abstraction] = [getattr(input, Image.abstraction_ops[abstraction])() for
                                                              input in self.train_input]
        self.output_abstracted_graphs_original[abstraction] = [getattr(output, Image.abstraction_ops[abstraction])() for
                                                               output in self.train_output]
        self.get_static_inserted_objects()
        test_input = self.test_input[0]
        abstracted_graph = getattr(test_input, Image.abstraction_ops[abstraction])()
        for call in apply_call:
            abstracted_graph.apply(**call)
        reconstructed = test_input.undo_abstraction(abstracted_graph)
        if save_images:
            test_input.arc_graph.plot(save_fig=True)
            reconstructed.plot(save_fig=True)
            self.test_output[0].arc_graph.plot(save_fig=True)
        return reconstructed
