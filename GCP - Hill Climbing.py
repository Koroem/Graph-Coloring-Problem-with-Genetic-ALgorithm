import random
import copy
import re
import networkx as nx
import matplotlib.pyplot as plt
import time
import math
import sys
import os
import numpy as np

def layout_many_components(graph,
                           component_layout_func=nx.layout.spring_layout,
                           pad_x=1., pad_y=1.):
    components = _get_components_sorted_by_size(graph)
    component_sizes = [len(component) for component in components]
    bboxes = _get_component_bboxes(component_sizes, pad_x, pad_y)

    pos = dict()
    for component, bbox in zip(components, bboxes):
        component_pos = _layout_component(component, bbox, component_layout_func)
        pos.update(component_pos)

    return pos

    
def _get_components_sorted_by_size(g):
    subgraphs = (g.subgraph(c) for c in nx.connected_components(g))
    return sorted(subgraphs, key=len)


def _get_component_bboxes(component_sizes, pad_x=1., pad_y=1.):
    bboxes = []
    x, y = (0, 0)
    current_n = 1
    for n in component_sizes:
        width, height = _get_bbox_dimensions(n, power=0.8)

        if not n == current_n: # create a "new line"
            x = 0 # reset x
            y += height + pad_y # shift y up
            current_n = n

        bbox = x, y, width, height
        bboxes.append(bbox)
        x += width + pad_x # shift x down the line
    return bboxes


def _get_bbox_dimensions(n, power=0.5):
    # return (np.sqrt(n), np.sqrt(n))
    return (n**power, n**power)


def _layout_component(component, bbox, component_layout_func):
    pos = component_layout_func(component)
    rescaled_pos = _rescale_layout(pos, bbox)
    return rescaled_pos


def _rescale_layout(pos, bbox):

    min_x, min_y = np.min([v for v in pos.values()], axis=0)
    max_x, max_y = np.max([v for v in pos.values()], axis=0)

    if not min_x == max_x:
        delta_x = max_x - min_x
    else: # graph probably only has a single node
        delta_x = 1.

    if not min_y == max_y:
        delta_y = max_y - min_y
    else: # graph probably only has a single node
        delta_y = 1.

    new_min_x, new_min_y, new_delta_x, new_delta_y = bbox

    new_pos = dict()
    for node, (x, y) in pos.items():
        new_x = (x - min_x) / delta_x * new_delta_x + new_min_x
        new_y = (y - min_y) / delta_y * new_delta_y + new_min_y
        new_pos[node] = (new_x, new_y)

    return new_pos

def generateColors(nr_of_nodes,chromosome):
  colors=[]
  colored_graph=[]
  for _ in range(max(chromosome)+1):
    colors.append("#"+''.join([random.choice('ABCDEF0123456789') for i in range(6)]))
  for i in range(nr_of_nodes):
    colored_graph.append(colors[chromosome[i]])
  return colored_graph

def generateChromosome(nr_of_nodes,nr_of_colors):
    chromosome=[]
    for _ in range(nr_of_nodes):
        chromosome.append(random.randint(0,nr_of_colors-1))
    return chromosome

def fitness2(chromosome,conf):
    conflicts = 0
    num_nodes = len(chromosome)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adjancy_matrix[i][j] == 1 and chromosome[i] == chromosome[j]:
                conflicts += 1
            if conflicts>conf:
                return 1
    return 0

def fitness_node(chromosome,index):
    color=chromosome[index]
    adj_list = adjancy_matrix[index]

    conflicts=0

    for i, is_adjacent in enumerate(adj_list):
       if is_adjacent and chromosome[i] == color:
            conflicts += 1
    return conflicts

def generate_neighbor(coloring,nr_of_colors):
        neighbors = []
        for i in range(len(coloring)):
            node_conflicts=fitness_node(coloring,i)
            if node_conflicts > 0:
                for color in range(nr_of_colors):
                    if coloring[i] != color:
                        new_coloring = coloring.copy()
                        new_coloring[i] = color
                        new_conf_node = fitness_node(new_coloring,i)
                        if new_conf_node < node_conflicts:
                            return new_coloring
        return neighbors

def fitness(chromosome):
    conflicts = 0
    num_nodes = len(chromosome)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adjancy_matrix[i][j] == 1 and chromosome[i] == chromosome[j]:
                conflicts += 1
    return conflicts

def HC(file_name,nr_of_nodes,generations):
    
 start_time = time.time()
 path=r'C:\Users\Koroem\Desktop\[Algoritmi Genetici] Tema 3\Core1\Instances\{}'.format(file_name)
 descriptor=open(path, "r")
 file=descriptor.readlines()

 global adjancy_matrix
 global total_edges
 global nr_of_colors
 global best_coloring_found

 adjancy_matrix=[[0]*nr_of_nodes for _ in range(nr_of_nodes)]
 adjancy=[]

 total_edges=0
 max_degree=0
 generation=0

 for line in file:
   if re.findall("^e",line):
    adjancy.append(list(map(int,(re.findall(r'\d+',line)))))


 for line in adjancy:
      adjancy_matrix[line[0]-1][line[1]-1]=1
      adjancy_matrix[line[1]-1][line[0]-1]=1
      total_edges+=1

 for i in range (nr_of_nodes):
  degree=0
  for j in range(nr_of_nodes):
    if(adjancy_matrix[i][j]==1):
      degree+=1
  if degree>max_degree:
    max_degree=degree

 nr_of_colors=max_degree+1
 best_coloring_found=nr_of_colors

 coloring_array=[]
 generations_array=[]
 time_array=[]
 coloring_time_array=[]

 parent_directory=r"C:\Users\Koroem\Desktop\[Algoritmi Genetici] Tema 3\Core1\Solutions\HCFI"
 path = os.path.join(parent_directory, file_name)
 os.makedirs(path)

 generation_file="{}\Progress tracking.txt".format(path,file_name)

 gf = open(generation_file, "x")

 solution=generateChromosome(nr_of_nodes,nr_of_colors)
 best_solution=copy.deepcopy(solution)

 best_coloring_found=len(set(solution))+1

 while generation<=generations:

    conflicts=fitness(solution)

    if conflicts==0:

        best_coloring_found=len(set(solution))
        nr_of_colors=len(set(solution))-1

        best_solution=copy.deepcopy(solution)

        time_array.append(time.time()-start_time)
        coloring_time_array.append(best_coloring_found)

        solution=generateChromosome(nr_of_nodes,nr_of_colors)
        continue
 
    first_improved_neighbor=generate_neighbor(solution,nr_of_colors) ### First improvement

    if len(first_improved_neighbor)>0:
        solution = first_improved_neighbor
    else:
        solution=generateChromosome(nr_of_nodes,nr_of_colors)

    if generation%25==0:
        with open(generation_file, 'a+') as gf:
            gf.seek(0, 2)
            text='Iteration: {}, attempting coloring: {} , Best fitness: {}\n'.format(generation,nr_of_colors,conflicts)
            gf.write(text)
            coloring_array.append(best_coloring_found)
            generations_array.append(generation)
    generation+=1
 end_time=time.time()
 finish=int(end_time-start_time)
 

 plt.plot(generations_array, coloring_array)
 plt.xlabel('Iterations')
 plt.ylabel('Best coloring')
 plt.title('Algorithm analysis')
 plot_name='{}_Algorithm_analysis.png'.format(file_name)
 plt.savefig('Solutions/HCFI/{}/{}'.format(file_name,plot_name))
 plt.close()

 plt.plot(time_array,coloring_time_array,)
 plt.xlabel('Time(s)')
 plt.ylabel('Best coloring')
 plt.title('Time analysis')
 plot_name='{}_Time_analysis.png'.format(file_name)
 plt.savefig('Solutions/HCFI/{}/{}'.format(file_name,plot_name))
 plt.close()

 color_list=generateColors(nr_of_nodes,best_solution);
 network = nx.Graph()
 nodes=list(range(1,nr_of_nodes))
 network.add_nodes_from(nodes)
 for edges in adjancy:
    network.add_edge(edges[0],edges[1])

 pos = layout_many_components(network, component_layout_func=nx.layout.circular_layout)
 nx.draw_networkx(network,pos,node_color=color_list,node_size=30,with_labels=False)
 plot_name='{}_Graph_coloring_expanded.png'.format(file_name)
 plt.savefig('Solutions/HCFI/{}/{}'.format(file_name,plot_name))
 plt.close()
 

 nx.draw_networkx(network,node_color=color_list,node_size=20,with_labels=False)
 plot_name='{}_Graph_coloring.png'.format(file_name)
 plt.savefig('Solutions/HCFI/{}/{}'.format(file_name,plot_name))
 plt.close()

 text_file='{} - Results.txt'.format(file_name)
 with open('Solutions/HCFI/{}/{}'.format(file_name,text_file), 'w') as f:
    sys.stdout = f 
    print('Problem:',file_name,"\n\n",'Nr of vertexes:',nr_of_nodes,"\n",'Nr of edges:',total_edges,"\n",'Max degree:',max_degree,"\n\n",'First valid coloring found:',
    coloring_time_array[0],"\n",'Best coloring found:',best_coloring_found,"\n\n",
   'Iterations:',generations,"\n\n",'Time for first solution(s):',"{:.4f}".format(time_array[0]),"\n",'Time for the best solution(s):',"{:.4f}".format(time_array[-1]),"\n",'Total Execution time(s):',finish,"\n\n","Conflicts left for next coloring:",conflicts)

def main():
  directory = 'Instances'
  files=os.listdir(directory)

  files.sort(key=lambda x: os.stat(os.path.join(directory, x)).st_size)


  for filename in files:
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
      with open(f,'r') as file:
        text=file.readlines()
        for line in text:
          if re.findall(r"p edge (\d*)", line):
            nr_of_vertex=re.findall(r"edge (\d*)", line)
            nr_of_vertex=int(nr_of_vertex[0])
        try:
            HC(filename,nr_of_vertex,100000)
        except:
            useless=1

main()  