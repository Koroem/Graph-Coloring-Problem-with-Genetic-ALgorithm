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

def generatePopulation(population_size,nr_of_nodes,nr_of_colors):
    population=[]
    for _ in range(population_size):
        population.append(generateChromosome(nr_of_nodes,nr_of_colors))
    return population

def fill(population,population_size,nr_of_nodes,nr_of_colors):
    pop=population.copy()
    while len(pop)<population_size:
        pop.append(generateChromosome(nr_of_nodes,nr_of_colors))
    return pop

def fitness(chromosome):
    conflicts = 0
    num_nodes = len(chromosome)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adjancy_matrix[i][j] == 1 and chromosome[i] == chromosome[j]:
                conflicts += 1
    return conflicts

def parent_selection_1(population):
    temp_parents = random.sample(population, 2)
    parent1 = min(temp_parents, key=fitness)
    temp_parents = random.sample(population, 2)
    parent2 = min(temp_parents, key=fitness)
    return [parent1, parent2]

def parent_selection_2(population):
    return population[0]

def mutation_1(chromosome,adjacency_matrix,mutation_chance,all_colors):
    if random.random()<mutation_chance:
        for i, color in enumerate(chromosome):
                adjacent_colors = [chromosome[j] for j, is_adjacent in enumerate(adjacency_matrix[i]) if is_adjacent and i != j]
                if color in adjacent_colors:
                    valid_colors = [c for c in all_colors if c not in adjacent_colors]
                    if len(valid_colors)>0:
                        new_color = random.choice(valid_colors)
                        chromosome[i] = new_color
    return chromosome

def mutation_2(chromosome,adjancy_matrix,mutation_chance,all_colors):
    if random.random()<mutation_chance:
        for i, color in enumerate(chromosome):
                adjacent_colors = [chromosome[j] for j, is_adjacent in enumerate(adjancy_matrix[i]) if is_adjacent and i != j]
                if color in adjacent_colors:
                    new_color = random.choice(all_colors)
                    chromosome[i] = copy.deepcopy(new_color)
    return chromosome

def crossover(parent1, parent2):
    crosspoint = random.randint(1, len(parent1) - 1)
    child = parent1[:crosspoint+1] + parent2[crosspoint+1:]
    return child


def gao(file_name,nr_of_nodes,population_size,generations):

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

 mutation_chance=0.7
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

 all_colors=[]
 for i in range(nr_of_colors):
    all_colors.append(i)
 
 coloring_array=[]
 generations_array=[]
 time_array=[]
 coloring_time_array=[]

 parent_directory=r"C:\Users\Koroem\Desktop\[Algoritmi Genetici] Tema 3\Core1\Solutions\GA"
 path = os.path.join(parent_directory, file_name)
 try:
    os.makedirs(path)
 except: return
 
 generation_file="{}\Progress tracking (minimezed) .txt".format(path,file_name)
 generation_file_real="{}\Progress tracking.txt".format(path,file_name)
 gf = open(generation_file, "x")

 population=generatePopulation(population_size,nr_of_nodes,nr_of_colors)

 best_chromosome=population[0]

 while generation<=generations:

    population=fill(population,population_size*2,nr_of_nodes,nr_of_colors)
    population=sorted(population, key=fitness)
    population=population[:len(population)//2]

    best_fitness=fitness(population[0])
    
    if best_fitness>=0:
        if best_fitness>4:
                parents=parent_selection_1(population)
                child=crossover(parents[0],parents[1])
                child=mutation_1(child,adjancy_matrix,mutation_chance,all_colors)
        else:
                child=parent_selection_2(population)
                child=mutation_2(child,adjancy_matrix,mutation_chance,all_colors)
    
    population.insert(0, child)

    if fitness(population[0])==0:
                nr_of_colors=len(set(population[0]))-1
                best_coloring_found=len(set(population[0]))
                best_chromosome=copy.deepcopy(population[0])
                population=generatePopulation(population_size,nr_of_nodes,nr_of_colors)
                all_colors=[]
                time_array.append(time.time()-start_time)
                coloring_time_array.append(best_coloring_found)
                for i in range(nr_of_colors):
                      all_colors.append(i)
 
    if generation%25==0:
     with open(generation_file, 'a+') as gf:
      gf.seek(0, 2)
      text='Generation: {}, attempting coloring: {} , Best fitness: {}\n'.format(generation,nr_of_colors,best_fitness)
      gf.write(text)
      coloring_array.append(best_coloring_found)
      generations_array.append(generation)

    """with open(generation_file_real, 'a+') as gg: # slows down execution too much
          gg.seek(0, 2)
          text='Generation: {}, attempting coloring: {} , Best fitness: {}\n'.format(generation,nr_of_colors,best_fitness)
          gg.write(text)"""

    generation=generation+1


 end_time=time.time()
 finish=int(end_time-start_time)
 

 plt.plot(generations_array, coloring_array)
 plt.xlabel('Generations')
 plt.ylabel('Best coloring')
 plt.title('Algorithm analysis')
 plot_name='{}_Algorithm_analysis.png'.format(file_name)
 plt.savefig('Solutions/GA/{}/{}'.format(file_name,plot_name))
 plt.close()

 plt.plot(time_array,coloring_time_array,)
 plt.xlabel('Time(s)')
 plt.ylabel('Best coloring')
 plt.title('Time analysis')
 plot_name='{}_Time_analysis.png'.format(file_name)
 plt.savefig('Solutions/GA/{}/{}'.format(file_name,plot_name))
 plt.close()

 color_list=generateColors(nr_of_nodes,best_chromosome);
 network = nx.Graph()
 nodes=list(range(1,nr_of_nodes))
 network.add_nodes_from(nodes)
 for edges in adjancy:
    network.add_edge(edges[0],edges[1])

 pos = layout_many_components(network, component_layout_func=nx.layout.circular_layout)
 nx.draw_networkx(network,pos,node_color=color_list,node_size=30,with_labels=False)
 plot_name='{}_Graph_coloring_expanded.png'.format(file_name)
 plt.savefig('Solutions/GA/{}/{}'.format(file_name,plot_name))
 plt.close()
 

 nx.draw_networkx(network,node_color=color_list,node_size=20,with_labels=False)
 plot_name='{}_Graph_coloring.png'.format(file_name)
 plt.savefig('Solutions/GA/{}/{}'.format(file_name,plot_name))
 plt.close()

 text_file='{} - Results.txt'.format(file_name)
 with open('Solutions/GA/{}/{}'.format(file_name,text_file), 'w') as f:
    sys.stdout = f 
    print('Problem:',file_name,"\n\n",'Nr of vertexes:',nr_of_nodes,"\n",'Nr of edges:',total_edges,"\n",'Max degree:',max_degree,"\n\n",'First valid coloring found:',
    coloring_time_array[0],"\n",'Best coloring found:',best_coloring_found,"\n\n",
    'Population size:',population_size,"\n",'Generations:',generations,"\n\n",'Time for first solution(s):',"{:.4f}".format(time_array[0]),"\n",'Time for the best solution(s):',"{:.4f}".format(time_array[-1]),"\n",'Total Execution time(s):',finish,"\n\n","Conflicts left for next coloring:",best_fitness)

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
            gao(filename,nr_of_vertex,200,2000)

main()  
 

 


