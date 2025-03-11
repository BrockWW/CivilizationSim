import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pop_class import *
from city_class import *

class Civilization:
    def __init__(self, run_t = 50, world_max_pop = 0.5, cities_max_pop = 0.75, grid_size = [10,10], world_class_dict = {}, world_cities_dict = {}, world_max_class_dict = {}):
        # initialize pop
        self.pop = 0
        # fraction of grid representing maximum population
        self.world_max_pop = world_max_pop
        # max class population and current class pop set to 0
        self.world_max_class_dict = world_max_class_dict
        self.world_current_class_dict = world_max_class_dict.copy()
        for key in self.world_current_class_dict.keys():
            self.world_current_class_dict[key] = 0
        # alive population
        self.pop_alive = []
        # city size, row by column
        self.grid_size = grid_size
        # location of all pops
        self.pop_locs = []
        # creating cities
        self.cities_max_pop = cities_max_pop
        self.cities = []
        self.create_cities(world_cities_dict)
        # civ run time
        self.run_t = run_t
        # initialize list of all citizens
        self.citizens = []
        # initialize list of relations between pops
        self.citizens_relation = []
        # initialize count for births and deaths
        self.births = 0
        self.deaths = 0
        # initialize list for number of births and deaths per time step
        self.num_births = []
        self.num_deaths = []
        # initalize number of classes
        self.historical_class_count = []
        # dict of citizen world classes and colormap
        self.classes = world_class_dict
        pop_class = list(world_class_dict.keys())
        pop_probs = []
        pop_colors = []

        for key in pop_class:
            pop_probs.append(world_class_dict[key]["probability"])
            pop_colors.append(world_class_dict[key]["color"])

        self.pop_dist_dict = dict(zip(pop_class, pop_probs))
        self.pop_color_dict = dict(zip(pop_class, pop_colors))

        self.world_city_coords = np.empty((0,2))
        # creating initial population
        for city in self.cities:
            # setting pop starting locations
            self.world_city_coords = np.vstack((self.world_city_coords, city.coord_arr))
            if city.name == "capital":
                self.capital_center = city.center_arr
        for i in range(self.pop):
            self.add_person(self.world_city_coords)
    
    def add_person(self, open_locations):
        id = len(self.citizens)
        # setting up probabilities and classes based on maximum class constraints
        reg_classes_prob = []
        special_classes_prob = []
        reg_classes = []
        special_classes = []
        for key in self.world_max_class_dict:
            if self.world_max_class_dict[key] > self.world_current_class_dict[key] and not self.classes[key]["special"]:
                reg_classes_prob.append(self.pop_dist_dict[key])
                reg_classes.append(key)
            elif self.world_max_class_dict[key] > self.world_current_class_dict[key] and self.classes[key]["special"]:
                special_classes_prob.append(self.pop_dist_dict[key])
                special_classes.append(key)
        # adjusting total probability of regular classes
        reg_classes_prob = reg_classes_prob + ((1-np.sum(reg_classes_prob))/len(reg_classes_prob)) - np.sum(special_classes_prob)
        classes_prob = np.concatenate([reg_classes_prob, special_classes_prob])
        classes = reg_classes + special_classes
        # assigning class
        world_class = np.random.choice(classes, size = 1, p = classes_prob)[0]
        # updating current class count
        self.world_current_class_dict[world_class] += 1
        
        if(world_class == "black_swordsman"):
            print("\rSTARBURST STREAM!!!\n")
        # creating new citizen
        if world_class == "king":
            if len(open_locations.shape) == 2 and open_locations.shape[0] > 1:
                index = np.random.choice(np.arange(open_locations.shape[0]), 1)
                location = open_locations[index].flatten()
            else:
                location = open_locations.flatten()
            new_citizen = King(location, id, self.classes[world_class], self)
            self.citizens.append(new_citizen)
            self.citizens_relation.append(new_citizen.like_arr)
        elif world_class == "peasant":
            # checking if open location array is 2d or not
            if len(open_locations.shape) == 2 and open_locations.shape[0] > 1:
                rem_mask = np.all(~np.isin(open_locations, self.capital_center), axis=1)
                peasant_open_locs = open_locations[rem_mask]
                if len(peasant_open_locs.shape) == 2 and peasant_open_locs.shape[0] > 1:
                    index = np.random.choice(np.arange(peasant_open_locs.shape[0]), 1)
                    location = peasant_open_locs[index].flatten()
                elif len(open_locations.shape) == 2 and len(peasant_open_locs) == 2:
                    location = peasant_open_locs.flatten()
            else:
                rem_mask = np.all(~np.equal(open_locations, self.capital_center))
                peasant_open_locs = open_locations[rem_mask]
                if len(peasant_open_locs) == 0:
                    # no valid space to spawn pop
                    return
                else:
                    location = peasant_open_locs.flatten()
            new_citizen = Peasant(location, id, self.classes[world_class], self)
            self.citizens.append(new_citizen)
            self.citizens_relation.append(new_citizen.like_arr)
        else:
            if len(open_locations.shape) == 2 and open_locations.shape[0] > 1:
                index = np.random.choice(np.arange(open_locations.shape[0]), 1)
                location = open_locations[index].flatten()
            else:
                location = open_locations.flatten()
            new_citizen = Person(location, id, self.classes[world_class], self)
            self.citizens.append(new_citizen)
            self.citizens_relation.append(new_citizen.like_arr)

        # append new pop location
        self.pop_locs.append(location)

        # removing occupied location from consideration
        if len(open_locations.shape) == 2:
            rem_mask = np.all(~np.isin(open_locations, location), axis=1)
            open_locations = open_locations[rem_mask]
        # if current element is last element do not create new pop as viable location not available
        # will happen for a peasant trying to be born in capital center
        else:
            return

    def create_cities(self, cities_dict):
        for key in cities_dict.keys():
            if key == "capital":
                self.cities.append(Capital(self.cities_max_pop, cities_dict[key], self, cities_dict[key]["center_size"]))
                self.pop += cities_dict[key]["pop"]
            else:
                self.cities.append(City(self.cities_max_pop, cities_dict[key], self))
                self.pop += cities_dict[key]["pop"]
    
    def draw_grid(self, pop_locations):
        # initialize current step city
        world = pop_locations
        alive = len(np.where(pop_locations != 0)[0])
        # create colormap
        city_cmap = colors.ListedColormap(["white"]+list(self.pop_color_dict.values()))
        
        # creating colorbar labels
        bounds = np.arange(len(list(self.pop_color_dict.values()))+1)
        labels = ["Empty"]+list(self.pop_color_dict.keys())
        lin_norm = colors.Normalize(vmin = -1, vmax = len(labels))
        
        # draw city
        fig, ax = plt.subplots()
        world_map = ax.imshow(world, norm = lin_norm, cmap = city_cmap)
        # including colorbar for pop label
        cbar = fig.colorbar(world_map, norm = lin_norm, cmap = city_cmap, pad = 0)
        cbar.ax.set_yticklabels([])
        cbar.ax.tick_params(length = 0)
        # Create custom colorbar labels
        for i, bound in enumerate(bounds):
            name = labels[i].split(sep = "_")
            capitalized_name = [letter[0].upper() + letter[1:].lower() for letter in name]
            if len(capitalized_name) > 1:
                adj_pad = 0.4 - 0.1*len(capitalized_name)
            else:
                adj_pad = 0.4
            label = "\n ".join(capitalized_name)
            cbar.ax.text(0.95, (bound+adj_pad)*(len(labels)/(len(labels)-1))-1, " "+label)
        
        # drawing cities
        for city in self.cities:
            self.draw_city(ax, city)
        
        # updating currently alive pops
        self.pop_alive.append(alive)
    
    def draw_city(self, ax, city):
        # Define the selected area
        center = city.center
        size = city.size
        lwidth = 1
        col = "red"

        # choosing x/y size of city
        x_low = center[1] - size
        x_high = center[1] + size
        y_low = center[0] - size
        y_high = center[0] + size

        # drawing lines around city cells
        ax.hlines(y_low-0.5, x_low-0.5, x_high+0.5, color=col, linewidth=lwidth)
        ax.hlines(y_high+0.5, x_low-0.5, x_high+0.5, color=col, linewidth=lwidth)

        ax.vlines(x_low-0.5, y_low-0.5, y_high+0.5, color=col, linewidth=lwidth)
        ax.vlines(x_high+0.5, y_low-0.5, y_high+0.5, color=col, linewidth=lwidth)

    def run_civilization(self):
        max_chars = len(str(self.run_t-1))
        color_index_dict = dict(zip(list(self.pop_color_dict.keys()), np.arange(1,len(list(self.pop_color_dict.keys()))+1)))
        for t in range(self.run_t):
            print("Simulation Completion:", np.round(100*(t+1)/self.run_t, 1), "%", end="\r")
            # save number of births and deaths
            self.num_births.append(self.births)
            self.num_deaths.append(self.deaths)
            self.historical_class_count.append(np.array(list(self.world_current_class_dict.values())))
            # saving population of cities
            for city in self.cities:
                city.historical_pop.append(city.pop)
            # reset births and deaths for next step
            self.births = 0
            self.deaths = 0
            # initialize color coded civ grid for GIF
            civ_grid = np.zeros(self.grid_size)
            # creating string numbers for ordered images
            str_t = str(t)
            if len(str_t) < max_chars:
                str_t = int(max_chars-len(str_t))*"0"+str_t
            iteration = 0
            for person in self.citizens:
                if(len(self.pop_alive) == 0):
                    self.current_alive = self.pop
                else:
                    self.current_alive = self.pop_alive[-1] + iteration
                if person.alive:
                    iteration += 1
                    civ_grid[int(person.loc[0]),int(person.loc[1])] = color_index_dict[person.pop_class]
                    person.move()
                    person.update_like_arr()
                    person.rand_death()
            self.draw_grid(civ_grid)
            plt.title("Time: "+str_t)
            plt.savefig("civ_ims/time_"+str_t+".png")
            plt.close()