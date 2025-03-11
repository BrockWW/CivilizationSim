import numpy as np

class Person:
    def __init__(self, location, id, class_stats, CivClass):
        # assigning initial variables
        self.alive = True
        self.age = 0
        self.num_children = 0
        self.num_attack = 0
        self.num_killed = 0
        self.loc = location
        self.id = id
        self.last_move = np.array([np.nan,np.nan])
        # assigning stats
        self.pop_class = class_stats["class"]
        self.health = class_stats["health"]
        self.physical_strength = class_stats["physical_strength"]
        self.magic_strength = class_stats["magic_strength"]
        self.popularity = class_stats["popularity"]
        self.lifespan = class_stats["lifespan"]
        # linking civilization class for tracking other pops
        self.CivClass = CivClass
        # initial in city flag
        self.in_city = False
        self.current_city = None
        # pulling city centers and starting city
        self.city_coords = np.empty((0,2))
        for city in self.CivClass.cities:
            self.city_coords = np.concatenate([self.city_coords,city.coord_arr], axis = 0)
            if any(np.equal(city.coord_arr,self.loc).all(1)):
                self.current_city = city
                self.in_city = True
        # initial like array
        self.like_arr = np.ones(CivClass.pop)*0.5

    def move(self):
        # checking legal moves
        possible_moves = self.check_surroundings()
        
        if not self.in_city:
            # finding minimum distance to city coordinates
            city_dist = np.linalg.norm(np.array(self.loc)-np.array(self.city_coords), axis = 1)
            min_index = np.where(city_dist == np.min(city_dist))[0]
            # randomly choosing point if equidistant from multiple points
            choice_index = np.random.choice(min_index)

            # finding distance to nearest city
            x_diff = self.city_coords[choice_index][1] - self.loc[1]
            y_diff = self.city_coords[choice_index][0] - self.loc[0]

            # weigthing movements based on direction of nearest city
            if x_diff >=0 and y_diff >= 0:
                move_prob_dict = {"--":[[-1, -1], 1/24],
                                  "0-":[[0, -1], 1/24],
                                  "+-":[[1, -1], 1/24],
                                  "-0":[[-1, 0], 1/24],
                                  "00":[[0, 0], 1/24],
                                  "+0":[[1, 0], 1/24],
                                  "-+":[[-1, 1], 1/4],
                                  "0+":[[0, 1], 1/4],
                                  "++":[[1, 1], 1/4]}
            elif x_diff < 0 and y_diff >= 0:
                move_prob_dict = {"--":[[-1, -1], 1/24],
                                  "0-":[[0, -1], 1/4],
                                  "+-":[[1, -1], 1/4],
                                  "-0":[[-1, 0], 1/24],
                                  "00":[[0, 0], 1/24],
                                  "+0":[[1, 0], 1/4],
                                  "-+":[[-1, 1], 1/24],
                                  "0+":[[0, 1], 1/24],
                                  "++":[[1, 1], 1/24]}
            elif x_diff >= 0 and y_diff < 0:
                move_prob_dict = {"--":[[-1, -1], 1/24],
                                  "0-":[[0, -1], 1/24],
                                  "+-":[[1, -1], 1/24],
                                  "-0":[[-1, 0], 1/4],
                                  "00":[[0, 0], 1/24],
                                  "+0":[[1, 0], 1/24],
                                  "-+":[[-1, 1], 1/4],
                                  "0+":[[0, 1], 1/4],
                                  "++":[[1, 1], 1/24]}
            elif x_diff < 0 and y_diff < 0:
                move_prob_dict = {"--":[[-1, -1], 1/4],
                                  "0-":[[0, -1], 1/4],
                                  "+-":[[1, -1], 1/24],
                                  "-0":[[-1, 0], 1/4],
                                  "00":[[0, 0], 1/24],
                                  "+0":[[1, 0], 1/24],
                                  "-+":[[-1, 1], 1/24],
                                  "0+":[[0, 1], 1/24],
                                  "++":[[1, 1], 1/24]}
            
            # weight possible movements
            weight_list = []
            for movement in possible_moves:
                for key in move_prob_dict.keys():
                    if np.equal(movement, move_prob_dict[key][0]).all():
                        weight_list.append(move_prob_dict[key][1])
            weight_list = np.array(weight_list)+(1-np.sum(weight_list))/len(weight_list)

        # while loop to check for bordering citizens and move
        flag = True
        while flag:
            # weight movement if out of city
            if not self.in_city:
                index = np.random.choice(np.arange(len(possible_moves)), size = 1, p = weight_list)
                move = possible_moves[index][0]
            else:
                # randomly move
                index = np.random.randint(len(possible_moves))
                move = possible_moves[index]

            next_pos = np.array([self.loc[0]+move[0], self.loc[1]+move[1]])
            # check if next space is not occupied by different citizen, else remove that possible move
            if (any(np.equal(self.CivClass.pop_locs,next_pos).all(1)) and (self.loc == next_pos).all()) or not any(np.equal(self.CivClass.pop_locs,next_pos).all(1)):
                self.loc = next_pos
                self.last_move = move
                self.CivClass.pop_locs[self.id] = next_pos
                self.age += 1
                flag = False

                for city in self.CivClass.cities:
                    # pop in city coords and current_city is selected city
                    if any(np.equal(city.coord_arr,self.loc).all(1)) and self.current_city == city:
                        self.in_city = True
                        break
                    # pop in city coords and current_city is not selected city
                    elif any(np.equal(city.coord_arr,self.loc).all(1)) and self.current_city != city:
                        self.in_city = True
                        self.current_city = city
                        self.current_city.pop += 1
                        break
                    # pop in city coords and current_city is selected city
                    elif any(np.equal(city.coord_arr,self.loc).all(1)) and self.current_city == city:
                        self.in_city = True
                        break
                    # pop not in city coords and current_city is selected city
                    elif not any(np.equal(city.coord_arr,self.loc).all(1)) and self.current_city == city:
                        self.current_city.pop -= 1
                        self.current_city = None
                        self.in_city = False
                        break
                    # pop not in city coords and current_city is not selected city and selected city is the last city in the list
                    elif not any(np.equal(city.coord_arr,self.loc).all(1)) and self.current_city != city and city == self.CivClass.cities[-1]:
                        self.current_city = None
                        self.in_city = False

            else:
                possible_moves = np.delete(possible_moves, index, axis = 0)
                if not self.in_city:
                    weight = weight_list[index]
                    weight_list = np.delete(weight_list, index)
                    # fix unbalanced weight list
                    weight_list += weight/len(weight_list)
                
    def check_surroundings(self):
        # setting movement constraints
        row_l = -1
        row_h = 1
        col_l = -1
        col_h = 1
        # check row boundary
        if self.loc[0] == self.CivClass.grid_size[0]-1:
            row_h = 0
        elif self.loc[0] == 0:
            row_l = 0
        # check column boundary
        if self.loc[1] == self.CivClass.grid_size[1]-1:
            col_h = 0
        elif self.loc[1] == 0:
            col_l = 0

        # creating list of possible moves
        row_moves = np.arange(row_l,row_h+1)
        col_moves = np.arange(col_l,col_h+1)

        ROW_MOVES, COL_MOVES = np.meshgrid(row_moves, col_moves)
        possible_moves = np.vstack((ROW_MOVES.ravel(),COL_MOVES.ravel())).T

        return possible_moves
    
    def update_like_arr(self):
        # updates like array of other citizens when close
        neighboring_positions = self.check_surroundings()+np.array(self.loc)
        for pos in neighboring_positions:
            # only update like array if a neighbor is located in adjacent position, neighboring position is not current position, and last move was not stand still
            # the not standing still condition will cut down on explosive growth when pops get trapped by other pops
            #print(self.CivClass.pop_locs)
            #print(pos)
            #print(self.loc)
            #print(self.last_move)
            #print(any(np.equal(self.CivClass.pop_locs,pos).all(1)))
            #print((self.loc == pos).all())
            #print((self.last_move == np.array([0,0])).all())
            if any(np.equal(self.CivClass.pop_locs,pos).all(1)) and not (self.loc == pos).all() and not (self.last_move == np.array([0,0])).all():
                # determine id of neighbor
                neighbor_id = np.where((self.CivClass.pop_locs == pos).all(1))[0][0]
                neighbor = self.CivClass.citizens[neighbor_id]
                
                # add modifier based on class
                mod = 0
                if self.pop_class == neighbor.pop_class:
                    mod += 0.05
                elif self.pop_class != neighbor.pop_class:
                    mod -= 0.05*(10/neighbor.popularity)
                # add modifier based on if neighbor is already liked or not
                if self.like_arr[neighbor_id] > 0.5:
                    mod = 0.05
                elif self.like_arr[neighbor_id] < 0.5:
                    mod = -0.05

                # change like status with modifier
                like_change = 0.01*np.random.randint(-10,25)+mod

                self.like_arr[neighbor_id] += like_change

                # attack person
                if np.round(self.like_arr[neighbor_id],2) <= 0:
                    self.like_arr[neighbor_id] == 0.5
                    self.attack_person(neighbor_id)
                # have child while under pop thresh - out of city
                elif np.round(self.like_arr[neighbor_id],2) >= 1 and self.CivClass.current_alive <= self.CivClass.world_max_pop*np.prod(self.CivClass.grid_size) and not self.in_city:
                    self.like_arr[neighbor_id] == 0.5
                    # low chance to have child outside of city (balance world border out of control spawning)
                    if np.random.rand() > 0.80:
                        self.create_child()
                # have child while under pop thresh - inside of city
                elif np.round(self.like_arr[neighbor_id],2) >= 1 and self.CivClass.current_alive <= self.CivClass.world_max_pop*np.prod(self.CivClass.grid_size) and self.in_city and self.current_city.pop <= self.current_city.max_pop:
                    self.like_arr[neighbor_id] == 0.5
                    self.create_child()
                # like condition for over pop thresh
                elif np.round(self.like_arr[neighbor_id],2) >= 1 and (self.CivClass.current_alive > self.CivClass.world_max_pop*np.prod(self.CivClass.grid_size) or (self.in_city and self.current_city.pop > self.current_city.max_pop)):
                    self.like_arr[neighbor_id] == 0.5

        # update civilization like array
        self.CivClass.citizens_relation[self.id] = self.like_arr

    def create_child(self):
        # like rating of another person goes to 1, create new person
        open_squares = self.check_surroundings()
        # remove parent location from consideration of child location
        parent_index = np.where((open_squares == np.array([0,0])).all(1))[0][0]
        open_squares = np.delete(open_squares, parent_index, axis = 0)
        # while loop to check for bordering citizens
        flag = True
        while flag:
            if len(open_squares) == 0:
                # if all surrounding squares are full
                flag = False
            else:
                index = np.random.randint(len(open_squares))
                place_child = open_squares[index]
                start_loc = np.array([self.loc[0]+place_child[0],self.loc[1]+place_child[1]])
                # check if next space is occupied by different citizen, if so remove that possible move
                if not any(np.equal(self.CivClass.pop_locs,start_loc).all(1)):
                    flag = False
                    self.num_children += 1
                    self.CivClass.births += 1
                    self.CivClass.pop += 1
                    self.CivClass.add_person(start_loc)

                    # adding to pop if child spawns in city
                    for city in self.CivClass.cities:
                        if any(np.equal(city.coord_arr,start_loc).all(1)):
                            city.pop += 1

                    # updating all existing like arrays
                    for i in range(len(self.CivClass.citizens_relation)-1):
                        self.CivClass.citizens_relation[i] = np.append(self.CivClass.citizens_relation[i],0.5)
                        self.CivClass.citizens[i].like_arr = self.CivClass.citizens_relation[i]
                else:
                    open_squares = np.delete(open_squares, index, axis = 0)
     
    def attack_person(self, kill_id):
        # like rating of another person goes to 0, attack person
        self.num_attack += 1
        neighbor = self.CivClass.citizens[kill_id]
        neighbor.health -= (self.physical_strength+self.magic_strength)
        # checking for death
        if neighbor.health <= 0:
            self.like_arr[kill_id] = 0
            self.kill_death(kill_id)

    def kill_death(self, kill_id):
        # like rating and health of another person goes to 0, kill person
        self.num_killed += 1
        self.CivClass.deaths += 1
        self.CivClass.citizens[kill_id].health = 0
        self.CivClass.citizens[kill_id].alive = False
        # remove pop from world
        dead_pos = np.array([np.nan, np.nan])
        self.CivClass.citizens[kill_id].loc = dead_pos
        self.CivClass.pop_locs[kill_id] = dead_pos
        # update city pops
        if self.CivClass.citizens[kill_id].current_city != None:
            self.CivClass.citizens[kill_id].current_city.pop -= 1
        # update class pops
        self.CivClass.world_current_class_dict[self.CivClass.citizens[kill_id].pop_class] -= 1

    def rand_death(self):
        # collecting random death chances together
        self.age_death()
        if not self.in_city and self.alive:
            self.wilderness_death()

    def age_death(self):
        # randomly die of age - sigmoid with age where chance is 50% to die at lifespan
        rand_val = np.random.random()
        death_chance = 1 / (1 + np.exp(-(self.age-self.lifespan)))
        if rand_val < death_chance:
            self.health = 0
            self.alive = False
            self.CivClass.deaths += 1
            # remove pop from world
            dead_pos = np.array([np.nan, np.nan])
            self.loc = dead_pos
            self.CivClass.pop_locs[self.id] = dead_pos
            # update city pops
            if self.current_city != None:
                self.current_city.pop -= 1
            # update class pops
            self.CivClass.world_current_class_dict[self.pop_class] -= 1

    def wilderness_death(self):
        # die when out of city chance
        rand_val = np.random.random()
        if rand_val > 0.95:
            self.health = 0
            self.alive = False
            self.CivClass.deaths += 1
            # remove pop from world
            dead_pos = np.array([np.nan, np.nan])
            self.loc = dead_pos
            self.CivClass.pop_locs[self.id] = dead_pos
            # update class pops
            self.CivClass.world_current_class_dict[self.pop_class] -= 1


#################################################
# King Class
#################################################
class King(Person):
    def __init__(self, location, id, class_stats, CivClass):
        # call Person __init__ function
        super().__init__(location, id, class_stats, CivClass)
        # add additional attributes below

        # defining capital center array
        capital = [city for city in self.CivClass.cities if city.name == "capital"][0]
        self.capital_center = capital.center_arr

    # overwrite Person check_surroundings to focus on capital center
    def check_surroundings(self):
        # setting movement constraints
        # king will not move out of capital center once arrived
        row_l = -1
        row_h = 1
        col_l = -1
        col_h = 1
        # check row boundary for capital center and world border
        if self.loc[0] == np.max(self.capital_center[0]):
            row_h = 0
        elif self.loc[0] == np.min(self.capital_center[0]):
            row_l = 0
        elif self.loc[0] == self.CivClass.grid_size[0]-1:
            row_h = 0
        elif self.loc[0] == 0:
            row_l = 0
        # check column boundary for capital center and world border
        if self.loc[1] == np.max(self.capital_center[1]):
            col_h = 0
        elif self.loc[1] == np.min(self.capital_center[1]):
            col_l = 0
        elif self.loc[1] == self.CivClass.grid_size[1]-1:
            col_h = 0
        elif self.loc[1] == 0:
            col_l = 0

        # creating list of possible moves
        row_moves = np.arange(row_l,row_h+1)
        col_moves = np.arange(col_l,col_h+1)

        ROW_MOVES, COL_MOVES = np.meshgrid(row_moves, col_moves)
        possible_moves = np.vstack((ROW_MOVES.ravel(),COL_MOVES.ravel())).T

        return possible_moves

    # overwrite Person move for king to move towards capital center and stay in vicinity
    def move(self):
        # comparing current location to capital center
        if not any(np.equal(self.capital_center,self.loc).all(1)):
            capital_center_dist = np.linalg.norm(np.array(self.loc)-self.capital_center, axis = 1)
            min_index = np.where(capital_center_dist == np.min(capital_center_dist))[0]
            # randomly choosing point if equidistant from multiple points
            choice_index = np.random.choice(min_index)

            # initializing possible moves allowed for when king is outside of city, will update based on directional weights
            possible_moves = np.empty((0,2))

            # finding distance to capital center
            x_diff = self.capital_center[choice_index][1] - self.loc[1]
            y_diff = self.capital_center[choice_index][0] - self.loc[0]

            # weigthing movements based on direction of capital center
            if x_diff >= 0 and y_diff >= 0:
                move_prob_dict = {"-+":[np.array([-1, 1]), 1/3],
                                  "0+":[np.array([0, 1]), 1/3],
                                  "++":[np.array([1, 1]), 1/3]}
            elif x_diff < 0 and y_diff >= 0:
                move_prob_dict = {"0-":[np.array([0, -1]), 1/3],
                                  "+-":[np.array([1, -1]), 1/3],
                                  "+0":[np.array([1, 0]), 1/3]}
            elif x_diff >= 0 and y_diff < 0:
                move_prob_dict = {"-0":[np.array([-1, 0]), 1/3],
                                  "-+":[np.array([-1, 1]), 1/3],
                                  "0+":[np.array([0, 1]), 1/3]}
            elif x_diff < 0 and y_diff < 0:
                move_prob_dict = {"--":[np.array([-1, -1]), 1/3],
                                  "0-":[np.array([0, -1]), 1/3],
                                  "-0":[np.array([-1, 0]), 1/3]}
            
            # weight possible movements
            weight_arr = np.empty((0))
            for key in move_prob_dict.keys():
                possible_moves = np.vstack([possible_moves, move_prob_dict[key][0]])
                weight_arr = np.append(weight_arr, np.array([move_prob_dict[key][1]]))
            weight_arr = np.array(weight_arr)+(1-np.sum(weight_arr))/len(weight_arr)
        
        elif any(np.equal(self.capital_center,self.loc).all(1)):
            # checking legal moves
            possible_moves = self.check_surroundings()

        # while loop to check for bordering citizens and move
        flag = True
        while flag:
            # weight movement if out of capital center
            if not any(np.equal(self.capital_center,self.loc).all(1)) and len(possible_moves) != 0:
                index = np.random.choice(np.arange(len(possible_moves)), size = 1, p = weight_arr)
                move = possible_moves[index][0]
            # stay in place only if unable to move toward capital center
            elif not any(np.equal(self.capital_center,self.loc).all(1)) and len(possible_moves) == 0:
                move = np.array([0,0])
            else:
                # randomly move
                index = np.random.randint(len(possible_moves))
                move = possible_moves[index]

            next_pos = np.array([self.loc[0]+move[0], self.loc[1]+move[1]])
            # check if next space is not occupied by different citizen, else remove that possible move
            if (any(np.equal(self.CivClass.pop_locs,next_pos).all(1)) and (self.loc == next_pos).all()) or not any(np.equal(self.CivClass.pop_locs,next_pos).all(1)):
                self.loc = next_pos
                self.last_move = move
                self.CivClass.pop_locs[self.id] = next_pos
                self.age += 1
                flag = False

                for city in self.CivClass.cities:
                    # pop in city coords and current_city is selected city
                    if any(np.equal(city.coord_arr,self.loc).all(1)) and self.current_city == city:
                        self.in_city = True
                        break
                    # pop in city coords and current_city is not selected city
                    elif any(np.equal(city.coord_arr,self.loc).all(1)) and self.current_city != city:
                        self.in_city = True
                        self.current_city = city
                        self.current_city.pop += 1
                        break
                    # pop not in city coords and current_city is selected city
                    elif not any(np.equal(city.coord_arr,self.loc).all(1)) and self.current_city == city:
                        self.current_city.pop -= 1
                        self.current_city = None
                        self.in_city = False
                        break
                    # pop not in city coords and current_city is not selected city and selected city is the last city in the list
                    elif not any(np.equal(city.coord_arr,self.loc).all(1)) and self.current_city != city and city == self.CivClass.cities[-1]:
                        self.current_city = None
                        self.in_city = False

            else:
                possible_moves = np.delete(possible_moves, index, axis = 0)
                if not any(np.equal(self.capital_center,self.loc).all(1)) and len(possible_moves) != 0:
                    weight = weight_arr[index]
                    weight_arr = np.delete(weight_arr, index)
                    # fix unbalanced weight list
                    weight_arr += weight/len(weight_arr)


#################################################
# Peasant Class
#################################################
class Peasant(Person):
    def __init__(self, location, id, class_stats, CivClass):
        # call Person __init__ function
        super().__init__(location, id, class_stats, CivClass)
        # add additional attributes below

        # defining capital center array
        capital = [city for city in self.CivClass.cities if city.name == "capital"][0]
        self.capital_center = capital.center_arr

    # overwrite Person check_surroundings to stay out of capital center
    def check_surroundings(self):
        # setting movement constraints
        # peasant will not move inside of capital center
        row_l = -1
        row_h = 1
        col_l = -1
        col_h = 1
        # check row boundary for capital center and world border
        if self.loc[0] == np.min(self.capital_center[0]):
            row_h = 0
        elif self.loc[0] == np.max(self.capital_center[0]):
            row_l = 0
        elif self.loc[0] == self.CivClass.grid_size[0]-1:
            row_h = 0
        elif self.loc[0] == 0:
            row_l = 0
        # check column boundary for capital center and world border
        if self.loc[1] == np.min(self.capital_center[1]):
            col_h = 0
        elif self.loc[1] == np.max(self.capital_center[1]):
            col_l = 0
        elif self.loc[1] == self.CivClass.grid_size[1]-1:
            col_h = 0
        elif self.loc[1] == 0:
            col_l = 0
        
        # creating list of possible moves
        row_moves = np.arange(row_l,row_h+1)
        col_moves = np.arange(col_l,col_h+1)

        ROW_MOVES, COL_MOVES = np.meshgrid(row_moves, col_moves)
        possible_moves = np.vstack((ROW_MOVES.ravel(),COL_MOVES.ravel())).T

        return possible_moves