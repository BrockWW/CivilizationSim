import numpy as np

class City:
    def __init__(self, max_pop, city_dict, CivClass):
        # pulling city info from dictionary
        self.name = city_dict["name"]
        self.size = city_dict["size"]
        self.center = np.array(city_dict["city_center"])
        self.pop = city_dict["pop"]
        self.historical_pop = []
        # setting population max and value
        self.max_pop = max_pop * self.size**2
        # linking civilization class for tracking other pops
        self.CivClass = CivClass
        # create city grid
        self.city_area()

    def city_area(self):
        # creating grid for city size
        rows = np.arange(-1*self.size+1,self.size)
        cols = np.arange(-1*self.size+1,self.size)

        # creating meshgrid for each square in city
        ROWS, COLS = np.meshgrid(rows, cols)
        # aligning city grid with world grid
        self.coord_arr = np.vstack((ROWS.ravel(), COLS.ravel())).T + self.center

class Capital(City):
    def __init__(self, max_pop, city_dict, CivClass, center_size):
        # calling City __init__ function
        super().__init__(max_pop, city_dict, CivClass)
        # add additional attributes below

        # size of city center
        self.center_size = center_size
        # define capital center
        self.capital_center()

    def capital_center(self):
        # creating grid for city center
        rows = np.arange(-1*self.center_size+1,self.center_size)
        cols = np.arange(-1*self.center_size+1,self.center_size)

        # creating meshgrid for each square in city
        ROWS, COLS = np.meshgrid(rows, cols)
        # aligning city grid with world grid
        self.center_arr = np.vstack((ROWS.ravel(), COLS.ravel())).T + self.center