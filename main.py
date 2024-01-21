import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Heating_model:
    def __init__(self, params: dict):
        self.params = params
        self.partial_matrix = {}
        self.result_matrix = np.zeros((100,100))
        self.mask_matrix = np.zeros((100,100))
        self.indx = {"windows": 1, "walls": 2, "doors": 3, "radiators": 5}
        self.build_partial_matrix()
        self.build_result_matrix()
        self.draw_heatmap()

    def build_partial_matrix(self):
        for room in self.params["rooms"].keys():
            coordinates = self.params["rooms"][room]
            self.partial_matrix[room] = np.zeros((coordinates["row_max"]-coordinates["row_min"], coordinates["col_max"]-coordinates["col_min"]))
            self.partial_matrix[room][:] = coordinates["init_func"](self.partial_matrix[room])

    def build_result_matrix(self):
        for room in self.params["rooms"].keys():
            coordinates = self.params["rooms"][room]
            self.result_matrix[coordinates["row_min"]:coordinates["row_max"],coordinates["col_min"]:coordinates["col_max"]] = self.partial_matrix[room]
    def evolve_in_unit_timestep(self, dt: float):
        force_term_full = self.params["force_term"](self.params["domain"]["grid"],
                                                     self.params["current_time"],
                                                     self.mask_matrix)
        for key in self.params["windows"].keys():
            self.result_matrix[
                self.params["windows"][key]["row_min"]: self.params["windows"][key]["row_max"],
                self.params["windows"][key]["col_min"]: self.params["windows"][key]["col_max"]
            ] = self.params["window_temp"](self.params["current_time"])
        self.build_partial_matrix()
        for key in self.params["doors"].keys():
            self.result_matrix[
            self.params["windows"][key]["row_min"]: self.params["windows"][key]["row_max"],
            self.params["windows"][key]["col_min"]: self.params["windows"][key]["col_max"]
            ] = self.params["window_temp"](self.params["current_time"])
        self.build_partial_matrix()
        for key in self.params["rooms"].keys():
            self.result_matrix[
            self.params["windows"][key]["row_min"]: self.params["windows"][key]["row_max"],
            self.params["windows"][key]["col_min"]: self.params["windows"][key]["col_max"]
            ] = self.params["window_temp"](self.params["current_time"])
        self.build_result_matrix()

        self.params["current_time"] += dt
        return self

    def evolve(self, n_steps: int, dt: float):
        for _ in tqdm.tqdm(range(n_steps), desc="TIME STEPS"):
            self.evolve_in_unit_timestep(dt)
        return self

    def draw_heatmap(self):
        return self.result_matrix

def draw_houses(object_house):
    for key, val in object_house.params['rooms'].items():
        object_house.partial_matrix[key] = np.zeros((val['row_max'] - val['row_min'], val['col_max'] - val['col_min']))
        mask = object_house.params["masks"][key]
        object_house.partial_matrix[key][mask == 1] = 4
        object_house.result_matrix[val['row_min']:val['row_max'], val['col_min']:val['col_max']] = object_house.partial_matrix[key]
    for v in object_house.indx:
        for key, val in object_house.params[v].items():
            object_house.result_matrix[val['row_min']:val['row_max'], val['col_min']:val['col_max']] = object_house.indx[v]
    return object_house


if __name__ == "__main__":
    model_parameters = {
        "rooms": {
            "A1": {
                "row_min": 0, "row_max": 60, "col_min": 0, "col_max": 50,
                "init_func": lambda x: 298 + np.random.random(x.shape)
            },
            "A2": {
                "row_min": 0, "row_max": 60, "col_min": 50, "col_max": 62,
                "init_func": lambda x: 294 + np.random.random(x.shape)
            },
            "A3": {
                "row_min": 0, "row_max": 45, "col_min": 62, "col_max": 100,
                "init_func": lambda x: 285 + np.random.random(x.shape)
            },
            "A4": {
                "row_min": 60, "row_max": 100, "col_min": 0, "col_max": 57,
                "init_func": lambda x: 296 + np.random.random(x.shape)
            },
            "A5": {
                "row_min": 60, "row_max": 100, "col_min": 57, "col_max": 100,
                "init_func": lambda x: 297 + np.random.random(x.shape)
            },
            "A6": {
                "row_min": 45, "row_max": 60, "col_min": 62, "col_max": 100,
                "init_func": lambda x: 295 + np.random.random(x.shape)
            }
        },
        "doors": {
            "D1": {
                "row_min": 30, "row_max": 35, "col_min": 60, "col_max": 62
            },
            "D2": {
                "row_min": 45, "row_max": 50, "col_min": 48, "col_max": 50
            },
            "D3": {
                "row_min": 60, "row_max": 62, "col_min": 50, "col_max": 55
            },
            "D4": {
                "row_min": 60, "row_max": 62, "col_min": 55, "col_max": 60
            },
            "D5": {
                "row_min": 53, "row_max": 58, "col_min": 60, "col_max": 62
            }
        },
        "windows": {
            "W1": {
                "row_min": 30, "row_max": 50, "col_min": 0, "col_max": 2
            },
            "W2": {
                "row_min": 65, "row_max": 80, "col_min": 0, "col_max": 2
            },
            "W3": {
                "row_min": 98, "row_max": 100, "col_min": 20, "col_max": 30
            },
            "W4": {
                "row_min": 98, "row_max": 100, "col_min": 60, "col_max": 70
            }
        },
        "walls": {
            "R1": {
                "row_min": 0, "row_max": 30, "col_min": 0, "col_max": 2
            },
            "R2": {
                "row_min": 50, "row_max": 65, "col_min": 0, "col_max": 2
            },
            "R3": {
                "row_min": 80, "row_max": 100, "col_min": 0, "col_max": 2
            },
            "R4": {
                "row_min": 98, "row_max": 100, "col_min": 2, "col_max": 20
            },
            "R5": {
                "row_min": 98, "row_max": 100, "col_min": 30, "col_max": 60
            },
            "R6": {
                "row_min": 98, "row_max": 100, "col_min": 70, "col_max": 98
            },
            "R7": {
                "row_min": 0, "row_max": 100, "col_min": 98, "col_max": 100
            },
            "R8": {
                "row_min": 0, "row_max": 2, "col_min": 2, "col_max": 98
            },
            "R9": {
                "row_min": 2, "row_max": 30, "col_min": 60, "col_max": 62
            },
            "R10": {
                "row_min": 35, "row_max": 53, "col_min": 60, "col_max": 62
            },
            "R11": {
                "row_min": 45, "row_max": 47, "col_min": 62, "col_max": 98
            },
            "R12": {
                "row_min": 60, "row_max": 62, "col_min": 62, "col_max": 98
            },
            "R13": {
                "row_min": 58, "row_max": 62, "col_min": 60, "col_max": 62
            },
            "R14": {
                "row_min": 50, "row_max": 62, "col_min": 48, "col_max": 50
            },
            "R15": {
                "row_min": 2, "row_max": 45, "col_min": 48, "col_max": 50
            },
            "R16": {
                "row_min": 60, "row_max": 62, "col_min": 2, "col_max": 48
            },
            "R17": {
                "row_min": 62, "row_max": 98, "col_min": 55, "col_max": 57
            }
        },
        "radiators": {
            "L1": {
                "row_min": 2, "row_max": 3, "col_min": 54, "col_max": 58
            },
            "L2": {
                "row_min": 32, "row_max": 47, "col_min": 2, "col_max": 3
            },
            "L3": {
                "row_min": 97, "row_max": 98, "col_min": 22, "col_max": 28
            }
        },
        "domain": {
            "grid": np.meshgrid(np.linspace(-1, 1, 101), np.linspace(-1, 1, 101))[0], "dx": 1
        },
        "masks1": lambda x, t, mask: np.where(
            mask == 1, (np.sin(24 * t / 3600) ** 2 + 2) / 10, np.where(
                mask == 2, (np.sin(24 * t / 3600) ** 2 + 1) / 10, np.where(
                    mask == 3, (np.sin(24 * t / 3600) ** 2 + 1) / 10, 0
                )
            )
        ),
        "masks": {
            "A1": 1,
            "A2": 1,
            "A3": 0,
            "A4": 1,
            "A5": 1,
            "A6": 1
        },
        "window_temp": lambda t: 280 - 10 * np.sin(24 * t / 3600),
        "diffusion": 0.1,
        "current_time": 0.0
    }

    cmap = ListedColormap(['white', 'lightskyblue', 'black', 'peachpuff', 'floralwhite', 'crimson'])
    model = Heating_model(model_parameters)
    model_c = Heating_model(model_parameters)

    draw_house = draw_houses(model_c)
    plt.imshow(draw_house.result_matrix, cmap=cmap)
    plt.show()

    plt.imshow(model.draw_heatmap(), cmap=plt.get_cmap("coolwarm"))
    plt.title(f"t = {model.params['current_time']}")
    plt.colorbar().set_label("Temperature[K]")
    plt.show()
