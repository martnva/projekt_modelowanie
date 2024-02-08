import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import tqdm


class HeatingModel:
    def __init__(self, params: dict):
        self.params = params
        self.partial_matrix = {}
        self.result_matrix = np.zeros((100, 100))
        self.mask_matrix = np.zeros((100, 100))
        self.indx = {"windows": 1, "walls": 2, "doors": 3, "radiators": 5}
        self.build_partial_matrix()
        self.build_result_matrix()
        self.build_mask_matrix()
        self.draw_heatmap()
        self.heatingData = []
        self.meanTemp = {}

    def build_partial_matrix(self):
        for room in self.params["rooms"].keys():
            coordinates = self.params["rooms"][room]
            if room not in self.partial_matrix.keys():
                self.partial_matrix[room] = np.zeros((coordinates["row_max"]-coordinates["row_min"], coordinates["col_max"]-coordinates["col_min"]))
                self.partial_matrix[room] = coordinates["init_func"](self.partial_matrix[room])
            else:
                self.partial_matrix[room] = self.result_matrix[coordinates["row_min"]: coordinates["row_max"],
                                            coordinates["col_min"]: coordinates["col_max"]]

    def build_result_matrix(self):
        for room in self.params["rooms"].keys():
            coordinates = self.params["rooms"][room]
            self.result_matrix[coordinates["row_min"]:coordinates["row_max"], coordinates["col_min"]:coordinates["col_max"]] = self.partial_matrix[room]

    def build_mask_matrix(self):
        for radiators in self.params["radiators"].keys():
            coordinates = self.params["radiators"][radiators]
            self.mask_matrix[coordinates["row_min"]:coordinates["row_max"], coordinates["col_min"]:coordinates["col_max"]] = coordinates["maskValue"]

    def evolve_in_unit_timestep(self, dt: float):
        coefficient = self.params["diffusion_coeff"] * dt / self.params["domain"]["dx"] ** 2
        force_term_full = self.params["force_term"](self.params["domain"]["grid"],
                                                    self.params["current_time"],
                                                    self.mask_matrix)
        for key in self.params["windows"].keys():
            self.result_matrix[
                self.params["windows"][key]["row_min"]: self.params["windows"][key]["row_max"],
                self.params["windows"][key]["col_min"]: self.params["windows"][key]["col_max"]
            ] = self.params["window_temp"](self.params["current_time"])
        self.build_partial_matrix()
        for key in self.params["rooms"].keys():
            if np.mean(self.partial_matrix[key]) > self.params["rooms"][key]["des"]:
                force_term_full[
                    self.params["rooms"][key]["row_min"] + 1: self.params["rooms"][key]["row_max"] - 1,
                    self.params["rooms"][key]["col_min"] + 1: self.params["rooms"][key]["col_max"] - 1
                ] = 0
            self.partial_matrix[key][1:-1, 1:-1] += coefficient * (self.partial_matrix[key][0:-2, 1:-1] +
                                                     self.partial_matrix[key][2:, 1:-1] +
                                                     self.partial_matrix[key][1:-1, 0:-2] +
                                                     self.partial_matrix[key][1:-1, 2:] -
                                                     4*self.partial_matrix[key][1:-1, 1:-1]) + \
                                                    force_term_full[
                                                        self.params["rooms"][key]["row_min"]+1: self.params["rooms"][key]["row_max"]-1,
                                                        self.params["rooms"][key]["col_min"]+1: self.params["rooms"][key]["col_max"]-1
                                                    ]
            self.partial_matrix[key][0, :] = self.partial_matrix[key][1, :]
            self.partial_matrix[key][-1, :] = self.partial_matrix[key][-2, :]
            self.partial_matrix[key][:, 0] = self.partial_matrix[key][:, 1]
            self.partial_matrix[key][:, -1] = self.partial_matrix[key][:, -2]
        self.build_result_matrix()

        for key in self.params["doors"].keys():
            self.result_matrix[
                self.params["doors"][key]["row_min"]: self.params["doors"][key]["row_max"],
                self.params["doors"][key]["col_min"]: self.params["doors"][key]["col_max"]
            ] = np.mean(self.result_matrix[
                            self.params["doors"][key]["row_min"]: self.params["doors"][key]["row_max"],
                            self.params["doors"][key]["col_min"]: self.params["doors"][key]["col_max"]
                        ]
                        )
        self.build_partial_matrix()
        self.heatingData.append(np.sum(force_term_full))
        self.params["current_time"] += dt
        return self

    def evolve(self, n_steps: int, dt: float):
        for _ in tqdm.tqdm(range(n_steps), desc="TIME STEPS"):
            self.evolve_in_unit_timestep(dt)
        self.heatingData = np.cumsum(self.heatingData)
        return self

    def draw_heatmap(self):
        return self.result_matrix

def parameteresFunc(v1, v2, v3, v4, t1, t2, t3, t4, t5, t6):
    model_parameters = {
        "rooms": {
            "A1": {
                "row_min": 0, "row_max": 60, "col_min": 0, "col_max": 50,
                "init_func": lambda x: t1 + np.random.random(x.shape),
                "des": 297
            },
            "A2": {
                "row_min": 0, "row_max": 60, "col_min": 50, "col_max": 62,
                "init_func": lambda x: t2 + np.random.random(x.shape),
                "des": 297
            },
            "A3": {
                "row_min": 0, "row_max": 45, "col_min": 62, "col_max": 100,
                "init_func": lambda x: t3 + np.random.random(x.shape),
                "des": 288
            },
            "A4": {
                "row_min": 60, "row_max": 100, "col_min": 0, "col_max": 57,
                "init_func": lambda x: t4 + np.random.random(x.shape),
                "des": 297
            },
            "A5": {
                "row_min": 60, "row_max": 100, "col_min": 57, "col_max": 100,
                "init_func": lambda x: t5 + np.random.random(x.shape),
                "des": 297
            },
            "A6": {
                "row_min": 45, "row_max": 60, "col_min": 62, "col_max": 100,
                "init_func": lambda x: t6 + np.random.random(x.shape),
                "des": 297
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
                "row_min": 2, "row_max": 3, "col_min": 54, "col_max": 58, "maskValue": 1
            },
            "L2": {
                "row_min": 32, "row_max": 47, "col_min": 2, "col_max": 3, "maskValue": 2
            },
            "L3": {
                "row_min": 97, "row_max": 98, "col_min": 22, "col_max": 28, "maskValue": 3
            },
            "L4": {
                "row_min": 97, "row_max": 98, "col_min": 62, "col_max": 68, "maskValue": 4
            }
        },
        "domain": {
            "grid": np.meshgrid(np.linspace(-1, 1, 101), np.linspace(-1, 1, 101))[0], "dx": 1
        },
        "force_term": lambda x, t, mask: np.where(
            mask == 1, (np.sin(24 * t / 3600) ** 2 + v1) / 10, np.where(
                mask == 2, (np.sin(24 * t / 3600) ** 2 + v2) / 10, np.where(
                    mask == 3, (np.sin(24 * t / 3600) ** 2 + v3) / 10, np.where(
                        mask == 4, (np.sin(24 * t / 3600) ** 2 + v4) / 10, 0)
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
        "diffusion_coeff": 0.1,
        "current_time": 0.0
    }
    return model_parameters

def drawUsage(m1, m2, m3, m4, v1):
    model1 = HeatingModel(m1)
    model2 = HeatingModel(m2)
    model3 = HeatingModel(m3)
    model4 = HeatingModel(m4)
    model1.evolve(10000, 0.1)
    model2.evolve(10000, 0.1)
    model3.evolve(10000, 0.1)
    model4.evolve(10000, 0.1)
    plt.plot(model1.heatingData, "r", label=f'Masks ={2, 2, 2, 2}')
    plt.plot(model2.heatingData, "b", label=f'Masks ={0, 5, 5, 5}')
    plt.plot(model3.heatingData, "g", label=f'Masks ={1, 4, 2, 1}')
    plt.plot(model4.heatingData, "y", label=f'Masks ={3, 4, 3, 3}')
    plt.legend(loc="upper left")
    plt.title(f'Łączne zużycie energii, K_0 = {v1}')

def drawHeatingModel(m1, time):
    model = HeatingModel(m1)
    fig, axes = plt.subplots(2, 1, figsize=(6, 6))
    im1 = axes[0].imshow(model.result_matrix, cmap=plt.get_cmap("coolwarm"))
    model.evolve(time, 0.1)
    im2 = axes[1].imshow(model.result_matrix, cmap=plt.get_cmap("coolwarm"))
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    fig.colorbar(im2, cax=cbar_ax).set_label("Temperature[K]")


def drawHeatingModel2(m1):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    model1 = HeatingModel(m1)
    model2 = HeatingModel(m1)
    model3 = HeatingModel(m1)
    model4 = HeatingModel(m1)
    model1.evolve(10, 0.1)
    model2.evolve(100, 0.1)
    model3.evolve(1000, 0.1)
    model4.evolve(10000, 0.1)
    im1=axes[0, 0].imshow(model1.result_matrix, cmap=plt.get_cmap("coolwarm"))
    im2=axes[0, 1].imshow(model2.result_matrix, cmap=plt.get_cmap("coolwarm"))
    im3=axes[1, 0].imshow(model3.result_matrix, cmap=plt.get_cmap("coolwarm"))
    im4=axes[1, 1].imshow(model4.result_matrix, cmap=plt.get_cmap("coolwarm"))
    axes[0, 0].set_title(f'Time = 10')
    axes[0, 1].set_title(f'Time = 100')
    axes[1, 0].set_title(f'Time = 1000')
    axes[1, 1].set_title(f'Time = 10000')
    #im1 = axes[0].imshow(model.result_matrix, cmap=plt.get_cmap("coolwarm"))
    #im2 = axes[1].imshow(model.result_matrix, cmap=plt.get_cmap("coolwarm"))
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    fig.colorbar(im4, cax=cbar_ax).set_label("Temperature[K]")
