from heatingProject import HeatingModel
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

model_parameters = {
    "rooms": {
        "A1": {
            "row_min": 0, "row_max": 60, "col_min": 0, "col_max": 50,
            "init_func": lambda x: 297 + np.random.random(x.shape),
            "des": 297
        },
        "A2": {
            "row_min": 0, "row_max": 60, "col_min": 50, "col_max": 62,
            "init_func": lambda x: 294 + np.random.random(x.shape),
            "des": 297
        },
        "A3": {
            "row_min": 0, "row_max": 45, "col_min": 62, "col_max": 100,
            "init_func": lambda x: 285 + np.random.random(x.shape),
            "des": 288
        },
        "A4": {
            "row_min": 60, "row_max": 100, "col_min": 0, "col_max": 57,
            "init_func": lambda x: 296 + np.random.random(x.shape),
            "des": 297
        },
        "A5": {
            "row_min": 60, "row_max": 100, "col_min": 57, "col_max": 100,
            "init_func": lambda x: 297 + np.random.random(x.shape),
            "des": 297
        },
        "A6": {
            "row_min": 45, "row_max": 60, "col_min": 62, "col_max": 100,
            "init_func": lambda x: 295 + np.random.random(x.shape),
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
            "row_min": 2, "row_max": 3, "col_min": 54, "col_max": 58
        },
        "L2": {
            "row_min": 32, "row_max": 47, "col_min": 2, "col_max": 3
        },
        "L3": {
            "row_min": 97, "row_max": 98, "col_min": 22, "col_max": 28
        },
        "L4": {
            "row_min": 97, "row_max": 98, "col_min": 62, "col_max": 68
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
        "A1": 1, "A2": 1, "A3": 0, "A4": 1, "A5": 1, "A6": 1
    },
    "window_temp": lambda t: 280 - 10 * np.sin(24 * t / 3600),
    "diffusion_coeff": 0.1,
    "current_time": 0.0
}

def draw_house(object_house):
    result_m = object_house
    for key, val in result_m.params['rooms'].items():
        result_m.partial_matrix[key] = np.zeros((val['row_max'] - val['row_min'], val['col_max'] - val['col_min']))
        mask = result_m.params["masks"][key]
        result_m.partial_matrix[key][mask == 1] = 4
        result_m.result_matrix[val['row_min']:val['row_max'], val['col_min']:val['col_max']] = result_m.partial_matrix[key]
    for v in result_m.indx:
        for key, val in result_m.params[v].items():
            result_m.result_matrix[val['row_min']:val['row_max'], val['col_min']:val['col_max']] = result_m.indx[v]
    return result_m

cmap = ListedColormap(['white', 'lightskyblue', 'black', 'orange', 'floralwhite', 'crimson'])
plan_draw = HeatingModel(model_parameters)
house_plot = draw_house(plan_draw)
plt.imshow(plan_draw.result_matrix, cmap=cmap)
plt.show()