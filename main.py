import numpy as np
import matplotlib.pyplot as plt
class apartment:
    def __init__(self, params: dict):
        self.params = params
        self.partial_matrix = {}
        self.result_matrix = np.zeros((100,100))
        self.mask_matrix = np.zeros((100,100))
        self.build_partial_matrix()
        self.build_result_matrix()
        self.draw_heatmap()

    def build_partial_matrix(self):
        for room in self.params["rooms"].keys():
            coordinates = self.params["rooms"][room]
            self.partial_matrix[room] = np.zeros((coordinates["row_max"]-coordinates["row_min"], coordinates["col_max"]-coordinates["col_min"]))


if __name__ == "__main__":
    model_parameters = {
        "rooms": {
            "A1": {
                "row_min": 0, "row_max": 60, "col_min": 0, "col_max": 50
            },
            "A2": {
                "row_min": 0, "row_max": 60, "col_min": 50, "col_max": 62
            },
            "A3": {
                "row_min": 0, "row_max": 50, "col_min": 62, "col_max": 100
            },
            "A4": {
                "row_min": 60, "row_max": 100, "col_min": 0, "col_max": 57
            },
            "A5": {
                "row_min": 60, "row_max": 100, "col_min": 57, "col_max": 100
            },
            "A6": {
                "row_min": 50, "row_max": 60, "col_min": 62, "col_max": 100
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
                "row_min": 50, "row_max": 52, "col_min": 62, "col_max": 98
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
        "force_term": lambda x, t, mask: np.where(
            mask == 1, (np.sin(24 * t / 3600) ** 2 + 2) / 10, np.where(
                mask == 2, (np.sin(24 * t / 3600) ** 2 + 1) / 10, np.where(
                    mask == 3, (np.sin(24 * t / 3600) ** 2 + 1) / 10, np.where(
                        mask == 4, (np.sin(24 * t / 3600) ** 2 + 1) / 10, 0
                    )
                )
            )
        ),
        "maska": {
            "A1": 1,
            "A2": 1,
            "A3": 1,
            "A4": 1,
            "A5": 1,
            "A6": 0
        },
        "window_temp": lambda t: 280 - 10 * np.sin(24 * t / 3600),
        "diffusion": 0.1,
        "current_time": 0.0
    }
