PRICE = (1, 5)
QUANTITY = (1, 20)
CARBON = ["high", "med", "low"]

N_LEVELS = (PRICE[-1], QUANTITY[-1], len(CARBON))
N_OUTCOMES = PRICE[-1] * QUANTITY[-1] * len(CARBON)
