def patch(code):
    old = 'temperatures = np.geomspace(start_temperature, T4_SA_SETTINGS[inst["scenario"]]["end_ratio"] * start_temperature, budget - 1)'
    assert code.count(old) == 1
    # reviewer mutation: annealing only gets 10 proposals -> results MUST change if they are recomputed
    return code.replace(old, old.replace("budget - 1)", "10)"))
