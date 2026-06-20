# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import itertools
import os
from typing import List

from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

def swarmColonyOptimization(trainX, trainY, baseSystem, tryToFindBetterSolution=False):
    from SwarmAlgorithm import swarm_algorithm
    optimizedSystem = Fuzzy.ControlSystem(baseSystem.rules) 

    if not tryToFindBetterSolution and os.path.exists("best_solution.txt"):
        with open("best_solution.txt", "r") as f:
            print("LOADING BEST SOLUTION FROM FILE")
            best_params = list(map(float, f.read().strip().split(",")))
            applyParametersToSystem(optimizedSystem, best_params)
            return optimizedSystem
            
    params_from_cached_filpe : List[float] | None = None
    if tryToFindBetterSolution and os.path.exists("best_solution.txt"):
        with open("best_solution.txt", "r") as f:
            print("LOADING BEST SOLUTION FROM FILE TO COMPARE WITH ACO")
            params_from_cached_filpe = list(map(float, f.read().strip().split(",")))
            
    params_calculated : List[float] = swarm_algorithm(baseSystem, trainX, trainY)
    
    if params_from_cached_filpe is not None:
        applyParametersToSystem(optimizedSystem, params_calculated)
        results_for_calculated = np.sum((trainY - evaluateCustomSystem(Fuzzy.ControlSystemSimulation(optimizedSystem), trainX))**2)
        
        applyParametersToSystem(optimizedSystem, params_from_cached_filpe)
        results_for_cached = np.sum((trainY - evaluateCustomSystem(Fuzzy.ControlSystemSimulation(optimizedSystem), trainX))**2)
        
        better_params : List[float] = params_calculated if results_for_calculated < results_for_cached else params_from_cached_filpe
    else:
        better_params = params_calculated
        
    optimizedSystem = applyParametersToSystem(baseSystem, better_params)
    if os.path.exists("best_solution.txt"):
        os.remove("best_solution.txt")
    with open("best_solution.txt", "w") as f:
        f.write(",".join(map(str, better_params)))
        
    return optimizedSystem

class Fuzzy:
    class Trimf:
        def __init__(self, universe, abc):
            self.universe = universe
            self.abc = abc
            self.values = Fuzzy.Trimf.evaluateTrimfArray(universe, abc)

        @staticmethod
        def trimf(universe, abc):
            return Fuzzy.Trimf(universe, abc)

        @staticmethod
        def evaluateTrimfArray(universe, abc):
            a, b, c = abc
            values = np.zeros_like(universe, dtype=float)
            for i, x in enumerate(universe):
                if a < x < b:
                    values[i] = (x - a) / (b - a) if b != a else 1.0
                elif b <= x < c:
                    values[i] = (c - x) / (c - b) if c != b else 1.0
                elif x == b:
                    values[i] = 1.0
            return values

        @staticmethod
        def evaluateMfScalar(mf, x):
            # OMIJAMY CYRKULARNY IMPORT: Zamiast isinstance(mf, Fuzzy.Trimf)
            if hasattr(mf, 'abc'):
                a, b, c = mf.abc
                if x < a or x > c:
                    return 0.0
                if a <= x <= b:
                    if b == a:
                        return 1.0 if x == a else 0.0
                    return (x - a) / (b - a) if b != a else 0.0
                if b < x <= c:
                    if b == c:
                        return 1.0 if x == c else 0.0
                    return (c - x) / (c - b) if c != b else 0.0
                if x == b:
                    return 1.0
            return 0.0

    class FuzzyTerm:
        def __init__(self, variable, label):
            self.variable = variable
            self.label = label

        def __and__(self, other):
            return Fuzzy.RuleCondition(self, other, "AND")

        def __or__(self, other):
            return Fuzzy.RuleCondition(self, other, "OR")

    class RuleCondition:
        def __init__(self, left, right, operator):
            self.left = left
            self.right = right
            self.operator = operator

        def __and__(self, other):
            return Fuzzy.RuleCondition(self, other, "AND")

        def __or__(self, other):
            return Fuzzy.RuleCondition(self, other, "OR")

    @staticmethod
    def evaluateCondition(condition, inputs):
        # OMIJAMY CYRKULARNY IMPORT za pomocą duck-typing (hasattr)
        if hasattr(condition, 'operator'):
            leftVal = Fuzzy.evaluateCondition(condition.left, inputs)
            rightVal = Fuzzy.evaluateCondition(condition.right, inputs)
            if condition.operator == "AND":
                return min(leftVal, rightVal)
            elif condition.operator == "OR":
                return max(leftVal, rightVal)
        elif hasattr(condition, 'variable'):
            varName = condition.variable.name
            label = condition.label
            x = inputs.get(varName, 0.0)
            mf = condition.variable.terms[label]
            return Fuzzy.Trimf.evaluateMfScalar(mf, x)
        return 0.0

    class Antecedent:
        def __init__(self, universe, name):
            self.universe = universe
            self.name = name
            self.terms = {}

        def __setitem__(self, label, mf):
            self.terms[label] = mf

        def __getitem__(self, label):
            return Fuzzy.FuzzyTerm(self, label)

    class Consequent:
        def __init__(self, universe, name):
            self.universe = universe
            self.name = name
            self.terms = {}

        def __setitem__(self, label, mf):
            self.terms[label] = mf

        def __getitem__(self, label):
            return Fuzzy.FuzzyTerm(self, label)

    class Rule:
        def __init__(self, antecedentCondition, consequentTerm):
            self.antecedentCondition = antecedentCondition
            self.consequentTerm = consequentTerm

    class ControlSystem:
        def __init__(self, rules=None):
            self.rules = rules if rules is not None else []

    class ControlSystemSimulation:
        def __init__(self, controlSystem):
            self.controlSystem = controlSystem
            self.input = {}
            self.output = {}

        def compute(self):
            consequentOutputs = {}
            for rule in self.controlSystem.rules:
                firingStrength = Fuzzy.evaluateCondition(
                    rule.antecedentCondition, self.input
                )
                consequentVar = rule.consequentTerm.variable
                consequentLabel = rule.consequentTerm.label
                mf = consequentVar.terms[consequentLabel]

                clippedValues = np.minimum(firingStrength, mf.values)
                if consequentVar.name not in consequentOutputs:
                    consequentOutputs[consequentVar.name] = np.zeros_like(
                        consequentVar.universe, dtype=float
                    )
                consequentOutputs[consequentVar.name] = np.maximum(
                    consequentOutputs[consequentVar.name], clippedValues
                )

            for varName, aggregatedMf in consequentOutputs.items():
                consequentVar = None
                for rule in self.controlSystem.rules:
                    if rule.consequentTerm.variable.name == varName:
                        consequentVar = rule.consequentTerm.variable
                        break
                sumMf = np.sum(aggregatedMf)
                if sumMf == 0:
                    self.output[varName] = np.mean(consequentVar.universe)
                else:
                    self.output[varName] = (
                        np.sum(consequentVar.universe * aggregatedMf) / sumMf
                    )

def generateGridPartitioningRules(antecedents, consequent):
    rules = []
    termLists = [(ant, list(ant.terms.keys())) for ant in antecedents]
    combinations = list(itertools.product(*[t[1] for t in termLists]))
    consequentTerms = list(consequent.terms.keys())
    numCons = len(consequentTerms)

    for combo in combinations:
        cond = antecedents[0][combo[0]]
        score = termLists[0][1].index(combo[0])

        for j in range(1, len(antecedents)):
            cond = cond & antecedents[j][combo[j]]
            score += termLists[j][1].index(combo[j])

        maxScore = sum([len(t[1]) - 1 for t in termLists])
        normalizedScore = score / maxScore if maxScore > 0 else 0
        consIndex = int(round(normalizedScore * (numCons - 1)))

        consTerm = consequentTerms[consIndex]
        rules.append(Fuzzy.Rule(cond, consequent[consTerm]))

    return rules

def getUniqueVariables(system):
    antecedents = {}
    consequents = {}

    def collectVariables(condition):
        # OMIJAMY CYRKULARNY IMPORT
        if hasattr(condition, 'operator'):
            collectVariables(condition.left)
            collectVariables(condition.right)
        elif hasattr(condition, 'variable'):
            antecedents[condition.variable.name] = condition.variable

    for rule in system.rules:
        collectVariables(rule.antecedentCondition)
        consequents[rule.consequentTerm.variable.name] = rule.consequentTerm.variable

    return list(antecedents.values()), list(consequents.values())

def applyParametersToSystem(system, parameterVector):
    antecedents, consequents = getUniqueVariables(system)
    idx = 0
    for var in antecedents + consequents:
        for label in var.terms.keys():
            abc = sorted(parameterVector[idx : idx + 3])
            var.terms[label] = Fuzzy.Trimf.trimf(var.universe, abc)
            idx += 3
    return system

def pythonGenfis(trainX, trainY, numMfs=3):
    numFeatures = trainX.shape[1]
    antecedents = []

    for i in range(numFeatures):
        featMin = trainX[:, i].min()
        featMax = trainX[:, i].max()
        if featMin == featMax:
            featMax += 1.0
        universe = np.linspace(featMin, featMax, 100)
        ant = ctrl.Antecedent(universe, f"input{i}")

        centers = np.linspace(featMin, featMax, numMfs)
        for j in range(numMfs):
            if j == 0:
                abc = [featMin, featMin, centers[j + 1] if numMfs > 1 else featMax]
            elif j == numMfs - 1:
                abc = [centers[j - 1], featMax, featMax]
            else:
                abc = [centers[j - 1], centers[j], centers[j + 1]]
            ant[f"mf{j}"] = fuzz.trimf(ant.universe, abc)
        antecedents.append(ant)

    outMin = trainY.min()
    outMax = trainY.max()
    if outMin == outMax:
        outMax += 1.0
    outUniverse = np.linspace(outMin, outMax, 100)
    consequent = ctrl.Consequent(outUniverse, "output")
    outCenters = np.linspace(outMin, outMax, numMfs)

    for j in range(numMfs):
        if j == 0:
            abc = [outMin, outMin, outCenters[j + 1] if numMfs > 1 else outMax]
        elif j == numMfs - 1:
            abc = [outCenters[j - 1], outMax, outMax]
        else:
            abc = [outCenters[j - 1], outCenters[j], outCenters[j + 1]]
        consequent[f"mf{j}"] = fuzz.trimf(consequent.universe, abc)

    rules = []
    seenRules = set()

    for i in range(len(trainX)):
        bestAntLabels = []
        for j, ant in enumerate(antecedents):
            bestLabel = None
            maxMu = -1.0
            for label in ant.terms.keys():
                mu = fuzz.interp_membership(ant.universe, ant[label].mf, trainX[i, j])
                if mu > maxMu:
                    maxMu = mu
                    bestLabel = label
            bestAntLabels.append(bestLabel)

        bestOutLabel = None
        maxMuOut = -1.0
        for label in consequent.terms.keys():
            mu = fuzz.interp_membership(consequent.universe, consequent[label].mf, trainY[i])
            if mu > maxMuOut:
                maxMuOut = mu
                bestOutLabel = label

        ruleKey = tuple(bestAntLabels)
        if ruleKey not in seenRules and bestOutLabel is not None:
            seenRules.add(ruleKey)
            cond = antecedents[0][bestAntLabels[0]]
            for k in range(1, len(antecedents)):
                cond = cond & antecedents[k][bestAntLabels[k]]
            rules.append(ctrl.Rule(cond, consequent[bestOutLabel]))

    if not rules:
        rules.append(ctrl.Rule(antecedents[0]["mf0"], consequent["mf0"]))

    system = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(system)

def loadSolarFlareData():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/solar-flare/flare.data2"
    try:
        df = pd.read_csv(url, sep=r"\s+", skiprows=1, header=None)
        df = df.dropna()
        for col in [0, 1, 2]:
            df[col] = df[col].astype("category").cat.codes
        data = df.values.astype(float)
        X = data[:, :10]
        y = data[:, 10]
        return X, y
    except Exception as e:
        print(f"ERROR WITH LOADING DATA: {e}, USING RANDOM DATA INSTEAD")
        np.random.seed(42)
        X = np.random.randint(1, 4, size=(200, 10)).astype(float)
        y = np.random.randint(0, 5, size=(200,)).astype(float)
        return X, y

def evaluateCustomSystem(simulation, XData):
    predictions = []
    for i in range(len(XData)):
        simulation.input["evolution"] = XData[i, 4]
        simulation.input["prevActivity"] = XData[i, 5]
        simulation.compute()
        predictions.append(simulation.output.get("cClassFlares", 0.0))
    return np.array(predictions)

def evaluateSkfuzzySystem(simulation, XData):
    predictions = []
    for i in range(len(XData)):
        for j in range(XData.shape[1]):
            simulation.input[f"input{j}"] = XData[i, j]
        try:
            simulation.compute()
            predictions.append(simulation.output["output"])
        except Exception:
            predictions.append(0.0)
    return np.array(predictions)

def calculateMse(yTrue, yPred):
    return np.mean((yTrue - yPred) ** 2)

if __name__ == "__main__":
    X, y = loadSolarFlareData()

    halfIndex = len(X) // 2
    trainX, testX = X[:halfIndex], X[halfIndex:]
    trainY, testY = y[:halfIndex], y[halfIndex:]

    evolutionUniverse = np.linspace(1, 3, 100)
    prevActivityUniverse = np.linspace(1, 3, 100)
    flaresUniverse = np.linspace(0, 10, 100)

    evolution = Fuzzy.Antecedent(evolutionUniverse, "evolution")
    prevActivity = Fuzzy.Antecedent(prevActivityUniverse, "prevActivity")
    cClassFlares = Fuzzy.Consequent(flaresUniverse, "cClassFlares")

    evolution["decay"] = Fuzzy.Trimf.trimf(evolutionUniverse, [1, 1, 2])
    evolution["noGrowth"] = Fuzzy.Trimf.trimf(evolutionUniverse, [1, 2, 3])
    evolution["growth"] = Fuzzy.Trimf.trimf(evolutionUniverse, [2, 3, 3])

    prevActivity["low"] = Fuzzy.Trimf.trimf(prevActivityUniverse, [1, 1, 2])
    prevActivity["medium"] = Fuzzy.Trimf.trimf(prevActivityUniverse, [1, 2, 3])
    prevActivity["high"] = Fuzzy.Trimf.trimf(prevActivityUniverse, [2, 3, 3])

    cClassFlares["none"] = Fuzzy.Trimf.trimf(flaresUniverse, [0, 0, 2])
    cClassFlares["few"] = Fuzzy.Trimf.trimf(flaresUniverse, [1, 3, 5])
    cClassFlares["many"] = Fuzzy.Trimf.trimf(flaresUniverse, [4, 10, 10])

    gridRules = generateGridPartitioningRules([evolution, prevActivity], cClassFlares)
    customSystem = Fuzzy.ControlSystem(gridRules)

    tunedSystem = swarmColonyOptimization(trainX, trainY, customSystem, True)

    customSimulation = Fuzzy.ControlSystemSimulation(customSystem)
    customSimulationACO = Fuzzy.ControlSystemSimulation(tunedSystem)
    genfisSimulation = pythonGenfis(trainX, trainY, numMfs=3)

    customPreds = evaluateCustomSystem(customSimulation, testX)
    customPredsACO = evaluateCustomSystem(customSimulationACO, testX)
    genfisPreds = evaluateSkfuzzySystem(genfisSimulation, testX)

    customMse = calculateMse(testY, customPreds)
    customMseACO = calculateMse(testY, customPredsACO)
    genfisMse = calculateMse(testY, genfisPreds)

    # MSE
    print(f"MSE Custom System (Przed ACO): {customMse:.4f}")
    print(f"MSE Custom System (Po ACO):    {customMseACO:.4f}")
    print(f"MSE genfis System:             {genfisMse:.4f}")

    # R2
    customR2 = r2_score(testY, customPreds)
    customR2ACO = r2_score(testY, customPredsACO)
    genfisR2 = r2_score(testY, genfisPreds)

    print(f"\nR2 Custom System (Przed ACO):  {customR2:.4f}")
    print(f"R2 Custom System (Po ACO):     {customR2ACO:.4f}")
    print(f"R2 genfis System:              {genfisR2:.4f}")

    # MAE
    customMae = mean_absolute_error(testY, customPreds)
    customMaeACO = mean_absolute_error(testY, customPredsACO)
    genfisMae = mean_absolute_error(testY, genfisPreds)

    print(f"\nMAE Custom System (Przed ACO): {customMae:.4f}")
    print(f"MAE Custom System (Po ACO):    {customMaeACO:.4f}")
    print(f"MAE genfis System:             {genfisMae:.4f}")