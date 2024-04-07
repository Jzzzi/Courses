# Variable naming Rules

Variable named "action" usually refers to a variable of string type indicating the direction of the move, like "West", "East", "North", "South".

---

The variable named "problem" in the function "depthFirstSearch" in search.py is in the type of class PositionSearchProblem in serchAgent.py, which has the following methods:

* getStartState(self)
* isGoalState(self, state)
* getSuccessors(self, state)
  returns the tuple in form of (nextState, action, cost),
* getCostOfActions(self, actions) 

in which the variable named "state" is a tuple of the position.

---

In cornersproblem, variable named "state" is a tuple in type of (position, cornersEaten). Also, in the problem of cornersproblem, the variable named cornersEaten is a list of bool variables that indicate whether the food at each corner has been eaten.

