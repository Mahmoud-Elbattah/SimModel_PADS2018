import pandas as pd
import urllib.request
import json

class SimClock:
    SimTime = 0
    def __init__(self, initialTime):
        self.SimTime = initialTime

    def Forward(self, printTime):
        self.SimTime= self.SimTime + 1
        if printTime:
            print("SimTime:", self.SimTime)

class Task:
    taskID = None
    arrivalTime = 0
    waitingTime = 0
    taskType = 0
    maxWaitingTime=0
    waiting = True
    feature1 = 0
    feature2 = 0
    delayPenalty = False

    def __init__(self,taskID ,arrivalTime,taskType,maxWaitingTime,feature1,feature2):  # Class constructor
        self.taskID = taskID
        self.arrivalTime = arrivalTime
        self.taskType = taskType
        self.maxWaitingTime = maxWaitingTime
        self.feature1 = feature1
        self.feature2 =feature2

    def AbondonQueue(self,simTime):
        if self.waiting and (simTime - self.arrivalTime)> self.maxWaitingTime:
            self.waitingTime = self.maxWaitingTime
            return True

class Agent:
    idle = True
    agentType = 0
    avgServiceTime = 0
    serviceStartTime = 0
    totalProcessedTasks = 0
    currentTask = Task(taskID=0, arrivalTime=0, taskType=0,maxWaitingTime=0,feature1=0,feature2=0)
    processingDelay = 0

    def printMsg(self):
        print("Agent of type:",self.agentType)

    def __init__(self, type, avgServiceTime):
        self.agentType = type
        self.avgServiceTime = avgServiceTime
        print("Type" + str(self.agentType), "Agent Created.")

    def HandleTask(self, simTime, currentTask):
        self.serviceStartTime = simTime
        self.currentTask = currentTask
        self.idle = False
        #print("#",self.currentTask.taskID)

    def UpdateStatus(self, simTime):
        if not self.idle:
            if simTime >= (self.serviceStartTime + self.avgServiceTime + self.processingDelay):
                self.idle = True #Bring agent available again
                self.serviceStartTime = 0 #Reset service start time
                self.totalProcessedTasks += 1
                #print("Task#", self.currentTask.taskID, " accomplished.")
                return True
            else:
                return False

def ReadDataset(filePath):
    df = pd.read_csv(filePath)
    return df
def PredictTaskType (feature1Val,feature2Val):
    data =  {
            "Inputs": {
                    "input1":
                    {
                        "ColumnNames": ["Feature1", "Feature2"],
                        "Values": [ [feature1Val, feature2Val], ]
                    },},
                "GlobalParameters": {
    }
        }
    body = str.encode(json.dumps(data))
    #Low accuracy API Key:
    #Poor classifier model deployed at Azure ML
    url = 'https://ussouthcentral.services.azureml.net/workspaces/ee457a75262244d4b299080febd209db/services/31a026709fdc47b988351d370f76d50c/execute?api-version=2.0&details=true'

    api_key = 'WsuKfPXPziXqHUrIEwCZ7vQs8sdMEvX0BtT5m3lzz0NCBZWTH+2Rs4+37759ccsmr6k/5eeoN4yThnQPmscJcA=='
    headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}
    req = urllib.request.Request(url, body, headers)

    try:
        response = urllib.request.urlopen(req)
        resultJSON = json.loads(response.read())
        return int(resultJSON["Results"]["output1"]["value"]["Values"][0][2]) + 1
        #print("Label:", resultJSON["Results"]["output1"]["value"]["Values"][0][2])
    except :
        print("The request failed!")

def SimModel(simulatedTime):
    # Model variables and parameters
    taskArrivalRate = 3 #A task arrives per 3 units of time
    taskQueue = []
    processedTasks = []
    failedTasks = []
    taskAutoID = 0
    totalTasks = 0
    agent1 = Agent(type=1, avgServiceTime=5)
    agent2 = Agent(type=2, avgServiceTime=8)
    simClock = SimClock(initialTime=0)
    mismatchCount=0
    correctCount=0
    misClassificationDelay = 4 #Extra time added as a penalty to task processing time in case of misclassifying its type

    taskData = pd.read_csv("Bad Classifier.csv")

    while simClock.SimTime in range(simulatedTime):#Simulation starts
        #First, check in-progress tasks (if any)
        if agent1.UpdateStatus(simClock.SimTime):#Returns True if agent finished processing and is idle now
            processedTasks.append(agent1.currentTask)
            agent1.currentTask = None
        if agent2.UpdateStatus(simClock.SimTime):
            processedTasks.append(agent2.currentTask)
            agent2.currentTask = None
       #Second, check for over-waiting tasks that should leave the queue (i.e. failed tasks)
        for i,task in enumerate(taskQueue):
            if task.AbondonQueue(simTime=simClock.SimTime):
                failedTasks.append(task) #Add task to the list of failed tasks
                taskQueue.pop(i)  #Removing task from the waiting queue

        # New task arrival
        # taskArrivalRate defined above
        if simClock.SimTime % taskArrivalRate == 0:
            #taskType = random.choice([1,2])
            randomTask = taskData.sample()# Randomly pick at ask from the generated data
            taskType = int(randomTask.iloc[0]["Label"]) + 1
            taskAutoID += 1
            feature1 = randomTask.iloc[0]["Feature1"]
            feature2 = randomTask.iloc[0]["Feature2"]
            #Creating a task object
            newTask = Task(taskID=taskAutoID, arrivalTime=simClock.SimTime, taskType=taskType, maxWaitingTime=6, feature1=feature1, feature2=feature2)
            if newTask.taskType != PredictTaskType(feature1Val=newTask.feature1, feature2Val=newTask.feature2):
                newTask.delayPenalty = True #This will incur an extra processing time to be added
                #print("Mismatch")
                mismatchCount += 1
            else:
                correctCount += 1
            taskQueue.append(newTask)
            totalTasks += 1

        #Getting current task in queue to process by agents
        if len(taskQueue) > 0:
            task = taskQueue[0]
            agent = None
            if task.taskType == 1 and agent1.idle:  #Task of Type1
                agent = agent1
            elif task.taskType == 2 and agent2.idle: #Task of Type2
                agent = agent2

            if agent is not None:#Assign agent to a task
                taskQueue.pop(0) #Remove task from waiting queue
                task.waiting = False
                if task.delayPenalty:
                    agent.processingDelay = misClassificationDelay
                task.waitingTime = simClock.SimTime - task.arrivalTime # Set total waiting time before assigning to an agent
                agent.HandleTask(simTime=simClock.SimTime, currentTask=task)
            #else:
            #    print("All agents are busy!")

        simClock.Forward(printTime=False)
    #End of simulation
    #Printing summary statistics about the simulation experiment
    #print("------------------------------End of Simulation------------------------------")
    #print("Count of processed tasks:", len(processedTasks))
    #print("Agent1 tasks:", agent1.totalProcessedTasks)
    #print("Agent2 tasks:", agent2.totalProcessedTasks)
    #print("Total Failed Tasks:", len(failedTasks))
    tasks = processedTasks + failedTasks
    avgWaitingTime = sum(task.waitingTime for task in tasks) / len(processedTasks)

    #print("Average Waiting Time=", round(avgWaitingTime,2))
    #print("Mismatch Count:",mismatchCount)
    #print("Correct Count:", correctCount)
    return (len(processedTasks), agent1.totalProcessedTasks, agent2.totalProcessedTasks, len(failedTasks), round(avgWaitingTime,2), mismatchCount, correctCount)
    #print (totalTasks)

def main():
    experimentCount = 50
    simulatedTime= 720 #8hrs
    simOutput = pd.DataFrame(columns=['ExperimentNo', 'ProcessedTasks', 'Agent1Tasks', 'Agent2Tasks', 'FailedTasks',
                                      'AvgWaitingTime', 'FalsePredictions', 'TruePredictions'])# Create empty dataframe with 3 co
    for experimentNo in range(0, experimentCount):
        print("Simulation Experiment#" + str(experimentNo+1), "started.")
        summary = SimModel(simulatedTime=simulatedTime)
        simOutput.loc[experimentNo,"ExperimentNo"] = experimentNo + 1
        simOutput.loc[experimentNo,"ProcessedTasks"] = summary[0]
        simOutput.loc[experimentNo, "Agent1Tasks"] = summary[1]
        simOutput.loc[experimentNo, "Agent2Tasks"] = summary[2]
        simOutput.loc[experimentNo, "FailedTasks"] = summary[3]
        simOutput.loc[experimentNo, "AvgWaitingTime"] = summary[4]
        simOutput.loc[experimentNo, "FalsePredictions"] = summary[5]
        simOutput.loc[experimentNo, "TruePredictions"] = summary[6]
        print("Simulation Experiment#" + str(experimentNo+1), "ended.")
        print("------------------------------------------------------")

    simOutput.to_csv('SimOutput_BadClassifier.csv')
    print("End of Experiments.")
    print("Simulation output saved to disk.")
    print(simOutput)
main()