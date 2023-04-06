from torch.autograd import backward
import torch
import numpy as np
import math			
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
################# CLASS OF GAME ##################
class Game:
    # initial the board; 0-->empty
    def __init__(self,board,player):
        self.board = board
        self.player = player
    def __eq__(self,other):
        if self.board == other.board and self.player == other.player:
            return True
        else:
            return False
    # return all legal moves (current empty positions in sigle number version)
    def legal_move(self):
        board = self.board
        position_list = [idx for idx, val in enumerate(board) if val == 0]
        return position_list
    
    # put stone for current player
    def put_stone(self,NextMove): # valid NextMove
        CurrentPlayer = self.player
        CurrentBoard = list(self.board)
        if (CurrentPlayer%2) == 1: # odd player -- black
            ## TBD format of NextMove
            CurrentBoard[NextMove] = 1
        else:
            CurrentBoard[NextMove] = -1
        CurrentPlayer += 1
        return Game(CurrentBoard,CurrentPlayer)

    # draw
    def is_the_board_full(self):
        if len(Game.legal_move(self)) == 0:
            return True
        else: 
            return False
    
    # display the board
    def show_the_board(self):
        # --------- check previous structure -----------#
        board = self.board
        black,white,empty,guard = "  x","  o","  .","|"
        print("                                1  1  1  1  1  1  1  1  1  1")
        print("     1  2  3  4  5  6  7  8  9  0  1  2  3  4  5  6  7  8  9")
        print("   __________________________________________________________")
        for i in range(19): # vertical
            if i < 9:
                line = " " + str(i+1) + guard
            else:
                line = str(i+1) + guard
            for j in range(19): # horizontal
                if board[i*19+j] == 1:
                    line = line + black
                elif board[i*19+j] == -1:
                    line = line + white
                else:
                    line = line + empty
            print(line)
    
    # check if black wins
    def is_black_win(self):
        board = self.board
        blakes = [idx for idx, val in enumerate(board) if val == 1]
        for i in blakes:
            # step1: check horizontally
            count = 0
            connection = 1
            x = i
            while x < 360 and count < 4:
                count = count + 1
                if x+1 in blakes:
                    connection = connection + 1
                x = x + 1
            if connection == 5:
                return True
            
            # step2: check horizontally
            count = 0
            connection = 1
            y = i
            while y < 360 and count < 4:
                count = count + 1
                if y+19 in blakes:
                    connection = connection + 1
                y = y + 19
            if connection == 5:
                return True
            
            # step3: check diagnally
            count = 0
            connection = 1
            d1 = i
            while d1 < 360 and count < 4:
                count = count + 1
                if d1+20 in blakes:
                    connection = connection + 1
                d1 = d1+20
            if connection == 5:
                return True
            
            # step3: check diagnally
            count = 0
            connection = 1
            d2 = i
            while d2%19 != 0 and d2 < 360 and count < 4:
                count = count + 1
                if d2+18 in blakes:
                    connection = connection + 1
                d2 = d2 + 18
            if connection == 5:
                return True
        return False
    
    # check if black wins
    def is_white_win(self):
        board = self.board
        white = [idx for idx, val in enumerate(board) if val == -1]
        for i in white:
            # step1: check horizontally
            count = 0
            connection = 1
            x = i
            while x < 360 and count < 4:
                count = count + 1
                if x+1 in white:
                    connection = connection + 1
                x = x + 1
            if connection == 5:
                return True
            
            # step2: check horizontally
            count = 0
            connection = 1
            y = i
            while y < 360 and count < 4:
                count = count + 1
                if y+19 in white:
                    connection = connection + 1
                y = y + 19
            if connection == 5:
                return True
            
            # step3: check diagnally
            count = 0
            connection = 1
            d1 = i
            while d1 < 360 and count < 4:
                count = count + 1
                if d1+20 in white:
                    connection = connection + 1
                d1 = d1+20
            if connection == 5:
                return True
            
            # step3: check diagnally
            count = 0
            connection = 1
            d2 = i
            while d2%19 != 0 and d2 < 360 and count < 4:
                count = count + 1
                if d2+18 in white:
                    connection = connection + 1
                d2 = d2 + 18
            if connection == 5:
                return True
        return False
    
    def result(self):
        if Game.is_the_board_full(self):
            return 0
        elif Game.is_black_win(self):
            return 1
        elif Game.is_white_win(self):
            return -1
        else:
            return 2 # not end yet

#################### NETWORK ##########################
class Net(nn.Module):
    def __init__(self):
        # get init from parent class of Net
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(19, 100, 10) # 3: 3 channels, 6: num of filters, 5: dem of filters
        self.banorm1 = nn.BatchNorm1d(10)
        self.pool = nn.MaxPool1d(2,stride=2)
        self.conv2 = nn.Conv1d(100, 265, 3)
        self.banorm2 = nn.BatchNorm1d(3)
        self.convP = nn.Conv1d(265,1,1)
        self.convV = nn.Conv1d(265,1,1)
        self.banorm = nn.BatchNorm1d(1)
        self.fcp = nn.Linear(1, 19*19)
        self.fcv0 = nn.Linear(1, 500)
        self.fcv1 = nn.Linear(500, 300)
        self.fcv2 = nn.Linear(300, 150)
        self.fcv3 = nn.Linear(300, 100)
        self.fcv4 = nn.Linear(100, 50)
        self.fcv5 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.tensor(x)
        x = x.reshape(1,19*19)
        x = x.reshape(19,19)
        x = self.conv1(x)
        x = self.pool(F.relu(self.banorm1(x)))
        x = self.conv2(x)
        x = self.pool(F.relu(self.banorm2(x)))
        p = self.convP(x)
        p = F.softmax(self.fcp(p))
        p = p.reshape(19*19)
        v = self.convV(x)
        v = F.relu(self.fcv0(v))
        v = F.relu(self.fcv1(v))
        v = F.relu(self.fcv3(v))
        v = F.relu(self.fcv4(v))
        v = F.sigmoid(self.fcv5(v))
        v = v.reshape(1)
        return [p,v]

PATH = './T100.pth'
net = Net()
net.load_state_dict(torch.load(PATH))

####################### MCTS #######################
class Node(): # state (Game class) and edge information
    def __init__(self,state,move,parent):
        self.children = [] # list of tree
        self.state = state
        self.N = 0
        self.W = 0
        self.Q = 0
        self.move = move
        self.parent = parent # tree
    
    def __eq__(self,other):
        if self.state == other.state and self.move == other.move and self.parent == other.parent:
            return True
        else:
            return False
    
    def end(self):
        State = self.state
        condition = Game.result(State)
        if condition != 2:
            return True
        else:
            return False

    def add_children(self,subtree):
        self.children.append(subtree)

class Tree():
    def __init__(self,root,f,parent): # f: CNN
      self.root = root
      self.policy = f
      # self.visited = [root] # list of nodes
      self.children = []
      self.parent = parent
  
    def _eq_(self,other):
      if self.root == other.root:
        return True
      else:
        return False

    def add_child(self,subtree):
      self.children.append(subtree)
  
def total_visited(tree):
    sum = 0
    for item in tree.children:
        node = item.root
        sum = sum + node.N
    return sum

def selection(tree,policy,c): # c determing the level of exploration
  if Node.end(tree.root):
    return tree
  children = tree.children
  if children == []:
      return tree
  else:
      maxQU = -float("inf")
      TotalVisited = tree.root.N
      for item in children: # trees
          node = item.root
          CurrentN = node.N
          CurrentQ = node.Q
          CurrentMove = node.move
          prob = float(policy(node.state.board)[0][CurrentMove])
          ActionValue = CurrentQ + c * prob * ((math.sqrt(TotalVisited))/(1+CurrentN))
          if ActionValue > maxQU:
              maxQU = ActionValue
              NextTree = item
      return selection(NextTree,policy,c)
    
def pi_value(tree,t):
  SearchProb = [0]*(19*19)
  children = tree.children
  a = tree.root.N
  for item in children:
    node = item.root
    SearchProb[node.move] = pow(node.N,1/t) / pow(a,1/t)
  return SearchProb

def expansion(tree,net,RT):
  node = tree.root
  State = node.state
  if Node.end(node):
    backup(tree,float(tree.policy(State.board)[1]),RT)
  else:
    LegalMoves = Game.legal_move(State)
    for position in LegalMoves:
        NewState = Game.put_stone(State,position)
        NewNode = Node(NewState,position,node)
        NewTree = Tree(NewNode,net,tree)
        Tree.add_child(tree,NewTree)
        backup(NewTree,float(tree.policy(State.board)[1]),RT)
    
def backup(tree,num,RT):
  node = tree.root
  node.N += 1
  node.W = node.W + num
  node.Q = node.W / node.N
  if not(tree == RT):
    parent = tree.parent
    backup(parent,1-num,RT)

def next_move(tree,t):
  pi = pi_value(tree,t)
  pos = pi.index(max(pi))
  CurrentState = Game.put_stone((tree.root).state,pos)
  for item in tree.children:
    State = item.root.state
    if CurrentState == State:
      return [pi,item] 
      
def next(tree,net,c,t):
  iteration = 0
  while iteration < 1000:
    if tree.children == []:
      expansion(tree,net,tree) # initialization and backup
    else:
      leaftree = selection(tree,net,c)
      expansion(leaftree,net,tree)
    iteration += 1
  return next_move(tree,t) # [pi(list),subtree]

######################## SELFPLAY ########################
def self_play(net,c,t): # c = level of exploration; tree with fixed root
    CurrentState = Game([0.]*(19*19),1)
    CurrentNode = Node(CurrentState,[],[])
    tree = Tree(CurrentNode,net,[])
    CurrentPolicy = net
    pi = []
    p = []
    v = []
    steps = 0
    while abs(Game.result(CurrentState)) == 2:
        Next = next(tree,net,c,t)
        CurrentPi = Next[0] # list
        tree = Next[1] # tree
        CurrentPi = torch.from_numpy(np.array(CurrentPi))
        pi.append(CurrentPi)
        PolicyReturn = CurrentPolicy(CurrentState.board)
        p.append(PolicyReturn[0])
        v.append(PolicyReturn[1])
        CurrentState = (tree.root).state
        steps = steps + 1
        # Game.show_the_board(CurrentState)
    z = [0]*steps
    for i in range(-1,-steps-1,-1):
      if abs(i)%2 == 1:
        z[i] = torch.tensor([1])
      else:
        z[i] = torch.tensor([0])
    p = torch.stack(p)
    v = torch.stack(v)
    pi = torch.stack(pi)
    z = torch.stack(z)
    return [p,v],[pi,z]

####################### TRAINING ###################
class Loss(nn.Module):
  def __init__(self):
      super(Loss,self).__init__()
  def forward(self,output,target):
    pi = target[0]
    z = target[1]
    p = output[0]
    v = output[1]
    steps = pi.shape[0]
    diff = z - v
    firstTerm = torch.mul(diff,diff)
    secondTerms = []
    for i in range (steps):
      second = 0
      current_p = p[i]
      current_pi = pi[i]
      current_p = torch.log(current_p)
      for a in range (current_p.shape[0]):
        second = second + current_p[a]*current_pi[a]
      secondTerms.append(torch.tensor([second]))
    secondTerm = torch.stack(secondTerms)
    l2_norm = sum(p.pow(2.0).sum() for p in net.parameters())
    loss = torch.sum((firstTerm - secondTerm))/steps + 1e-4*l2_norm
    return loss
  

def main():
# defined loss function and optimizer
    criterion = Loss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    optimizer.zero_grad()
    count = 0
    while count < 100000:
        if count <= 30:
            c = 1
            t = 1
            output,target = self_play(net,c,t)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            PATH = './T100.pth'
            torch.save(net.state_dict(), PATH)
            count += 1
        else:
            c = 0.5
            t = 0.1
            output,target = self_play(net,c,t)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            PATH = './T100.pth'
            torch.save(net.state_dict(), PATH)
            count += 1
if __name__ == "__main__":
    main()