import STcpClient
import numpy as np
import math


# define a infinite max value and a infinite min value
max_val = float('inf')
min_val = float('-inf')
dir_dict = {(-1,-1): 1, (0,-1): 2, (1,-1): 3, (-1,0): 4, (1,0): 6, (-1,1): 7, (0,1): 8, (1,1): 9}
boundary = 12
Debug = False

def output_map(mapStat, sheepStat):
    for i in range(12):
        for j in range(12):
            print(mapStat[i][j], end=' ')
        print()
    print("--------------------------")
    for i in range(12):
        for j in range(12):
            print(sheepStat[i][j], end=' ')
        print()
    print()




def get_legal_action(mapStat, sheepStat, playerID):
    actions = []
    for i in range(12):
        for j in range(12):
            if mapStat[i][j] == playerID and sheepStat[i][j] > 1:
                # 找可以移動的方向
                for dir in dir_dict.keys():
                    # 移動方向超出邊界或是該位置已經有人/障礙物
                    if i + dir[0] < 0 or i + dir[0] > 11 or j + dir[1] < 0 or j + dir[1] > 11 or mapStat[i+dir[0]][j+dir[1]] != 0:
                        continue
                    
                    actions.append([(i,j), sheepStat[i][j] // 2, dir_dict[dir]])
    return actions

def get_nxt_state(mapStat, sheepStat, action):
    x, y = action[0]
    split = action[1]
    dir = list()
    # find the direction with dir_dict
    for dir_k in dir_dict.keys():
        if dir_dict[dir_k] == action[2]:
            dir = dir_k
            break
        
    nxt_map = np.array(mapStat)
    nxt_sheep = np.array(sheepStat)
    tmp_x = x
    tmp_y = y
    while tmp_x + dir[0] >= 0 and tmp_x + dir[0] <= 11 and tmp_y + dir[1] >= 0 and tmp_y + dir[1] <= 11 and mapStat[tmp_x + dir[0]][tmp_y + dir[1]] == 0:
        tmp_x += dir[0]
        tmp_y += dir[1]
    nxt_map[tmp_x][tmp_y] = nxt_map[x][y]
    nxt_sheep[tmp_x][tmp_y] = split
    nxt_sheep[x][y] -= split
    return nxt_map, nxt_sheep


def is_win(mapStat, sheepStat, playerID):
    evaluate_dict = evaluate(playerID, mapStat, sheepStat, True)
    for key in evaluate_dict:
        if key != playerID and -1 * evaluate_dict[key] > evaluate_dict[playerID]:
            return False
    return True


        

def minmax(mapStat, sheepStat, iter_num, playerID, alpha, beta, index):
    def max_part( mapStat, sheepStat, iter_num, playerID, alpha, beta, index):
        val = float('-inf')
        best_action = None
        actions = get_legal_action(mapStat, sheepStat, index)
        # print(f'possible actions: {actions}')
        for action in actions:
            successor_map, successor_sheep = get_nxt_state(mapStat, sheepStat, action)
            successor_val, _ = minmax(successor_map, successor_sheep, iter_num - 1, playerID, alpha, beta, index + 1)
            if successor_val >= val:
                val = successor_val
                best_action = action
            if val > beta: break
            if val > alpha: alpha = val
        return val, best_action
                   
    def min_part( mapStat, sheepStat, iter_num, playerID, alpha, beta, index):
        val = float('inf')
        best_action = None
        actions = get_legal_action(mapStat, sheepStat, index)
        # print(f'possible actions: {actions}')
        for action in actions:
            successor_map, successor_sheep = get_nxt_state(mapStat, sheepStat, action)
            successor_val, _ = minmax(successor_map, successor_sheep, iter_num - 1, playerID, alpha, beta, index + 1)
            if successor_val <= val:
                val = successor_val
                best_action = action
            if val < alpha: break
            if val < beta: beta = val
        return val, best_action
               
    
    if index == 5: 
        index = 1  
        
    if Debug: print(f'index: {index} iter_num: {iter_num}, alpha: {alpha}, beta: {beta}')
    # input("Press Enter to continue...")
    if iter_num == 0 :
        score = evaluate(index, mapStat, sheepStat, is_oppoent= index != playerID)
        # print(f'score: {score}')
        return score, None
    if index == playerID:
        # print("max")
        return max_part( mapStat, sheepStat, iter_num, playerID, alpha, beta, index)
    else:
        # print("min")
        return min_part( mapStat, sheepStat, iter_num, playerID, alpha, beta, index)   
    
        
def evaluate(id, mapStat, sheepStat, is_oppoent = False, is_end = False):
    # id: 以哪個玩家的角度來評估
    # mapStat: 棋盤狀態
    # sheepStat: 羊群狀態
    # is_oppoent: 是否為對手, 若為對手, 則回傳負的評估值
    # is_end: 是否為結束狀態, 若為結束狀態, 則回傳結束狀態的評估值
    tmp = dict()
    scores = 0
    base_ratio = 1.25
    possible_step_ratio = 2
    for i in range(12):
        for j in range(12):
            if mapStat[i][j] != id:
                continue
            if sheepStat[i][j] not in tmp:
                tmp[sheepStat[i][j]] = 1
            else:
                tmp[sheepStat[i][j]] += 1
            if sheepStat[i][j] > 1:
                scores += evaulate_with_possible_move_nums(mapStat, i, j)  * sheepStat[i][j]         
            if (not is_end) and sheepStat[i][j] > 4:
                scores += (evaluate_with_future_possibility(mapStat, i, j, sheepStat[i][j]) * possible_step_ratio)
                pass
               
                
    for key in tmp:
        scores += tmp[key] ** base_ratio
    return scores if not is_oppoent else -scores


'''
    選擇起始位置
    選擇範圍僅限場地邊緣(至少一個方向為牆)
    
    return: init_pos
    init_pos=[x,y],代表起始位置
    
'''

def evaulate_with_possible_move_nums(mapStat, x, y):
    score = 0
    for dir in dir_dict.keys():
        # 超出邊界或是該位置已經有人/障礙物
        if x + dir[0] < 0 or x + dir[0] > 11 or y + dir[1] < 0 or y + dir[1] > 11 or mapStat[x + dir[0]][y + dir[1]] != 0:
            continue
        score += 1
    if score == 0:
        return -5 # 該位置無法移動  
    return score

def evaluate_with_future_possibility(mapStat, x, y, sheep_num = 8):
    if sheep_num <= 1: # 羊群數量不足，不考慮
        return 5
    score = 0
    tmp_map = np.array(mapStat)
    tmp_map[x][y] = 5 # 代表玩家
    for dir in dir_dict.keys():
        # 超出邊界或是該位置已經有人/障礙物
        if x + dir[0] < 0 or x + dir[0] > 11 or y + dir[1] < 0 or y + dir[1] > 11 or mapStat[x + dir[0]][y + dir[1]] != 0:
            continue
        tmp_x = x
        tmp_y = y
        # 移動到該位置
        while tmp_x + dir[0] >= 0 and tmp_x + dir[0] <= 11 and tmp_y + dir[1] >= 0 and tmp_y + dir[1] <= 11 and mapStat[tmp_x + dir[0]][tmp_y + dir[1]] == 0:
            tmp_x += dir[0]
            tmp_y += dir[1]
        score += evaluate_with_future_possibility(tmp_map, tmp_x, tmp_y, sheep_num // 2)
    # 如果該位置無法移動，則該位置的分數為-5 * 羊群數量
    if score == 0:
        return -5 * sheep_num
    return score       
        
    
    

def InitPos(mapStat):
    score = 0
    for i in range(boundary):
        for j in range(boundary):
            if i > 0 and i < boundary-1 and j > 0 and j < boundary-1:
                if mapStat[i - 1][j] != -1 and mapStat[i + 1][j] != -1 and mapStat[i][j - 1] != -1 and mapStat[i][j + 1] != -1:
                    continue
                else: # 可以選擇的位置
                    tmp = evaluate_with_future_possibility(mapStat, i, j, 8)
                    if tmp >= score:
                        score = tmp
                        x = i
                        y = j
    return [x, y]
    


'''
    產出指令
    
    input: 
    playerID: 你在此局遊戲中的角色(1~4)
    mapStat : 棋盤狀態(list of list), 為 12*12矩陣, 
              0=可移動區域, -1=障礙, 1~4為玩家1~4佔領區域
    sheepStat : 羊群分布狀態, 範圍在0~16, 為 12*12矩陣

    return Step
    Step : 3 elements, [(x,y), m, dir]
            x, y 表示要進行動作的座標 
            m = 要切割成第二群的羊群數量
            dir = 移動方向(1~9),對應方向如下圖所示
            1 2 3
            4 X 6
            7 8 9
'''
def GetStep(playerID, mapStat, sheepStat):
    # turn the number in the mapStat and sheepStat to interger instead of float
    mapStat = np.array(mapStat).astype(int)
    sheepStat = np.array(sheepStat).astype(int)
    # output_map(mapStat, sheepStat)
    _, action = minmax(mapStat, sheepStat, 4, playerID, min_val, max_val, playerID)
    if action is None:
        # assert error
        print("Error: action is None")
    else:
        print(f'action: {action}')
    
    return action


# player initial
(id_package, playerID, mapStat) = STcpClient.GetMap()
init_pos = InitPos(mapStat)
STcpClient.SendInitPos(id_package, init_pos)

# start game
while (True):
    (end_program, id_package, mapStat, sheepStat) = STcpClient.GetBoard()
    if end_program:
        STcpClient._StopConnect()
        break
    Step = GetStep(playerID, mapStat, sheepStat)

    STcpClient.SendStep(id_package, Step)
