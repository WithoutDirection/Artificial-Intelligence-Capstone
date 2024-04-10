
#include "STcpClient.h"
#include <bits/stdc++.h>

#define boundary 12
using namespace std;

typedef vector<int> Action;
typedef pair<vector<vector<int>>, vector<vector<int>>> State;
typedef pair<double, Action> Node;

// mutex mtx;


map<int, pair<int, int>> dir_map = {
	{1, make_pair(-1, -1)},
	{2, make_pair(0, -1)},
	{3, make_pair(1, -1)},
	{4, make_pair(-1, 0)},
	{6, make_pair(1, 0)},
	{7, make_pair(-1, 1)},
	{8, make_pair(0, 1)},
	{9, make_pair(1, 1)}
};

bool Debug = false;
bool show_scores = false;
bool show_step = false;
void print_state(State &s){
	vector<vector<int>> mapStat = s.first;
	vector<vector<int>> sheepStat = s.second;
	cout << "mapStat: " << endl;
	for(int i = 0; i < boundary; i++){
		for(int j = 0; j < boundary; j++){
			cout << mapStat[i][j] << " ";
		}
		cout << endl;
	}
	cout << "sheepStat: " << endl;
	for(int i = 0; i < boundary; i++){
		for(int j = 0; j < boundary; j++){
			cout << sheepStat[i][j] << " ";
		}
		cout << endl;
	}
}

vector<Action> get_action(vector<vector<int>> &mapStat, vector<vector<int>> &sheepStat, int playerID){
	vector<Action> actions;
	for(int i = 0; i < boundary; i++){
		for(int j = 0; j < boundary; j++){
			if (mapStat[i][j] == playerID && sheepStat[i][j] >1){
				for(auto &dir: dir_map){
					int dlt_x = dir.second.first;
					int dlt_y = dir.second.second;
					if(i + dlt_x < 0 || i + dlt_x > boundary - 1 || j + dlt_y < 0 || j + dlt_y > boundary - 1 || mapStat[i + dlt_x][j + dlt_y] != 0) continue;
					// for(int k = 1; k < sheepStat[i][j]; k++){
					// 	Action new_action = {i, j, k, dir.first};
					// 	actions.push_back(new_action);
					// }
					actions.push_back({i, j, sheepStat[i][j] / 2, dir.first});
					actions.push_back({i, j, sheepStat[i][j] - 1, dir.first});
					// actions.push_back({i, j, 1, dir.first});
				}
			}

		}
	}
	return actions;

}

State get_next_state(State &state, Action &action){
	if(Debug){
		cout << "Enter get_next_state" << endl;
		cout << "Action: (" << action[0] << " " << action[1] << " " << action[2] << " " << action[3] << ")" << endl;
		cout << "Current State: " << endl;
		print_state(state);
	}
	vector<vector<int>> mapStat = state.first;
	vector<vector<int>> sheepStat = state.second;
	int root_player = mapStat[action[0]][action[1]];
	int x = action[0];
	int y = action[1];
	int m = action[2];
	int dir = action[3];
	sheepStat[x][y] -= m;
	int dlt_x = dir_map[dir].first;
	int dlt_y = dir_map[dir].second;
	while(x + dlt_x >= 0 && x + dlt_x <= boundary - 1 && y + dlt_y >= 0 && y + dlt_y <= boundary - 1 && mapStat[x + dlt_x][y + dlt_y] == 0){
		x += dlt_x;
		y += dlt_y;
	}
	mapStat[x][y] = root_player;
	sheepStat[x][y] += m;
	return make_pair(mapStat, sheepStat);
}

int cal_connected_region(vector<vector<int>> &mapState, vector<vector<bool>> &visited, int x, int y, int id){
	if(x < 0 || x > boundary - 1 || y < 0 || y > boundary - 1 || visited[x][y] || mapState[x][y] != id) return 0;
	visited[x][y] = true;
	int cnt = 1;
	cnt += cal_connected_region(mapState, visited, x - 1, y, id);
	cnt += cal_connected_region(mapState, visited, x + 1, y, id);
	cnt += cal_connected_region(mapState, visited, x, y - 1, id);
	cnt += cal_connected_region(mapState, visited, x, y + 1, id);
	return cnt;
}

int cal_available_step(vector<vector<int>> &mapStat, vector<vector<int>> &sheepStat , int x, int y, int sheep_num){
	if(sheep_num <= 1) return 1; // 沒有羊可以切
	int cnt = 0;
	for(auto& dir : dir_map){
		int dlt_x = dir.second.first;
		int dlt_y = dir.second.second;
		// 到達邊界或是遇到障礙物
		while(x + dlt_x >= 0 && x + dlt_x <= boundary - 1 && y + dlt_y >= 0 && y + dlt_y <= boundary - 1 && mapStat[x + dlt_x][y + dlt_y] == 0){
			x += dlt_x;
			y += dlt_y;
		}
		// 如果是敵人的旗子且羊的數量大於1
		if(mapStat[x][y] != -1 && sheepStat[x + dlt_x][y + dlt_y] > 1){
			cnt += 1;
		}
		else{
			cnt+= 4; //不會被攻擊到的方向
		}
		
	}
	if(cnt == 0) return -5; // 扣分: 沒有可走的方向
	return cnt;
}

double eva_with_future_possibility(vector<vector<int>> &mapStat, int x, int y, int sheep_num);

double evaluate(int playerID, State state, bool is_oppoent = false){
	double score = 0;
	vector<vector<int>> mapStat = state.first;
	vector<vector<int>> sheepStat = state.second;
	vector<vector<bool>> visited(boundary, vector<bool>(boundary, false));
	float region_ratio = 1.25;
	float available_step_ratio = 0.6;
	float future_possibility_ratio = 0.8;
	int total_num = 0;
	// calculate the score of the map
	for(int i = 0; i < boundary; i++){
		for(int j = 0; j < boundary; j++){
			if(mapStat[i][j] == playerID ){
				score += cal_available_step(mapStat, sheepStat, i, j, sheepStat[i][j]) * sheepStat[i][j] * available_step_ratio;
				if(sheepStat[i][j] > 8) score += eva_with_future_possibility(mapStat, i, j, sheepStat[i][j]) * future_possibility_ratio;
				total_num++;
			}
		}
	}
	if(total_num >= 8){
		for(int i = 0; i < boundary; i++){
			for(int j = 0; j < boundary; j++){
				if(mapStat[i][j] == playerID){
					score += pow(cal_connected_region(mapStat, visited, i, j, playerID),region_ratio);
				}
			}
		}
	}
	
	if (is_oppoent) score = -score;
	return score;
}

void copy_mapState(int src[boundary][boundary], int dst[boundary][boundary]){
	for (int i = 0; i < boundary; i++){
		for (int j = 0; j < boundary; j++){
			dst[i][j] = src[i][j];
		}
	}
}

double eva_with_future_possibility(vector<vector<int>> &mapStat, int x, int y, int sheep_num = 8){
	if (sheep_num <= 1) return 1;
	int score = 0;
	mapStat[x][y] = 5; // 5 代表這裡有羊
	vector<vector<int>> future_map = mapStat;
	
	for(auto& dir : dir_map){
		int dlt_x = dir.second.first;
		int dlt_y = dir.second.second;
		
		if(x + dlt_x < 0 || x + dlt_x > boundary - 1 || y + dlt_y < 0 || y + dlt_y > boundary - 1 || mapStat[x + dlt_x][y + dlt_y] != 0) continue;
		int tmp_x = x;
		int tmp_y = y;
		while(tmp_x + dlt_x >= 0 && tmp_x + dlt_x <= boundary - 1 && tmp_y + dlt_y >= 0 && tmp_y + dlt_y <= boundary - 1 && mapStat[tmp_x + dlt_x][tmp_y + dlt_y] == 0){
			tmp_x += dlt_x;
			tmp_y += dlt_y;
		}
		score += eva_with_future_possibility(future_map, tmp_x, tmp_y, int(sheep_num / 2));
		score += eva_with_future_possibility(future_map, x, y, int(sheep_num / 2));
	}
	if(score == 0) score = -5 * sheep_num;
	return score;

}
Node minmax(State state, int iter_num, int playerID, double alpha, double beta, int index);
Node min_part(State state, int iter_num, int playerID, double alpha, double beta, int index){
	if(Debug) cout << "Enter Min Part, parameter: " << iter_num << " " << playerID << " " << alpha << " " << beta << " " << index << endl;
	double val = INT_MAX;
	Action best_action = Action();
	vector<vector<int>> mapStat = state.first;
	vector<vector<int>> sheepStat = state.second;
	vector<Action> actions = get_action(mapStat, sheepStat, index);
	if(Debug){
		for(auto &action: actions){
			cout << action[0] << " " << action[1] << " " << action[2] << " " << action[3] << endl;
		}
	}
	for(auto &action: actions){
		State next_state = get_next_state(state, action);
		Node tmp_node = minmax(next_state, iter_num - 1, playerID, alpha, beta, index + 1);
		if(tmp_node.first < val){
			val = tmp_node.first;
			best_action = action;
		}
		if(val <= alpha){
			return make_pair(val, best_action);
		}
		beta = min(beta, val);
	}
	return make_pair(val, best_action);
}

Node max_part(State state, int iter_num, int playerID, double alpha, double beta, int index){
	if(Debug) cout << "Enter Max Part, parameter: " << iter_num << " " << playerID << " " << alpha << " " << beta << " " << index << endl;
	double val = -INT_MAX;
	Action best_action = Action();
	vector<vector<int>> mapStat = state.first;
	vector<vector<int>> sheepStat = state.second;
	vector<Action> actions = get_action(mapStat, sheepStat, index);
	if(Debug){
		for(auto &action: actions){
			cout << action[0] << " " << action[1] << " " << action[2] << " " << action[3] << endl;
		}
	}
	for(auto &action: actions){
		State next_state = get_next_state(state, action);
		if (Debug){
			cout << "Next mapStat: " << endl;
			for(int i = 0; i < boundary; i++){
				for(int j = 0; j < boundary; j++){
					cout << next_state.first[i][j] << " ";
				}
				cout << endl;
			}
			cout << "Next sheepStat: " << endl;
			for(int i = 0; i < boundary; i++){
				for(int j = 0; j < boundary; j++){
					cout << next_state.second[i][j] << " ";
				}
				cout << endl;
			}

			
		}
		Node tmp_node = minmax(next_state, iter_num - 1, playerID, alpha, beta, index + 1);
		if(show_scores) cout << "for action: (" << action[0] << " " << action[1] << " " << action[2] << " " << action[3] << ") score: " << tmp_node.first << endl;
		if(tmp_node.first > val){
			val = tmp_node.first;
			best_action = action;
		}
		if(val >= beta){
			if(Debug) cout << "Action: (" << best_action[0] << " " << best_action[1] << " " << best_action[2] << " " << best_action[3] << ")" << endl;
			return make_pair(val, best_action);
		}
		alpha = max(alpha, val);
	}
	if(Debug) cout << "Action: (" << best_action[0] << " " << best_action[1] << " " << best_action[2] << " " << best_action[3] << ")" << endl;

	return make_pair(val, best_action);
	
}
Node minmax(State state, int iter_num, int playerID, double alpha, double beta, int index){
	if (index == 5) index = 1;
	if(iter_num == 0) {
		if(Debug) cout << "Enter evaluate" << endl;
		return make_pair(evaluate(playerID, state, index != playerID), Action());
	}
	if(index == playerID) return max_part(state, iter_num, playerID, alpha, beta, index);
	else return min_part(state, iter_num, playerID, alpha, beta, index);
}

/*
    選擇起始位置
    選擇範圍僅限場地邊緣(至少一個方向為牆)
    
    return: init_pos
    init_pos=<x,y>,代表你要選擇的起始位置
    
*/

std::vector<int> InitPos(int mapStat[12][12])
{
	double score = INT_MIN;
	vector<vector<int>> pos;
	std::vector<int> init_pos;
	init_pos.resize(2);
	vector<vector<int>> mapStat_cpy;
	mapStat_cpy.resize(boundary);
	for(int i = 0; i < boundary; i++){
		mapStat_cpy[i].resize(boundary);
		for(int j = 0; j < boundary; j++){
			mapStat_cpy[i][j] = mapStat[i][j];
		}
	}
	for(int i = 0; i < boundary; i++){
		for(int j = 0; j < boundary; j++){
			if(mapStat[i][j] != 0) continue;
			if(i > 0 && i < boundary && j > 0 && j < boundary){
				if(mapStat[i-1][j] != -1 && mapStat[i+1][j] != -1 && mapStat[i][j-1] != -1 && mapStat[i][j+1] != -1) continue; // 周圍沒有障礙
				else{
					double tmp = eva_with_future_possibility(mapStat_cpy, i, j, 16);
					if(tmp > score){
						pos.clear();
						pos.push_back({i, j});
						score = tmp;
					}
					else if(tmp == score){
						pos.push_back({i, j});
					}
					else continue;
				}
			}
		}
	
	}

	int idx = rand() % pos.size();
	return pos[idx];    
}

/*
	產出指令
    
    input: 
	playerID: 你在此局遊戲中的角色(1~4)
    mapStat : 棋盤狀態, 為 12*12矩陣, 
					0=可移動區域, -1=障礙, 1~4為玩家1~4佔領區域
    sheepStat : 羊群分布狀態, 範圍在0~16, 為 12*12矩陣

    return Step
    Step : <x,y,m,dir> 
            x, y 表示要進行動作的座標 
            m = 要切割成第二群的羊群數量
            dir = 移動方向(1~9),對應方向如下圖所示
            1 2 3
			4 X 6
			7 8 9
*/
std::vector<int> GetStep(int playerID,int mapStat[12][12], int sheepStat[12][12])
{
	// translate the mapStat and sheepStat to the boundary 12*12 matrix with vector type
	vector<vector<int>> mapStat_vec;
	vector<vector<int>> sheepStat_vec;
	mapStat_vec.resize(boundary);
	sheepStat_vec.resize(boundary);
	for(int i = 0; i < boundary; i++){
		mapStat_vec[i].resize(boundary);
		sheepStat_vec[i].resize(boundary);
		for(int j = 0; j < boundary; j++){
			mapStat_vec[i][j] = mapStat[i][j];
			sheepStat_vec[i][j] = sheepStat[i][j];
		}
	}
	if(Debug) cout << "Enter MinMax" << endl;
	Node best_node = minmax(make_pair(mapStat_vec, sheepStat_vec), 3, playerID, -INT_MAX, INT_MAX, playerID);
	vector<int> step = best_node.second;
	

	/*
		Write your code here
	*/
    
    return step;
}

int main()
{
	ios::sync_with_stdio(false);
	cout.tie(0);
	int id_package;
	int playerID;
    int mapStat[12][12];
    int sheepStat[12][12];

	// player initial

	// Debug = true;
	show_scores = true;
	show_step = true;
	int step_num = 1;
	GetMap(id_package,playerID,mapStat);
	std::vector<int> init_pos = InitPos(mapStat);
	if(Debug) cout << "set init_pos: (" << init_pos[0] << " " << init_pos[1] << ")" << endl;
	SendInitPos(id_package,init_pos);

	while (true)
	{
		if (GetBoard(id_package, mapStat, sheepStat))
			break;
		if(show_step) cout << "step: " << step_num++ << "\n";
		std::vector<int> step = GetStep(playerID,mapStat,sheepStat);
		
		SendStep(id_package, step);
	}
}
