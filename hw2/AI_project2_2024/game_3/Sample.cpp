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
bool show_map = false;
bool show_dir = false;
int step_num = 1;
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
	// cout << "sheepStat: " << endl;
	// for(int i = 0; i < boundary; i++){
	// 	for(int j = 0; j < boundary; j++){
	// 		cout << sheepStat[i][j] << " ";
	// 	}
	// 	cout << endl;
	// }
}
bool isnt_bound(int x, int y){
	return x >= 0 && x < boundary && y >= 0 && y < boundary;
}
vector<Action> get_action(vector<vector<int>> &mapStat, vector<vector<int>> &sheepStat, int playerID){
	vector<Action> actions;
	for(int i = 0; i < boundary; i++){
		for(int j = 0; j < boundary; j++){
			if(mapStat[i][j] != playerID) continue;
			else if (sheepStat[i][j] >1){
				for(auto &dir: dir_map){
					int dlt_x = dir.second.first;
					int dlt_y = dir.second.second;
					if(!isnt_bound(i + dlt_x, j + dlt_y) || mapStat[i + dlt_x][j + dlt_y] != 0) continue; // 不能移動
					// for(int k = 1; k < sheepStat[i][j]; k++){
					// 	Action new_action = {i, j, k, dir.first};
					// 	actions.push_back(new_action);
					// }
					actions.push_back({i, j, sheepStat[i][j] / 2, dir.first});
					// actions.push_back({i, j, sheepStat[i][j] - 1, dir.first});
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
	while(isnt_bound(x + dlt_x, y + dlt_y) && mapStat[x + dlt_x][y + dlt_y] == 0){
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
	if(sheep_num <= 1) return 2; // 沒有羊可以切
	int cnt = 0;
	int save_place = 0;
	for(auto& dir : dir_map){
		int dlt_x = dir.second.first;
		int dlt_y = dir.second.second;
		int tmp_x = x;
		int tmp_y = y;
		if(!isnt_bound(tmp_x + dlt_x, tmp_y + dlt_y) || mapStat[tmp_x + dlt_x][tmp_y + dlt_y] != 0) continue; // 無法移動
		// 檢查是否可以移動
		while( isnt_bound(tmp_x + dlt_x, tmp_y + dlt_y) && mapStat[tmp_x + dlt_x][tmp_y + dlt_y] == 0){
			tmp_x += dlt_x;
			tmp_y += dlt_y;
		}		
		if(!isnt_bound(tmp_x + dlt_x, tmp_y + dlt_y) || mapStat[tmp_x + dlt_x][tmp_y + dlt_y] == -1 || mapStat[tmp_x + dlt_x][tmp_y + dlt_y] == mapStat[x][y]|| sheepStat[tmp_x + dlt_x][tmp_y + dlt_y] <= 1){
			save_place += 1; // 不是牆壁或是自己的地盤或是敵人羊群數量不足 = 安全的地方
		}
		cnt += 1; // 可以移動的方向
	}
	if(show_dir) cout << "for (" << x << " " << y << ") available place: " << cnt << " save_place: " << save_place << endl;
	if(cnt == 0) return -50; // 扣分: 沒有可走的方向
	else if(cnt < sheep_num && save_place < cnt / 2) return -10; // 扣分: 可走的方向不足且容易被封死
	else if(cnt < sheep_num && save_place >= cnt / 2) return -5; // 加分: 可走的方向不足但不容易被封死
	else if(save_place <= cnt / 2) return 7; // 加分: 可走的方向足夠但容易被封死
	else if(save_place > cnt / 2) return 10; // 加分: 可走的方向足夠且不容易被封死
	else return 0; // 不加分也不扣分(不會發生
}


double evaluate(int playerID, State state, bool is_oppoent = false, bool region_flag = false){
	// randomly setthe init score between 10 and -10
	double score = rand() % 10 - 5 ;
	vector<vector<int>> mapStat = state.first;
	vector<vector<int>> sheepStat = state.second;
	vector<vector<bool>> visited(boundary, vector<bool>(boundary, false));
	map<int, int> region_cnt;
	float region_ratio = 1.25;
	// float available_step_ratio = 0.95;
	float sepreate_ratio = 1.75;
	// calculate the score of the map
	for(int i = 0; i < boundary; i++){
		for(int j = 0; j < boundary; j++){
			if(mapStat[i][j] == playerID ){
				// score += (cal_available_step(mapStat, sheepStat, i, j, sheepStat[i][j]) * sheepStat[i][j] * available_step_ratio);
				if(region_flag) score += pow(cal_connected_region(mapStat, visited, i, j, playerID),region_ratio);
				if(region_cnt.find(sheepStat[i][j]) == region_cnt.end()) region_cnt[sheepStat[i][j]] = 1;
				else region_cnt[sheepStat[i][j]] += 1;
			}
		}
	}
	for(auto& region: region_cnt){
		score += pow(region.second, sepreate_ratio);
	}
	
	if (is_oppoent) score = -1 * score;
	return score;
}

double eva_with_future_possibility(vector<vector<int>> mapStat, int x, int y, int sheep_num = 8){
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

Node minmax(State state, int iter_num, int playerID, double alpha, double beta, int index){
	if (index == 5) index = 1;
	if(iter_num <= 0) {
		if(Debug) cout << "Enter evaluate" << endl;
		double score = evaluate(playerID, state, index != playerID, step_num >= 8);
		if(score <= INT_MIN) score = -999999;
		else if(score >= INT_MAX) score = 999999;
		return make_pair(score, Action());
	}
	if(index == playerID) { // Max part
		if(Debug) cout << "Enter Max Part, parameter: " << iter_num << " " << playerID << " " << alpha << " " << beta << " " << index << endl;
		double val = INT_MIN;
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
			if (show_map){
				cout << "Next State: " << endl;
				print_state(next_state);				
			}
			Node tmp_node = minmax(next_state, iter_num - 1, playerID, alpha, beta, index + 1);
			if(show_scores) cout << "for action: (" << action[0] << " " << action[1] << " " << action[2] << " " << action[3] << ") score: " << tmp_node.first << endl;
			if(tmp_node.first == INT_MIN) continue;
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
	else { // Min part
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
			if(tmp_node.first == INT_MAX) continue;
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
}

/*
    選擇起始位置
    選擇範圍僅限場地邊緣(至少一個方向為牆)
    
    return: init_pos
    init_pos=<x,y>,代表你要選擇的起始位置
    
*/

vector<int> InitPos(int mapStat[12][12])
{
	double score = INT_MIN;
	vector<vector<int>> pos;
	vector<int> init_pos;
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
			if(isnt_bound(i,j)){
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
vector<int> GetStep(int playerID,int mapStat[12][12], int sheepStat[12][12])
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
	Node best_node = minmax(make_pair(mapStat_vec, sheepStat_vec), 6, playerID, INT_MIN, INT_MAX, playerID);
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

	Debug = false;
	show_scores = false;
	show_step = false;
	show_map = false;	
	show_dir = false;
	GetMap(id_package,playerID,mapStat);
	vector<int> init_pos = InitPos(mapStat);
	if(Debug) cout << "set init_pos: (" << init_pos[0] << " " << init_pos[1] << ")" << endl;
	SendInitPos(id_package,init_pos);

	while (true)
	{
		if (GetBoard(id_package, mapStat, sheepStat))
			break;
		// hide other player's sheep number start
		for (int i = 0; i < 12; i++)
			for (int j = 0; j < 12; j++)
				if (mapStat[i][j] != playerID)
					sheepStat[i][j] = 0;
		// hide other player's sheep number end 
		if(show_step) cout << "step: " << step_num << "\n";
		vector<int> step = GetStep(playerID,mapStat,sheepStat);
		step_num++;
		
		SendStep(id_package, step);
	}
}