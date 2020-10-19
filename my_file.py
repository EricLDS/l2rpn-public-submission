from grid2op.Agent import BaseAgent
from grid2op.Reward import BaseReward

import os
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch

N_TIME_S = 0
N_TIME_E = 5
N_GEN_P_S = 6
N_GEN_P_E = 68
N_GEN_QV_S = 68
N_GEN_QV_E = 192
N_LOAD_P_S = 192
N_LOAD_P_E = 291
N_LOAD_QV_S = 291
N_LOAD_QV_E = 489
N_LINE_OR_P_S = 489
N_LINE_OR_P_E = 675
N_LINE_OR_QVA_S = 675
N_LINE_OR_QVA_E = 1233
N_LINE_EX_P_S = 1233
N_LINE_EX_P_E = 1419
N_LINE_EX_QVA_S = 1419
N_LINE_OR_QVA_E = 1977
N_RHO_S = 1977
N_RHO_E = 2163
N_LINE_STAT_S = 2163
N_LINE_STAT_E = 2349
N_TINE_OF_S = 2349
N_TINE_OF_E = 2535
N_TOPO_S = 2535
N_TOPO_E = 3068
N_CD_LINE_S = 3068
N_CD_LINE_E = 3254
N_CD_SUB_S = 3254
N_CD_SUB_E = 3372
N_TIME_MANT_S = 3372
N_TIME_MANT_E = 3558
N_TIME_DURA_S = 3558
N_TIME_DURA_E = 3744
N_TAR_DISP_S = 3744
N_TAR_DISP_E = 3806
N_ACTL_DISP_S = 3806
N_ACTL_DISP_E = 3869

n_actions = 166
n_features = 719

class Network(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(n_features, 512)
        self.l2 = nn.Linear(512, 512)
        self.l3 = nn.Linear(512, 256)
        self.l4 = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.l4(x)
        return x
    


class MyAgent(BaseAgent):
    """
    The template to be used to create an agent: any controller of the power grid is expected to be a daughter of this
    class.
    """
    def __init__(self, action_space,net_dict,effective_topo,sub_info):
        """Initialize a new agent."""
        BaseAgent.__init__(self, action_space=action_space)
        self.policy_net = Network()
        self.policy_net.load_state_dict(net_dict)
        self.policy_net.eval()
        self.effective_topo = effective_topo
        self.sub_info = sub_info
        
    def get_state(self,observation):
        obs_as_vect = observation.to_vect()
        states = np.append(obs_as_vect[N_RHO_S:N_RHO_E],obs_as_vect[N_TOPO_S:N_TOPO_E]) # rho & topo
        return states
        
    def emergency_monitor(self,obs):
        has_overflow = False
        obs_as_vect = obs.to_vect()
        rho = obs_as_vect[N_RHO_S:N_RHO_E]
        if np.amax(rho)>=1.0:
            has_overflow = True
        return has_overflow 
    
    def normal_operation(self, obs, N_actions):
        # if there is a line disconnected and cooldown is OK, then reconnect it
        line_status = obs.line_status
        line_CD = obs.time_before_cooldown_line
        reconnected_id = -1
        
        grid2op_action = self.action_space({}) # do nothing as baseline
        action_idx = N_actions-1
        
        if 0 in line_status: # if there is a line disconnected
            # detect which line is disconnected and cooldown is OK
            disconnected_line_id = np.asarray(np.where(line_status == 0))[0]
            for i in range(np.size(disconnected_line_id)):
                line_id = disconnected_line_id[i]
                if line_CD[line_id]==0:
                    reconnected_id = disconnected_line_id[i]
                    break
                
            if reconnected_id != -1: # all disconnected line in cooldown
                grid2op_action = self.action_space({"set_line_status": [(reconnected_id, 1)]})
        
        return action_idx, grid2op_action   
    
    def act(self, observation, reward, done):
        if self.emergency_monitor(observation):
            # cal Q_val
            state = torch.FloatTensor([self.get_state(observation)])
            Q_val = self.policy_net(Variable(state, requires_grad=True).type(torch.FloatTensor)).data
            Q_val = Q_val.detach().numpy()
            Q_sorted = np.argsort(-Q_val)[0]
            # action_prd = self.policy_net(Variable(state, requires_grad=True).type(torch.FloatTensor)).data.max(1)[1].view(1, 1)
            action_buff = Q_sorted[0:50] # select top Q_vals index
            
            obs_as_vect = observation.to_vect()
            line_status = obs_as_vect[N_LINE_STAT_S:N_LINE_STAT_E]
            line_CD = obs_as_vect[N_CD_LINE_S:N_CD_LINE_E]
            sub_CD = obs_as_vect[N_CD_SUB_S:N_CD_SUB_E]
            topo = obs_as_vect[N_TOPO_S:N_TOPO_E]
            reconnected_id = -1
            
            max_rw = -1.0
            action_selected = self.action_space({}) # do nothing as baseline
            for i in range (0,len(action_buff)):
                action_index = action_buff[i]
                if 0 in line_status: # if there is a line disconnected
                    disconnected_line_id = np.asarray(np.where(line_status == 0))[0]
                    for i in range(np.size(disconnected_line_id)):
                        line_id = disconnected_line_id[i]
                        if line_CD[line_id]==0:
                            reconnected_id = disconnected_line_id[i]
                            break
                    if reconnected_id == -1: # line in cooldown time
                        if action_index<n_actions-1: # re-config
                            sub_id = self.effective_topo[action_index,533]
                            idx_node_start = self.sub_info[sub_id,1]
                            idx_node_end = self.sub_info[sub_id,1]+self.sub_info[sub_id,0]
                            sub_topo = topo[idx_node_start:idx_node_end]
                            if sub_CD[sub_id] != 0 or (-1 in sub_topo):# sub in CD or node disconnected
                                action = self.action_space({})
                            else:
                                target_topo = self.effective_topo[action_index,idx_node_start:idx_node_end]
                                action = self.action_space({"set_bus": {"substations_id": [(sub_id, target_topo)]}})
                        else:# do nothing
                            action = self.action_space({})# do nothing is selected 
                    else: # reconnect line
                        # print("reconnecting transmission line:",reconnected_id)
                        if action_index<n_actions-1: # re-config + reconnect
                            sub_id = self.effective_topo[action_index,533]
                            idx_node_start = self.sub_info[sub_id,1]
                            idx_node_end = self.sub_info[sub_id,1]+self.sub_info[sub_id,0]
                            sub_topo = topo[idx_node_start:idx_node_end]
                            if sub_CD[sub_id] != 0 or (-1 in sub_topo):# sub in CD or node disconnected
                                action = self.action_space({ "set_line_status": [(reconnected_id, 1)],
                                                            "set_bus": {"lines_or_id": [(reconnected_id, 1)],
                                                                        "lines_ex_id": [(reconnected_id, 1)]}})
                            else: # re-config + reconnect
                                target_topo = self.effective_topo[action_index,idx_node_start:idx_node_end]
                                action = self.action_space({ "set_line_status": [(reconnected_id, 1)],
                                                            "set_bus": {"lines_or_id": [(reconnected_id, 1)],
                                                                        "lines_ex_id": [(reconnected_id, 1)],
                                                                        "substations_id": [(sub_id, target_topo)]}})
                        else: # do nothing is selected, just reconnect
                            action = self.action_space({"set_line_status": [(reconnected_id, 1)],
                                                       "set_bus": {"lines_or_id": [(reconnected_id, 1)],
                                                                   "lines_ex_id": [(reconnected_id, 1)]}})
                else: # no line disconnected
                    if action_index<n_actions-1: # re-config
                        sub_id = self.effective_topo[action_index,533]
                        idx_node_start = self.sub_info[sub_id,1]
                        idx_node_end = self.sub_info[sub_id,1]+self.sub_info[sub_id,0]
                        sub_topo = topo[idx_node_start:idx_node_end]
                        if sub_CD[sub_id] != 0 or (-1 in sub_topo):# sub in CD or node disconnected
                                action = self.action_space({})
                        else:
                            target_topo = self.effective_topo[action_index,idx_node_start:idx_node_end]
                            action = self.action_space({"set_bus": {"substations_id": [(sub_id, target_topo)]}})
                    else: # do nothing is selected 
                        action = self.action_space({})# do nothing
                        
                obs_sim,rw_sim,done_sim,info_sim = observation.simulate(action)
                
                if not done_sim and rw_sim > max_rw:
                    max_rw = rw_sim
                    action_selected = action
                    
        # no overflow
        else: 
            _, action_selected = self.normal_operation(observation,n_actions)
        
        return action_selected


class reward(BaseReward):
    """
    if you want to control the reward used by the envrionment when your agent is being assessed, you need
    to provide a class with that specific name that define the reward you want.

    It is important that this file has the exact name "reward" all lowercase, we apologize for the python convention :-/
    """
    def __init__(self):
        BaseReward.__init__(self)

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if has_error or is_illegal or is_ambiguous or is_done:
            rw = 0.0
        else:
            rw = 0.0
            obs = env.current_obs
            obs_as_vect = obs.to_vect()
            load_P = np.sum(obs_as_vect[192:291])
            gen_P = np.sum(obs_as_vect[6:68])
            line_status = obs_as_vect[2163:2349]
            rho = obs_as_vect[1977:2163]
            maintance_time = obs_as_vect[3372:3558]
            
            rw = load_P/gen_P
            for i in range(0,186):
                if line_status[i]==0 and maintance_time[i] != 0: rw += -0.1 # overflow disconnection
                elif rho[i] >= 1.0: rw += -0.05 # overflow
                
        return rw

def make_agent(env, submission_dir):
    """
    This function will be used by codalab to create your agent. It should accept exactly an environment and a path
    to your sudmission directory and return a valid agent.
    """
    effective_topo = np.load(os.path.join(submission_dir, "weights", "effective_topo.npy"))
    sub_info = np.load(os.path.join(submission_dir, "weights", "sub_info.npy"))
    net_dict = torch.load(os.path.join(submission_dir, "weights", "DQN_weights_0906.h5"),map_location='cpu')
    res = MyAgent(env.action_space,net_dict,effective_topo,sub_info)
    return res
