import os
import sys
import time
import subprocess
from PIL import Image
import numpy as np
import torch
import yaml
from torchvision import transforms
from experiment import VAEXperiment
from models import *

import gymnasium as gym
from gymnasium import ObservationWrapper
from gymnasium.wrappers import PixelObservationWrapper, FrameStack
from gymnasium.spaces import Box, Discrete

from stable_baselines3 import SAC, PPO, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy


###
# when you restart the with the same names, delete data, VAE, RL end of loop saves, logs and rewards
###

# get the vae
def get_vae(version='version_0',log_directory='logs/BCE_test_VAE_1/MSSIMVAE/',
            hparam_path = "configs/bces_vae.yaml"):
    #model_path= log_directory+'/'+version+'/hparams.yaml'
    ckpt_path=log_directory+'/'+version+'/checkpoints/last.ckpt'

    config = yaml.safe_load(open(hparam_path))
    model = vae_models[config['model_params']['name']](**config['model_params'])
    ckpt = torch.load(ckpt_path)
    experiment = VAEXperiment(model, config['exp_params'])
    experiment.load_state_dict(ckpt['state_dict'])      
    vae = experiment.model
    return vae

#Make a funciton to create environment, this allows to vectorize it
def make_env(env_id: str = "MountainCarContinuous-v0", rank: int = 0, seed: int = 42, 
            data_dir: str = "Data/MountainCar/test2", collect_frames: bool = True, env_iterator: int = 0,
            vae_version: int = 0, latent_dim: int = 1,
            vae_directory: str = 'logs/MountainCar/BCE_test_VAE_1/MSSIMVAE/',
            hparam_path: str = "configs/bces_no_pretrained.yaml",
            agent_name: str = ''):
    def _init():
        save_path= data_dir+'/train_env_id_'+str(env_iterator)+'_nenv_'+str(rank)+'_'
        vae = get_vae(version='version_'+str(vae_version),
                      log_directory = vae_directory,
                      hparam_path = hparam_path)
        
        env = gym.make(env_id,
                    render_mode ='rgb_array')
        
        seed = 42
        env.reset(seed=seed + rank)
        env = PixelObservationWrapper(env)
        if collect_frames:
            env = frame_saver(env, save_path, agent_name=agent_name)
        env = VAE_ENC(env, vae, latent_dim)
        env = FrameStack(env, num_stack=2)
        env = Monitor(env)
        return env
    set_random_seed(seed)
    return _init

#to save the encodings of all pictures used during training of the vae
def save_known_universe(vae, observation_dir, save_dir, save_name, batch_size):
    os.makedirs(save_dir, exist_ok=True)
    
    # Collect filenames of all JPEG images in the observation directory
    image_filenames = [filename for filename in os.listdir(observation_dir) if filename.lower().endswith('.jpeg')]

    latent_vectors = []
      
    for i in range(0, len(image_filenames), batch_size):
        batch_filenames = image_filenames[i:i + batch_size]
        batch_images = [Image.open(os.path.join(observation_dir, filename)) for filename in batch_filenames]
        batch_encodings = encode_images_batch(vae, batch_images)
        latent_vectors.extend(batch_encodings)

    # Save latent vectors as a CSV file
    file_name = save_name + '_latent_vectors.csv'
    csv_path = os.path.join(save_dir, file_name)
    np.savetxt(csv_path, np.array(latent_vectors), delimiter=',')


    
def encode_images_batch(vae, images):
    val_transforms = transforms.Compose([transforms.ToTensor(),
        #transforms.RandomHorizontalFlip(),
        AddGaussianNoise(0, 0.1),
        transforms.Resize((64,64)),
        #transforms.Grayscale(),
        #transforms.Normalize(self.mean, self.std),
        ])
    processed_images = torch.stack([val_transforms(img) for img in images])
    enc = vae.encode(processed_images)
    enc = np.array([tensor.detach().cpu().numpy() for tensor in enc])
    return enc[0]

def eval_agent(agent, env, n_eval_episodes):
    total_rewards = []
    device = 'cpu'
    policy = agent.policy.to(device)  # Move observation to the same device as the model
    observation_space = env.observation_space
    print('observation space:', observation_space)
    
    for episode in range(n_eval_episodes):
        observation, info = env.reset()

        episode_reward = 0
        done = False

        while not done: 
            action = policy.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            if not observation_space.contains(observation):
                print("Observation is not valid:", observation)
            episode_reward += reward
            if terminated:
                done = True
            if truncated:
                done = True                   

        total_rewards.append(episode_reward)

    total_rewards = np.array(total_rewards)
    
    agent.policy.to('cuda')
    return np.mean(total_rewards), np.std(total_rewards), total_rewards





def main():
    #init
    print("init")
    #collect episodes with random actions for picture data
    #no training
    data_name='SAC_l1_test1'
    save_path='Data/MountainCar/'+data_name+'/'
    os.makedirs(save_path, exist_ok=True)

    num_of_episodes = 0

    env = gym.make("MountainCarContinuous-v0",
                    render_mode ='rgb_array')

    i=0
    for episode in range(num_of_episodes):
        observation, info = env.reset()
        done = False
        while not done: 
            action= env.env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            #state, reward, terminated, truncated, info = env.step(action)
            #print(terminated, truncated)
            if terminated:
                done = True
            if truncated:
                done = True
            current_frame = env.render()

            i+=1
            im = Image.fromarray(np.array(current_frame))
            im.save(save_path+data_name+'_'+str(i)+'.jpeg')

    print('collected frames from', num_of_episodes,"episodes with random action in dir:", save_path)



    #Double loop vae and RL training
    target_reward = 80 #90
    current_reward = -4#-990
    last_reward = -8#-999 #just so the getting worse counter dosent count up in the very first round
    vae_version = 9#0
    n_envs = 1  #how many envs for parallel training #there will be up to n_envs*n_steps 'too many' observations/trainings steps for the rl agent
    getting_worse = 1#0 # track if the agent is getting worse 
    train_new_vae = True # create the first agent only in the first round
    agent_model_dir = "RLmodels/MountainCarContinuous-v0/Double_loop"#where to save the RL agents
    agent_log_dir = agent_model_dir+"/logs" #where to log RL progress
    vae_name = "BCE_VAE_l1_test1_SAC"
    vae_directory = 'logs/MountainCar/'+vae_name+'/MSSIMVAE/' # directory for versions of the vae
    latent_save_dir = agent_model_dir + "/latent_space_encodings/"+vae_name
    
    #num of resets
    n_rl_resets = 2#-1
    n_vae_resets = 1#-1
    reset_num_timesteps = False #True
    env_iter = 9#0

    cont = False

    

    print("starting double loop")
    print("target reward", target_reward)
    print("n_envs", n_envs)
    print("RL logging and saveing to", agent_model_dir)
    print("vae_dir", vae_directory)

    while current_reward < target_reward:
        agent_name = vae_name+"_v"+str(vae_version)+"__"+str(n_vae_resets)+"vae_resets__"+str(n_rl_resets)+"rl_resets__"

        num_old_obs = len(os.listdir(save_path))
        print("num of collected obs", num_old_obs)
        num_new_obs = 0
        if cont:
            ## do maybe 5% or soemething,
            #do percentage of old reward, but if it is close to 0, that percentage is low, so also hava a fixed term, adequatly picked for the environment
            #doesnt work so good with negative #do 3% + 2, so min stagnation is if current_reward(=0)<last_reward+2, max stagnation current_reward(=99,-99)<last_reward+5
            #just do + 4
            #do +1 to crack assymptotic improvements that never surpass a threshold, aka -1, -0.1, -0.01, -0.001 gets better but is also stuck
            if current_reward <= last_reward + 2 :
                getting_worse += 1 
            else:
                getting_worse = 0
            print('getting_worse count:', getting_worse)
            #stagnation threshold to train new vae from scratch
            if getting_worse >= 3:
                train_new_vae = True
            
            #make eval callback to count up the getting worse
            #or make brake point of -1/3 of inicial value
            if getting_worse >= 2 or train_new_vae:

                #train vae from scratch for first run or if its very stagnant
                if train_new_vae:
                    print("training new vae")
                    # get the vae directories
                    config_path = "configs/bces_no_pretrained_sac.yaml"
                    
                    
                    #os.system("python run.py  -c configs/bces_no_pretrained.yaml")
                    command = [sys.executable, "run.py", "-c", "configs/bces_no_pretrained_sac.yaml"]
                    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    print("Return code:", result.returncode)
                    print("Have {} bytes in stdout:\n{}".format(len(result.stdout), result.stdout.decode('utf-8')))
                    n_vae_resets += 1
                    train_new_vae = False
                else:
                    print("continue with old vae")
                    command = [sys.executable, "run.py", "-c", "configs/bces_continue_sac.yaml"]
                    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    print("Return code:", result.returncode)
                    print("Have {} bytes in stdout:\n{}".format(len(result.stdout), result.stdout.decode('utf-8')))

                print('making new agent and env')
                #make
                env = DummyVecEnv([make_env(env_id = "MountainCarContinuous-v0", rank=i, 
                data_dir = save_path, collect_frames = True, env_iterator = env_iter,
                vae_version = vae_version,
                vae_directory = vae_directory,
                hparam_path = config_path,
                agent_name = agent_name) for i in range(n_envs)])

                
                #new agent from scratch 

                # Tuned
                
                agent = SAC(
                    env = env,
                    policy = 'MlpPolicy',
                    learning_rate= 0.0003,
                    buffer_size= 50000,
                    batch_size= 512,
                    ent_coef= 0.1,
                    train_freq= 32,
                    gradient_steps= 32,
                    gamma= 0.9999,
                    tau= 0.01,
                    learning_starts= 0,
                    use_sde= True,
                    policy_kwargs= dict(log_std_init=-3.67, net_arch=[64, 64]),
                    tensorboard_log=agent_log_dir
                )


                
                n_rl_resets += 1
                #reset stepcount in training
                #reset_num_timesteps = True
                #ToDo
                #if new train with old trajectoires Immitation learning
            else:
                print("continiue with old vae and rl agent")
                #continiue training of vae


                command = [sys.executable, "run.py", "-c", "configs/bces_continue_sac.yaml"]
                result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                print("Return code:", result.returncode)
                print("Have {} bytes in stdout:\n{}".format(len(result.stdout), result.stdout.decode('utf-8')))

                #os.system("python run.py  -c configs/bces_continiue_training.yaml")
                #config_path = "configs/bces_continiue_training.yaml"
                
                #make new env with current vae
                env = DummyVecEnv([make_env(env_id = "MountainCarContinuous-v0", rank=i, 
                    data_dir = save_path, collect_frames = True, env_iterator = env_iter,
                    vae_version = vae_version,
                    vae_directory = vae_directory,
                    hparam_path = config_path,
                    agent_name = agent_name) for i in range(n_envs)])
                
                #continue with old agent
                #update env of the agent
                agent.env = env
        else:
            print("continue with old vae")
            command = [sys.executable, "run.py", "-c", "configs/bces_continue_sac.yaml"]
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("Return code:", result.returncode)
            print("Have {} bytes in stdout:\n{}".format(len(result.stdout), result.stdout.decode('utf-8')))
            print('RLmodels/MountainCarContinuous-v0/Double_loop/end_of_loop_save/BCE_VAE_l1_test1_SAC_v8__1vae_resets__2rl_resets__.zip')
            agent = PPO.load('RLmodels/MountainCarContinuous-v0/Double_loop/end_of_loop_save/BCE_VAE_l1_test1_SAC_v8__1vae_resets__2rl_resets__.zip')
            config_path = "configs/bces_no_pretrained_sac.yaml"
            env = DummyVecEnv([make_env(env_id = "MountainCarContinuous-v0", rank=i, 
                    data_dir = save_path, collect_frames = False, env_iterator = env_iter,
                    vae_version = vae_version,
                    vae_directory = vae_directory,
                    hparam_path = config_path,
                    agent_name = agent_name) for i in range(n_envs)])
            agent.env = env    
            #continue with old agent
            #update env of the agent
            # agent.env = env
            # agent.set_env(agent.env, force_reset=True)
            # num_of_steps = 32306
            # print('train for ',num_of_steps,)
            
            # agent.learn(total_timesteps=num_of_steps,reset_num_timesteps=reset_num_timesteps, tb_log_name=agent_name)
            cont = True
            
        
        if cont:
            #save all pictures used during training in latent space
            print('save latent space')
            latent_save_name = vae_name+ '_v'+str(vae_version)
            vae = get_vae(version='version_'+str(vae_version),
                        log_directory = vae_directory,
                        hparam_path = config_path)    
            save_known_universe(vae, observation_dir = save_path, save_dir = latent_save_dir, save_name=latent_save_name, batch_size=256)

        


            agent.set_env(agent.env, force_reset=True)
            #hack solution to have every obs from training saved
            #do a inital run 
            agent_name = vae_name+"_v"+str(vae_version)+"__"+str(n_vae_resets)+"vae_resets__"+str(n_rl_resets)+"rl_resets__"
            #agent.learn(total_timesteps=1,reset_num_timesteps=reset_num_timesteps, tb_log_name=agent_name)
            while num_new_obs < num_old_obs:
                num_missing_obs = num_old_obs - num_new_obs
                print('missing_obs', num_missing_obs)
                num_of_steps = int(num_missing_obs) #int((num_missing_obs / n_steps) / n_envs)
                print("training rl agent for", num_of_steps)
                
                ## make callback to check for longtime decline in later itarations where training takes 50.000+ steps
                #callback = CheckpointCallback(save_freq=10000, save_path=agent_model_dir)
                agent.learn(total_timesteps=num_of_steps,reset_num_timesteps=reset_num_timesteps, tb_log_name=agent_name)


                #agent.learn(total_timesteps=num_of_steps,reset_num_timesteps=reset_num_timesteps, callback=callback, tb_log_name=agent_name)

                num_new_obs = len(os.listdir(save_path)) - num_old_obs
                print(len(os.listdir(save_path)), '-', num_old_obs, '=', num_new_obs)
                reset_num_timesteps = False

        #make eval env no save
        print("evaluate rl agent")

        # eval_env = DummyVecEnv([make_env(env_id = "MountainCarContinuous-v0", rank=i, 
        #         data_name = "test2", collect_frames = False,
        #         vae_version = vae_version,
        #         vae_directory = vae_directory,
        #         hparam_path = config_path) for i in range(n_envs)])

        eval_env = make_env(env_id = "MountainCarContinuous-v0", rank=0, 
                data_dir = save_path, collect_frames = False,
                vae_version = vae_version,
                vae_directory = vae_directory,
                hparam_path = config_path)()


        
        #eval
        
        n_eval_ep= int((num_of_steps/999)/2)    #more training, more eval
        if n_eval_ep < 10:                  #at least 10 episodes of eval
            n_eval_ep = 10 
        #mean_reward, std_reward = evaluate_policy(agent, eval_env, n_eval_episodes=10, deterministic=True)
        #print(agent_name," mean reward, std ", str(mean_reward), str(std_reward))
        mean_reward, std_reward, reward_list = eval_agent(agent, eval_env, n_eval_episodes= n_eval_ep)
        print('all rewards', reward_list)
        print(agent_name," mean reward, std ", str(mean_reward), str(std_reward))
        if (mean_reward > target_reward) and n_eval_ep < 20:
            mean_reward, std_reward, reward_list = eval_agent(agent, eval_env, n_eval_episodes= 20)
        last_reward = current_reward
        current_reward = mean_reward

        #save the model
        agent.save("RLmodels/MountainCarContinuous-v0/Double_loop/end_of_loop_save/"+agent_name)
        #save the rewards
        np.savetxt('RLmodels/MountainCarContinuous-v0/Double_loop/rewards/'+agent_name+'_rewards.csv', reward_list, delimiter=',', fmt='%.4f')


        #count up vae version for naming
        vae_version += 1
        env_iter += 1 
        cont = True
        #inititally =0 want to increase for the next iteraiton
        

#save frame wrapper class for env
class frame_saver(ObservationWrapper):
    def __init__(self, env,
                 collect_frames_dir = None,
                 start_index = 0,
                 agent_name =''):
        super().__init__(env)
        
        self.collect_frames_dir = collect_frames_dir
        self.frame_idx = start_index
        self.agent_name = agent_name
                
        
    def observation(self, obs):
        frame = obs['pixels']#.to('cuda')
        if self.collect_frames_dir != None:
            im = Image.fromarray(np.array(frame))
            im.save(self.collect_frames_dir+'_'+self.agent_name+'_'+str(self.frame_idx)+'.jpeg')
            self.frame_idx += 1
        return obs

# to add Gaussian noise to the observations
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.1):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

#create VAE wrapper
class VAE_ENC(ObservationWrapper):
    def __init__(self, env, vae, latent_dim,
                 mean=0,std=0.1,
                 size=(64,64),
                 start_index = 0):
        super().__init__(env)
        #new obs space with std
        #self.observation_space = Box(shape=(2, latent_dim), low=-np.inf, high=np.inf)
        #just mean
        self.observation_space = Box(shape=(latent_dim,), low=-np.inf, high=np.inf)
        
        self.vae = vae
        #transforms
        self.mean = mean
        self.std = std
        self.size = size
        
        self.frame_idx = start_index
        
        
        
    def observation(self, obs):
        #get frame
        #print(obs)
        frame = obs['pixels']#.to('cuda')
        #transform for VAE
        val_transforms = transforms.Compose([transforms.ToTensor(),
        #transforms.RandomHorizontalFlip(),
        AddGaussianNoise(self.mean, self.std),
        transforms.Resize(self.size),
        #transforms.Grayscale(),
        #transforms.Normalize(self.mean, self.std),
        ])
        frame = val_transforms(frame) #(c,h,w)
        frame = torch.unsqueeze(frame, 0)#.to(self.device) #make it (1,c,h,w)
        enc = self.vae.encode(frame)    
        enc = np.array([tensor.detach().cpu().numpy() for tensor in enc])
        #with std
        #enc = np.array([enc[0][0], enc[1][0]]) ## mu, std #  give only mu?
        #just mean
        enc = np.array(enc[0][0])

        return enc



if __name__ == '__main__':
    main()



            # # Tuned
            # print("making new rl agent")
            # n_steps = 100 
            # agent = A2C(
            #     env = env,
            #     n_steps= n_steps,           
            #     policy='MlpPolicy',
            #     ent_coef= 0.0,
            #     use_sde=True,
            #     sde_sample_freq = 16,
            #     policy_kwargs= dict(log_std_init=0.0, ortho_init=False),
            #     tensorboard_log=agent_log_dir)