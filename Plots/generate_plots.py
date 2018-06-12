import pickle
import matplotlib.pyplot as plt

'''Swimmer Environment Plots'''
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
pickle_off = open("Swimmer_EVO_rewards_testing_LN_final.p", "rb")
emp = pickle.load(pickle_off)
EA, = plt.plot(emp, label='ERL')

pickle_off = open("SwimmerV2_ERL_fitness_2000.p", "rb")
emp = pickle.load(pickle_off)
ERL, = plt.plot(emp, label="EA")

pickle_off = open("SwimmerV2_RLonly_rewards_2000.p", "rb")
emp = pickle.load(pickle_off)
DDPG, = plt.plot(emp, label="DDPG")

plt.legend(handles=[ERL, EA, DDPG])
plt.xlabel("Million steps", fontsize=14)
plt.ylabel("Rewards", fontsize=14)
plt.title("Swimmer EA vs DDPG vs ERL", fontsize=16)

plt.show()



''' Swimmer Environment Motivation'''
# pickle_off = open("SwimmerV2_ERL_RL_rewards_2000.p", "rb")
# emp = pickle.load(pickle_off)
# DDPG_with_ERL_experienve, = plt.plot(emp, label='DDPG w/ERL experience')
#
# pickle_off = open("SwimmerV2_RLonly_rewards_2000.p", "rb")
# emp = pickle.load(pickle_off)
# DDPG_only, = plt.plot(emp, label='DDPG only')
#
# plt.legend(handles=[DDPG_only, DDPG_with_ERL_experienve])
# plt.xlabel("Million steps", fontsize=14)
# plt.ylabel("Rewards", fontsize=14)
# plt.title("Rewards with and without ERL experience", fontsize=16)
# plt.rcParams["font.weight"] = "bold"
# plt.rcParams["axes.labelweight"] = "bold"
# plt.show()


'''HalfCheetah Environment Plots'''
pickle_off = open("HalfCheetah_ERL_rewards_testing_LN.p", "rb")
emp = pickle.load(pickle_off)
ERL, = plt.plot(emp, label='ERL')

pickle_off = open("HalfCheetah_EVO_rewards_testing_LN_final.p", "rb")
emp = pickle.load(pickle_off)
EA, = plt.plot(emp, label="EA")

pickle_off = open("HalfCheetah_RLonly_rewards_testing_LN_final.p", "rb")
emp = pickle.load(pickle_off)
DDPG, = plt.plot(emp, label="DDPG")

plt.legend(handles=[ERL, EA, DDPG])
plt.xlabel("Million steps", fontsize=14)
plt.ylabel("Rewards", fontsize=14)
plt.title("HalfCheetah EA vs DDPG vs ERL", fontsize=16)
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.show()

