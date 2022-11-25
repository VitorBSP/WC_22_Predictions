import copy
import numpy as np 
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
from operator import itemgetter
team_stats_raw = pd.read_csv('data/team_stats_by_match.csv')

def analyze(model, y_train, y_test, X_train, X_test):
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test.values)[:,1]) #test AUC
    plt.figure(figsize=(15,10))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label="test")

    fpr_train, tpr_train, _ = roc_curve(y_train, model.predict_proba(X_train.values)[:,1]) #train AUC
    plt.plot(fpr_train, tpr_train, label="train")
    auc_test = roc_auc_score(y_test, model.predict_proba(X_test.values)[:,1])
    auc_train = roc_auc_score(y_train, model.predict_proba(X_train.values)[:,1])
    plt.legend()
    plt.title('AUC score is %.2f on test and %.2f on training'%(auc_test, auc_train))
    plt.show()
    
    plt.figure(figsize=(15, 10))
    cm = confusion_matrix(y_test, model.predict(X_test.values))
    sns.heatmap(cm, annot=True, fmt="d")

def scraping():
    dfs = pd.read_html(r"https://en.wikipedia.org/wiki/2022_FIFA_World_Cup#Teams")
    matches = []
    groups = ["A", "B", "C", "D", "E", "F", "G", "H"]
    group_count = 0 

    table = {}
    #TABLE -> TEAM, POINTS, WIN PROBS (CRITERIO DE DESEMPATE)
    table[groups[group_count]] = [[a.split(" ")[0], 0, []] for a in list(dfs[13].iloc[:, 1].values)]
    for i in range(14, 69, 1):
        if len(dfs[i].columns) == 3:
            team_1 = dfs[i].columns.values[0]
            team_2 = dfs[i].columns.values[-1]
            matches.append((groups[group_count], team_1, team_2))
        else:
            group_count+=1
            table[groups[group_count]] = [[a, 0, []] for a in list(dfs[i].iloc[:, 1].values)]
    table_rf = copy.deepcopy(table)
    table_rlog = copy.deepcopy(table)

    return matches, table, table_rf, table_rlog

def find_stats(team_1):
#team_1 = "Qatar"
    global team_stats_raw
    past_games = team_stats_raw[(team_stats_raw["team"] == team_1)].sort_values("date")
    last5 = team_stats_raw[(team_stats_raw["team"] == team_1)].sort_values("date").tail(5)

    team_1_rank = past_games["rank"].values[-1]
    team_1_goals = past_games['score'].mean()
    team_1_goals_l5 = last5['score'].mean()
    team_1_goals_suf = past_games['suf_score'].mean()
    team_1_goals_suf_l5 = last5['suf_score'].mean()
    team_1_rank_suf = past_games['rank_suf'].mean()
    team_1_rank_suf_l5 = last5['rank_suf'].mean()
    team_1_gp_rank = past_games['points_by_rank'].mean()
    team_1_gp_rank_l5 = last5['points_by_rank'].mean()
    team_1_points_by_classification = past_games['points_by_classification'].mean()

    return [team_1_rank, team_1_goals, team_1_goals_l5, team_1_goals_suf, team_1_goals_suf_l5, team_1_rank_suf, team_1_rank_suf_l5, team_1_gp_rank, team_1_gp_rank_l5, team_1_points_by_classification]

def find_features(team_1, team_2):
    rank_dif = team_1[0] - team_2[0]
    goals_dif = team_1[1] - team_2[1]
    goals_dif_l5 = team_1[2] - team_2[2]
    goals_suf_dif = team_1[3] - team_2[3]
    goals_suf_dif_l5 = team_1[4] - team_2[4]
    goals_per_ranking_dif = (team_1[1]/team_1[5]) - (team_2[1]/team_2[5])
    dif_rank_agst = team_1[5] - team_2[5]
    dif_rank_agst_l5 = team_1[6] - team_2[6]
    dif_gp_rank = team_1[7] - team_2[7]
    dif_gp_rank_l5 = team_1[8] - team_2[8]
    dif_points_by_classification = team_1[9] - team_2[9]
    
    return [rank_dif, goals_dif, goals_dif_l5, goals_suf_dif, goals_suf_dif_l5, goals_per_ranking_dif, dif_rank_agst, dif_rank_agst_l5, dif_gp_rank, dif_gp_rank_l5, 1, 0, dif_points_by_classification]

def simulation_groups(table, model, matches):
    advanced_group = []
    last_group = ""

    for k in table.keys():
        for t in table[k]:
            t[1] = 0
            t[2] = []
            
    for teams in matches:
        draw = False
        team_1 = find_stats(teams[1])
        team_2 = find_stats(teams[2])

        

        features_g1 = find_features(team_1, team_2)
        features_g2 = find_features(team_2, team_1)

        probs_g1 = model.predict_proba([features_g1])
        probs_g2 = model.predict_proba([features_g2])
        
        team_1_prob_g1 = probs_g1[0][0] # Probabilidade de vitória do time 1 em casa
        team_1_prob_g2 = probs_g2[0][1] # Probabilidade de vitória do time 1 fora de casa
        team_2_prob_g1 = probs_g1[0][1] # Probabilidade de vitória do time 2 fora de casa
        team_2_prob_g2 = probs_g2[0][0] # Probabilidade de vitória do time 2 em casa

        team_1_prob = (probs_g1[0][0] + probs_g2[0][1])/2*100
        team_2_prob = (probs_g2[0][0] + probs_g1[0][1])/2*100
        
        if ((team_1_prob_g1 > team_2_prob_g1) & (team_2_prob_g2 > team_1_prob_g2)) | ((team_1_prob_g1 < team_2_prob_g1) & (team_2_prob_g2 < team_1_prob_g2)):
            draw=True
            for i in table[teams[0]]:
                if i[0] == teams[1] or i[0] == teams[2]:
                    i[1] += 1
                    
        elif team_1_prob > team_2_prob:
            winner = teams[1]
            winner_proba = team_1_prob
            for i in table[teams[0]]:
                if i[0] == teams[1]:
                    i[1] += 3
                    
        elif team_2_prob > team_1_prob:  
            winner = teams[2]
            winner_proba = team_2_prob
            for i in table[teams[0]]:
                if i[0] == teams[2]:
                    i[1] += 3
        
        for i in table[teams[0]]: #adding criterio de desempate (probs por jogo)
                if i[0] == teams[1]:
                    i[2].append(team_1_prob)
                if i[0] == teams[2]:
                    i[2].append(team_2_prob)

        if last_group != teams[0]:
            if last_group != "":
                print("\n")
                print("Grupo %s classificados: "%(last_group))
                
                for i in table[last_group]: #adding crieterio de desempate
                    i[2] = np.mean(i[2])
                
                final_points = table[last_group]
                final_table = sorted(final_points, key=itemgetter(1, 2), reverse = True)
                advanced_group.append([final_table[0][0], final_table[1][0]])
                for i in final_table:
                    print("%s -------- %d"%(i[0], i[1]))
            print("\n")
            print("-"*10+" Começando a simulação para o Grupo %s "%(teams[0])+"-"*10)
            
            
        if draw == False:
            print("Grupo %s - %s x %s: vencedor %s com %.2f probabilidade"%(teams[0], teams[1], teams[2], winner, winner_proba))
        else:
            print("Grupo %s - %s x %s: Empate"%(teams[0], teams[1], teams[2]))
        last_group =  teams[0]

    print("\n")
    print("Group %s advanced: "%(last_group))

    for i in table[last_group]: #adding crieterio de desempate
        i[2] = np.mean(i[2])
                
    final_points = table[last_group]
    final_table = sorted(final_points, key=itemgetter(1, 2), reverse = True)
    advanced_group.append([final_table[0][0], final_table[1][0]])
    for i in final_table:
        print("%s -------- %d"%(i[0], i[1]))

    return advanced_group


def simulation_playoff(advanced, model):
    playoffs = {"Oitavas de Final": [], "Quartas de Final": [], "Semi-Final": [], "Final": []}
    for p in playoffs.keys():
        playoffs[p] = []
    actual_round = ""
    next_rounds = []

    for p in playoffs.keys():
        if p == "Oitavas de Final":
            control = []
            for a in range(0, len(advanced*2), 1):
                if a < len(advanced):
                    if a % 2 == 0:
                        control.append((advanced*2)[a][0])
                    else:
                        control.append((advanced*2)[a][1])
                else:
                    if a % 2 == 0:
                        control.append((advanced*2)[a][1])
                    else:
                        control.append((advanced*2)[a][0])

            playoffs[p] = [[control[c], control[c+1]] for c in range(0, len(control)-1, 1) if c%2 == 0]
            
            for i in range(0, len(playoffs[p]), 1):
                game = playoffs[p][i]
                
                home = game[0]
                away = game[1]
                team_1 = find_stats(home)
                team_2 = find_stats(away)

                features_g1 = find_features(team_1, team_2)
                features_g2 = find_features(team_2, team_1)
                
                probs_g1 = model.predict_proba([features_g1])
                probs_g2 = model.predict_proba([features_g2])
                
                team_1_prob = (probs_g1[0][0] + probs_g2[0][1])/2*100
                team_2_prob = (probs_g2[0][0] + probs_g1[0][1])/2*100
                
                if actual_round != p:
                    print("-"*10)
                    print("Começando a simulação de %s"%(p))
                    print("-"*10)
                    print("\n")
                
                if team_1_prob < team_2_prob:
                    print("%s x %s: %s avança com probabilidade %.2f"%(home, away, away, team_2_prob))
                    next_rounds.append(away)
                else:
                    print("%s x %s: %s avança com probabilidade %.2f"%(home, away, home, team_1_prob))
                    next_rounds.append(home)
                
                game.append([team_1_prob, team_2_prob])
                playoffs[p][i] = game
                actual_round = p
            
        else:
            playoffs[p] = [[next_rounds[c], next_rounds[c+1]] for c in range(0, len(next_rounds)-1, 1) if c%2 == 0]
            next_rounds = []
            for i in range(0, len(playoffs[p])):
                game = playoffs[p][i]
                home = game[0]
                away = game[1]
                team_1 = find_stats(home)
                team_2 = find_stats(away)
                
                features_g1 = find_features(team_1, team_2)
                features_g2 = find_features(team_2, team_1)
                
                probs_g1 = model.predict_proba([features_g1])
                probs_g2 = model.predict_proba([features_g2])
                
                team_1_prob = (probs_g1[0][0] + probs_g2[0][1])/2*100
                team_2_prob = (probs_g2[0][0] + probs_g1[0][1])/2*100
                
                if actual_round != p:
                    print("-"*10)
                    print("Começando a simulação de  %s"%(p))
                    print("-"*10)
                    print("\n")
                
                if team_1_prob < team_2_prob:
                    print("%s vs. %s: %s avança com probabilidade %.2f"%(home, away, away, team_2_prob))
                    next_rounds.append(away)
                else:
                    print("%s vs. %s: %s avança com probabilidade %.2f"%(home, away, home, team_1_prob))
                    next_rounds.append(home)
                game.append([team_1_prob, team_2_prob])
                playoffs[p][i] = game
                actual_round = p
    return playoffs
                