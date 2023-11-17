import tensorflow as tf
from nn import linearND, linear
from mol_graph_direct_useScores import atom_fdim as adim, bond_fdim as bdim, max_nb, smiles2graph, smiles2graph, bond_types
from models import *
import math, sys, random
from optparse import OptionParser
import threading
from multiprocessing import Queue
import rdkit
from rdkit import Chem
import os
import numpy as np 

'''
This module defines the DirectCandRanker class, which is for deploying the candidate ranking model
'''

TOPK = 100
hidden_size = 500
depth = 3
core_size = 16
MAX_NCAND = 1500
model_path = os.path.join(os.path.dirname(__file__), "model-core16-500-3-max150-direct-useScores", "model.ckpt-2400000")

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def stereo_remove_and_canonicalize(smiles_string):
    mol = Chem.MolFromSmiles(smiles_string)
    Chem.RemoveStereochemistry(mol)
    smiles = Chem.MolToSmiles(mol)
    smiles = Chem.CanonSmiles(smiles)
    return(smiles)

class DirectCandRanker():
    def __init__(self, hidden_size=hidden_size, depth=depth, core_size=core_size,
            MAX_NCAND=MAX_NCAND, TOPK=TOPK):
        self.hidden_size = hidden_size 
        self.depth = depth 
        self.core_size = core_size 
        self.MAX_NCAND = MAX_NCAND 
        self.TOPK = TOPK 

    def load_model(self, model_path=model_path):
        hidden_size = self.hidden_size 
        depth = self.depth 
        core_size = self.core_size 
        MAX_NCAND = self.MAX_NCAND 
        TOPK = self.TOPK 

        self.graph = tf.Graph()
        with self.graph.as_default():
            input_atom = tf.placeholder(tf.float32, [None, None, adim])
            input_bond = tf.placeholder(tf.float32, [None, None, bdim])
            atom_graph = tf.placeholder(tf.int32, [None, None, max_nb, 2])
            bond_graph = tf.placeholder(tf.int32, [None, None, max_nb, 2])
            num_nbs = tf.placeholder(tf.int32, [None, None])
            core_bias = tf.placeholder(tf.float32, [None])
            self.src_holder = [input_atom, input_bond, atom_graph, bond_graph, num_nbs, core_bias]

            graph_inputs = (input_atom, input_bond, atom_graph, bond_graph, num_nbs) 
            with tf.variable_scope("mol_encoder"):
                fp_all_atoms = rcnn_wl_only(graph_inputs, hidden_size=hidden_size, depth=depth)

            reactant = fp_all_atoms[0:1,:]
            candidates = fp_all_atoms[1:,:]
            candidates = candidates - reactant
            candidates = tf.concat([reactant, candidates], 0)

            with tf.variable_scope("diff_encoder"):
                reaction_fp = wl_diff_net(graph_inputs, candidates, hidden_size=hidden_size, depth=1)

            reaction_fp = reaction_fp[1:]
            reaction_fp = tf.nn.relu(linear(reaction_fp, hidden_size, "rex_hidden"))

            score = tf.squeeze(linear(reaction_fp, 1, "score"), [1]) + core_bias # add in bias from CoreFinder
            scaled_score = tf.nn.softmax(score)

            tk = tf.minimum(TOPK, tf.shape(score)[0])
            _, pred_topk = tf.nn.top_k(score, tk)
            self.predict_vars = [score, scaled_score, pred_topk]

            self.session = tf.Session()
            saver = tf.train.Saver()
            saver.restore(self.session, model_path)
    
    def predict(self, react, top_cand_bonds, top_cand_scores=[], scores=True, top_n=100):
        '''react: atom mapped reactant smiles
        top_cand_bonds: list of strings "ai-aj-bo"'''

        cand_bonds = []
        if not top_cand_scores:
            top_cand_scores = [0.0 for b in top_cand_bonds]
        for i, b in enumerate(top_cand_bonds):
            x,y,t = b.split('-')
            x,y,t = int(float(x))-1,int(float(y))-1,float(t)

            cand_bonds.append((x,y,t,float(top_cand_scores[i])))

        while True:
            src_tuple,conf = smiles2graph(react, None, cand_bonds, None, core_size=core_size, cutoff=MAX_NCAND, testing=True)
            if len(conf) <= MAX_NCAND:
                break
            ncore -= 1

        feed_map = {x:y for x,y in zip(self.src_holder, src_tuple)}
        cur_scores, cur_probs, candidates = self.session.run(self.predict_vars, feed_dict=feed_map)
        

        idxfunc = lambda a: a.GetAtomMapNum()
        bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
                      Chem.rdchem.BondType.AROMATIC]
        bond_types_as_double = {0.0: 0, 1.0: 1, 2.0: 2, 3.0: 3, 1.5: 4}

        # Don't waste predictions on bond changes that aren't actually changes
        rmol = Chem.MolFromSmiles(react)
        rbonds = {}
        for bond in rmol.GetBonds():
            a1 = idxfunc(bond.GetBeginAtom())
            a2 = idxfunc(bond.GetEndAtom())
            t = bond_types.index(bond.GetBondType()) + 1
            a1,a2 = min(a1,a2),max(a1,a2)
            rbonds[(a1,a2)] = t

        cand_smiles = []; cand_scores = []; cand_probs = [];
        for idx in candidates:
            cbonds = []
            # Define edits from prediction
            for x,y,t,v in conf[idx]:
                x,y = x+1,y+1
                if ((x,y) not in rbonds and t > 0) or ((x,y) in rbonds and rbonds[(x,y)] != t):
                    cbonds.append((x, y, bond_types_as_double[t]))
            pred_smiles = edit_mol(rmol, cbonds)
            cand_smiles.append(pred_smiles)
            cand_scores.append(cur_scores[idx])
            cand_probs.append(cur_probs[idx])

        outcomes = []
        if scores:
            for i in range(min(len(cand_smiles), top_n)):
                outcomes.append({
                    'rank': i + 1,
                    'smiles': cand_smiles[i],
                    'score': cand_scores[i],
                    'prob': cand_probs[i],
                })
        else:
            for i in range(min(len(cand_smiles), top_n)):
                outcomes.append({
                    'rank': i + 1,
                    'smiles': cand_smiles[i],
                })

        return outcomes

if __name__ == '__main__':

    #sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    if(sys.path[-1] != "/content/ConnorColey_rexgen_direct_2023"):
        sys.path.append("/content/ConnorColey_rexgen_direct_2023")
    #print(sys.path)

    from rexgen_direct.core_wln_global.directcorefinder import DirectCoreFinder 
    from rexgen_direct.scripts.eval_by_smiles import edit_mol
    import pandas as pd
    from rdkit import Chem

    directcorefinder = DirectCoreFinder()
    directcorefinder.load_model()
    if len(sys.argv) < 2:
        #print the example reactant
        print("Using example reaction")
        print("----------")
        react = '[CH3:26][c:27]1[cH:28][cH:29][cH:30][cH:31][cH:32]1.[Cl:18][C:19](=[O:20])[O:21][C:22]([Cl:23])([Cl:24])[Cl:25].[NH2:1][c:2]1[cH:3][cH:4][c:5]([Br:17])[c:6]2[c:10]1[O:9][C:8]([CH3:11])([C:12](=[O:13])[O:14][CH2:15][CH3:16])[CH2:7]2'
        react = "CCOC(=O)C1=CC=C[C@@H](NC(=O)OC(C)(C)C)C1"
        react = stereo_remove_and_canonicalize(react)
        print(react)
        print("----------")
        #get the score for the example reactant
        (react, bond_preds, bond_scores, cur_att_score) = directcorefinder.predict(react)
        directcandranker = DirectCandRanker()
        directcandranker.load_model()
        #get the predictions for the products
        outcomes = directcandranker.predict(react, bond_preds, bond_scores)
        #print the ranked products
        for outcome in outcomes:
            print(outcome)

    #expect a path to a csv file, passed as a parameter, to test a list of reactants against expected products
    else:
        #get the data from the csv and save it as a dataframe
        csv_path = str(sys.argv[1])
        test_dataframe = pd.read_csv(csv_path, names=["reactants", "products"])
        #Create a csv to add to
        csv_output = open("output_test.csv", 'w')
        csv_output.write("Reactants,Expected_Products,Predicted_Products,Rank,Comparison,Model_Probability,\n")
        #loop through each row in the csv compare the reditions of the program
        for i in range(len(test_dataframe)):
            try:
                #for each reactants in the row
                reactants = test_dataframe["reactants"][i]
                #get the predictions for those reactants
                (labelled_reactants, bond_preds, bond_scores, cur_att_score) = directcorefinder.predict(reactants)
                directcandranker = DirectCandRanker()
                directcandranker.load_model()
                outcomes = directcandranker.predict(labelled_reactants, bond_preds, bond_scores)
                #Sanitize reactants for canonicallization
                reactants = stereo_remove_and_canonicalize(reactants)
                
                #Sanitize expected product for canonicallization
                expected_product = stereo_remove_and_canonicalize(test_dataframe["products"][i])
                
                #make comparison between expected and predicted product
                correct_prediction = False
                for j in range(len(outcomes)):
                    predicted_products = ""
                    #iterate through each smiles string in the predicted list (checking for match to the expected product)
                    for predicted_product in outcomes[j]["smiles"]:
                        #loop through each SMILES in the prediction output
                        try:
                            predicted_product = stereo_remove_and_canonicalize(predicted_product)
                        except:
                            print("{} cannot be properly parsed; skip sanitization and assume correct format".format(predicted_product))
                        #add smiles to full list of products
                        predicted_products += predicted_product + "."
                        #make comparison between expected and predicted SMILES
                        if(predicted_product == expected_product):
                            correct_prediction = True
                    #remove period at end of smiles
                    predicted_products = predicted_products[0:len(predicted_products)-1]
                    #save the predicted probability of the row
                    prediction_probability = outcomes[j]["prob"]
                    #save the rank of the prediction
                    rank = outcomes[j]["rank"]
                    #Write the data to a csv for rank 1 and correct product
                    if(rank==1):
                        csv_output = open("output_test.csv", 'a')
                        row = "{},{},{},{},{},{},\n".format(reactants, expected_product, predicted_products, rank, correct_prediction, prediction_probability)
                        csv_output.write(row)
                        csv_output.close()
                    if(correct_prediction):
                        csv_output = open("output_test.csv", 'a')
                        row = "{},{},{},{},{},{},\n".format(reactants, expected_product, predicted_products, rank, correct_prediction, prediction_probability)
                        csv_output.write(row)
                        csv_output.close()
                        #break out of the loop for the current reactant if a correct prediction detected
                        break
                    #when about the finish the final loop, then we know that the correct prediction has not been found
                    if(j == len(outcomes)-1):
                        csv_output = open("output_test.csv", 'a')
                        row = "{},{},NONE_FOUND,{}_PREDICTIONS_CHECKED,NONE_FOUND,NONE_FOUND,\n".format(reactants, expected_product, len(outcomes))
                        csv_output.write(row)
                        csv_output.close()
            except:
                #skip the test if there is something wrong with gathering the smiles
                reactants = test_dataframe["reactants"][i]
                csv_output = open("output_test.csv", 'a')
                row = "{},ERROR,ERROR,ERROR,ERROR,ERROR,\n".format(reactants)
                csv_output.write(row)
                csv_output.close()

